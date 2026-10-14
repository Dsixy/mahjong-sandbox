"""Local HTTP game/replay adapter. Only projected observations cross the API."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, unquote, urlparse

from .agents import RandomAgent
from .engine import Engine, IllegalAction
from .greedy import GreedyAgent
from .model import Action, ActionKind, Phase, RuleConfig
from .replay import export_replay, replay, replay_engine
from .riichi_engine import RiichiEngine
from .riichi_model import RiichiConfig
from .riichi_agent import RiichiAgent
from .riichi_match import RiichiMatch, MatchConfig, replay_match

WEB_ROOT = Path(__file__).with_name("web")
NAMES = ("你", "阿青", "竹影", "阿川")


def projected_view(engine, session_id, *, mode="play", reveal=False):
    observation = engine.observe(0)
    if reveal or engine.state.phase == Phase.TERMINAL:
        for seat, hand in enumerate(engine.state.players):
            observation["players"][seat]["concealed"] = list(hand.concealed)
    observation["pending_tile"] = engine.state.pending_tile
    observation["settlement"] = deepcopy(engine.state.settlement)
    return {"id": session_id, "mode": mode, "observation": observation,
            "names": list(NAMES), "revision": engine.state.revision}


class GameSession:
    def __init__(self, *, seed=None, policy="greedy", exchange=True, auto_pass=True, variant="sichuan", match="hand", red=True, tanyao=True):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**53):
            raise ValueError("seed must be a nonnegative integer below 2^53")
        if policy not in ("greedy", "win-first", "random"):
            raise ValueError("unknown AI policy")
        if any(type(x) is not bool for x in (exchange, auto_pass, red, tanyao)):
            raise ValueError("settings must be boolean")
        if variant not in ('sichuan', 'riichi') or match not in ('hand', 'east', 'south') or (variant == 'sichuan' and match != 'hand'):
            raise ValueError('unsupported variant / match')
        self.id = secrets.token_urlsafe(18)
        seed = secrets.randbelow(2**32) if seed is None else seed
        self.variant, self.match = variant, None
        if variant == 'riichi':
            rules = RiichiConfig(red_fives=(16,52,88) if red else (), open_tanyao=tanyao)
            if match != 'hand': self.match = RiichiMatch(MatchConfig(length=match), rules, seed=seed)
            self.engine = self.match.engine if self.match else RiichiEngine(rules, seed=seed)
        else:
            self.engine = Engine(RuleConfig(exchange_three=exchange), seed)
        self.policy, self.auto_pass = policy, auto_pass
        self.agents = [(RiichiAgent if variant == "riichi" else GreedyAgent)(self.engine.seed * 4 + p) if policy == "greedy"
                       else RandomAgent(self.engine.seed * 4 + p, prefer_win=policy == "win-first") for p in range(4)]
        self.lock = threading.RLock()
        self._advance()

    def view(self):
        result = projected_view(self.engine, self.id)
        result["settings"] = {"policy": self.policy, "auto_pass": self.auto_pass,
                              "exchange": getattr(self.engine.config, "exchange_three", False), "variant": self.variant}
        if self.match:
            if self.engine.state.phase == Phase.TERMINAL: self.match.complete_hand()
            result["match"] = self.match.summary()
        return result

    def next_hand(self, body):
        with self.lock:
            if not self.match or set(body) != {'decision_id', 'hands_played'}:
                raise ValueError('next hand requires a match and current decision')
            if type(body['decision_id']) is not int or type(body['hands_played']) is not int:
                raise ValueError('decision and hand count must be integers')
            if body['decision_id'] != self.engine.state.decision_id or body['hands_played'] != len(self.match.rounds):
                raise IllegalAction('stale next hand request')
            self.engine = self.match.next_hand()
            self._advance()
            return self.view()

    def export(self, match=False):
        if match:
            if not self.match: raise ValueError('this game has no match')
            self.match.complete_hand()
            return self.match.export()
        if self.engine.state.phase != Phase.TERMINAL:
            raise ValueError('live games cannot reveal a complete replay')
        return export_replay(self.engine)

    def _advance(self):
        frames = []
        for _ in range(2000):
            if self.engine.state.phase == Phase.TERMINAL:
                return frames
            player = self.engine.next_player()
            actions = self.engine.legal_actions(player)
            if player == 0:
                if not self.auto_pass or actions != (Action(ActionKind.PASS),):
                    return frames
                action = actions[0]
            else:
                action = self.agents[player].act(self.engine.observe(player))
            transition = self.engine.submit(player, action, self.engine.state.decision_id)
            if transition.events:
                frames.append(self.view())
        raise RuntimeError("automatic turns exceeded the round limit")

    def submit(self, body):
        with self.lock:
            if set(body) != {"decision_id", "action"}:
                raise ValueError("action request requires decision_id and action")
            transition = self.engine.submit(0, Action.from_dict(body["action"]), body["decision_id"])
            frames = [self.view()] if transition.events else []
            frames.extend(self._advance())
            return {"view": self.view(), "frames": frames}


class ReplaySession:
    def __init__(self, data):
        if not isinstance(data, dict): raise ValueError('invalid replay')
        if data.get('format_version') == 2:
            replay_match(data)
            records = data['rounds']
        else:
            replay(data)
            records = [data]
        if any(len(r['commands']) > 2000 for r in records) or sum(len(r['commands']) for r in records) > 32000:
            raise ValueError('oversized replay')
        self.id = secrets.token_urlsafe(18)
        self.data = deepcopy(data)
        self.steps, self.hands = [], []
        for number, record in enumerate(records):
            if number: self.steps.append((record, None))
            self.hands.append({'start': len(self.steps), 'label': f"{'东南西北'[record['config'].get('round_wind', 27)-27]}{record['config'].get('hand_number', 1)}局 {record['config'].get('honba', 0)}本场"})
            self.steps.extend((None, c) for c in record['commands'])
        self.total = len(self.steps)
        self.lock = threading.RLock()
        engine = replay_engine(records[0])
        self.checkpoints = {0: deepcopy(engine)}
        for index, (record, command) in enumerate(self.steps, 1):
            engine = self._step(engine, record, command)
            if index % 20 == 0 or index == self.total:
                self.checkpoints[index] = deepcopy(engine)

    @staticmethod
    def _step(engine, record, command):
        if record is not None: return replay_engine(record)
        engine.submit(command['player'], Action.from_dict(command['action']), command['decision_id'])
        return engine

    def view(self, step=0, reveal=False):
        if type(step) is not int or not 0 <= step <= self.total:
            raise ValueError('replay step is outside the record')
        checkpoint = max(index for index in self.checkpoints if index <= step)
        engine = deepcopy(self.checkpoints[checkpoint])
        for record, command in self.steps[checkpoint:step]:
            engine = self._step(engine, record, command)
        result = projected_view(engine, self.id, mode='replay', reveal=reveal)
        result['observation']['legal_actions'] = []
        if self.data.get('format_version') == 2 and step == self.total:
            result['match'] = deepcopy(self.data['expected'])
        result['replay'] = {'step': step, 'total': self.total, 'reveal': reveal, 'verified': True, 'hands': self.hands}
        return result


class SessionStore:
    def __init__(self):
        self.games, self.replays = {}, {}
        self.lock = threading.RLock()

    def create_game(self, body):
        if set(body) - {"seed", "policy", "exchange", "auto_pass", "variant", "match", "red", "tanyao"}:
            raise ValueError("unknown game setting")
        session = GameSession(**body)
        with self.lock:
            if len(self.games) >= 32:
                self.games.pop(next(iter(self.games)))
            self.games[session.id] = session
        return session

    def create_replay(self, body):
        if set(body) == {"example"} and body['example'] in ('three_winners', 'riichi'):
            data = json.loads((WEB_ROOT / f"assets/{body['example']}.json").read_text(encoding='utf-8'))
        elif set(body) == {"game_id"}:
            game = self.get_game(body["game_id"])
            with game.lock:
                if game.engine.state.phase != Phase.TERMINAL:
                    raise ValueError("live games cannot reveal a complete replay")
                data = export_replay(game.engine)
        elif set(body) == {"replay"}:
            data = body["replay"]
        else:
            raise ValueError("provide a replay, a finished game, or the example")
        session = ReplaySession(data)
        with self.lock:
            if len(self.replays) >= 8:
                self.replays.pop(next(iter(self.replays)))
            self.replays[session.id] = session
        return session

    def get_game(self, session_id):
        with self.lock:
            if session_id not in self.games:
                raise KeyError("牌桌已关闭，请重新开局")
            return self.games[session_id]

    def get_replay(self, session_id):
        with self.lock:
            if session_id not in self.replays:
                raise KeyError("牌谱已关闭，请重新导入")
            return self.replays[session_id]


def make_server(host="127.0.0.1", port=8765):
    store = SessionStore()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, data, status=200, *, filename=None):
            encoded = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(encoded)

        def _error(self, error, status=400):
            self._json({"error": str(error)}, status)

        def do_GET(self):
            parsed = urlparse(self.path)
            parts = parsed.path.strip("/").split("/")
            try:
                if parts == ["api", "health"]:
                    return self._json({"status": "ok"})
                if len(parts) == 3 and parts[:2] == ["api", "games"]:
                    game = store.get_game(parts[2])
                    with game.lock:
                        return self._json(game.view())
                if len(parts) == 4 and parts[:2] == ["api", "games"] and parts[3] == "replay":
                    game = store.get_game(parts[2])
                    with game.lock:
                        if game.engine.state.phase != Phase.TERMINAL:
                            raise ValueError("牌局结束后才能下载完整牌谱")
                        return self._json(game.export(match=parse_qs(parsed.query).get("match", ["false"])[0] == "true"), filename=f"mahjong_{parts[2]}.json")
                if len(parts) == 3 and parts[:2] == ["api", "replays"]:
                    session = store.get_replay(parts[2])
                    query = parse_qs(parsed.query)
                    step = int(query.get("step", ["0"])[0])
                    reveal = query.get("reveal", ["false"])[0] == "true"
                    with session.lock:
                        return self._json(session.view(step, reveal))
                if parsed.path.startswith("/api/"):
                    return self._error("接口不存在", 404)
                relative = unquote(parsed.path).lstrip("/") or "index.html"
                path = (WEB_ROOT / relative).resolve()
                if not path.is_relative_to(WEB_ROOT.resolve()) or not path.is_file():
                    return self._error("资源不存在", 404)
                content = path.read_bytes()
                self.send_response(200)
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(content)
            except KeyError as error:
                self._error(error.args[0], 404)
            except (ValueError, TypeError) as error:
                self._error(error)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    return self._error("请求来源不匹配", 403)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2 * 1024 * 1024:
                    return self._error("请求内容为空或超过 2 MiB", 413)
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("请求必须是 JSON 对象")
                parts = urlparse(self.path).path.strip("/").split("/")
                if parts == ["api", "games"]:
                    return self._json(store.create_game(body).view(), 201)
                if len(parts) == 4 and parts[:2] == ["api", "games"] and parts[3] == "action":
                    return self._json(store.get_game(parts[2]).submit(body))
                if len(parts) == 4 and parts[:2] == ["api", "games"] and parts[3] == "next":
                    return self._json(store.get_game(parts[2]).next_hand(body))
                if parts == ["api", "replays"]:
                    return self._json(store.create_replay(body).view(), 201)
                return self._error("接口不存在", 404)
            except IllegalAction as error:
                self._error(f"操作已过期或不合法：{error}", 409)
            except KeyError as error:
                self._error(error.args[0], 404)
            except (ValueError, TypeError) as error:
                self._error(error)

        def log_message(self, format, *args):
            if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
                super().log_message(format, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.store = store
    return server


def serve(host="127.0.0.1", port=8765):
    server = make_server(host, port)
    actual_port = server.server_address[1]
    print(f"川流麻将：http://{host}:{actual_port}（Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
