"""Game transport, observation privacy, and deterministic replay navigation."""
from copy import deepcopy
import http.client
import json
import threading
import unittest
from xml.etree import ElementTree

from mahjong_sandbox import Action, ActionKind as K, IllegalAction, Phase
from mahjong_sandbox.greedy import GreedyAgent, shape_distance
from mahjong_sandbox.replay import export_replay
from mahjong_sandbox.tiles import parse_hand
from mahjong_sandbox.webserver import GameSession, ReplaySession, SessionStore, WEB_ROOT, make_server
from tests.helpers import scenario


def play(session, seed=10):
    human = GreedyAgent(seed)
    for _ in range(500):
        if session.engine.state.phase == Phase.TERMINAL:
            return session
        observation = session.view()["observation"]
        action = human.act(observation)
        session.submit({"decision_id": observation["decision_id"], "action": action.to_dict()})
    raise AssertionError("game did not terminate")


class BaselineTests(unittest.TestCase):
    def test_structural_distance_normal_pairs_and_open_hand(self):
        for hand, melds, expected in (("123456789m12344p", 0, -1), ("123456789m1234p", 0, 0),
                                      ("11112233445566m", 0, -1), ("123m44p", 3, -1)):
            with self.subTest(hand=hand):
                self.assertEqual(shape_distance(parse_hand(hand), melds), expected)
        self.assertEqual(shape_distance(parse_hand("11112233556699m")), -1)
        self.assertGreaterEqual(shape_distance(parse_hand("11112233556699m"), seven_pairs_quads=False), 0)

    def test_agent_chooses_legal_actions_without_mutating_observation(self):
        session = GameSession(seed=44)
        human = GreedyAgent(123)
        for _ in range(100):
            if session.engine.state.phase == Phase.TERMINAL:
                break
            obs = session.view()["observation"]
            before = deepcopy(obs)
            action = human.act(obs)
            self.assertEqual(obs, before)
            self.assertIn(action.to_dict(), obs["legal_actions"])
            session.submit({"decision_id": obs["decision_id"], "action": action.to_dict()})
        self.assertEqual(session.engine.state.phase, Phase.TERMINAL)


class SessionTests(unittest.TestCase):
    def test_live_views_and_every_animation_frame_hide_opponent_hands(self):
        session = GameSession(seed=44)
        human = GreedyAgent(33)
        for _ in range(100):
            view = session.view()
            self.assertNotIn("wall", view["observation"])
            if session.engine.state.phase == Phase.TERMINAL:
                self.assertTrue(all("concealed" in p for p in view["observation"]["players"]))
                break
            self.assertEqual(session.engine.next_player(), 0)
            self.assertFalse(any("concealed" in p for p in view["observation"]["players"][1:] if p["can_act"]))
            action = human.act(view["observation"])
            result = session.submit({"decision_id": view["observation"]["decision_id"], "action": action.to_dict()})
            for frame in result["frames"]:
                obs = frame["observation"]
                if obs["phase"] != "terminal":
                    self.assertFalse(any("concealed" in p for p in obs["players"][1:] if p["can_act"]))
                for event in obs["events"]:
                    self.assertNotIn("private", event)
                    if event["kind"] == "draw" and event["data"]["player"] != 0:
                        self.assertNotIn("tile", event["data"])
        self.assertEqual(session.engine.state.phase, Phase.TERMINAL)

    def test_stale_duplicate_submission_is_side_effect_free(self):
        session = GameSession(seed=3)
        obs = session.view()["observation"]
        body = {"decision_id": obs["decision_id"], "action": obs["legal_actions"][0]}
        session.submit(body)
        before = session.engine.state_hash()
        with self.assertRaises(IllegalAction):
            session.submit(body)
        self.assertEqual(session.engine.state_hash(), before)

    def test_auto_pass_option_and_human_win_continue_play(self):
        session = GameSession(seed=3, auto_pass=False)
        session.engine = scenario({0: "4p"}, actor=1)
        discard = next(a for a in session.engine.legal_actions(1) if a.kind == K.DISCARD)
        session.engine.submit(1, discard, session.engine.state.decision_id)
        session._advance()
        self.assertEqual(session.engine.next_player(), 0)
        self.assertEqual(session.engine.legal_actions(0), (Action(K.PASS),))
        session = GameSession(seed=3)
        session.engine = scenario({0: "123456789m12344p"})
        session.submit({"decision_id": session.engine.state.decision_id, "action": Action(K.TSUMO).to_dict()})
        self.assertEqual(session.engine.state.phase, Phase.TERMINAL)
        self.assertFalse(session.engine.state.players[0].can_act)

    def test_mixed_policies_complete_real_rounds_and_verify(self):
        for seed in range(12):
            policy = ("greedy", "win-first", "random")[seed % 3]
            session = play(GameSession(seed=seed, policy=policy, exchange=bool(seed % 2), auto_pass=bool(seed % 3)))
            session.engine.assert_invariants()
            self.assertEqual(sum(session.engine.state.scores), 0)
            replay = ReplaySession(export_replay(session.engine))
            self.assertEqual(replay.view(replay.total)["observation"]["scores"], session.engine.state.scores)

    def test_live_replay_cannot_reveal_private_tiles(self):
        store = SessionStore()
        session = store.create_game({"seed": 44})
        with self.assertRaises(ValueError):
            store.create_replay({"game_id": session.id})
        with self.assertRaises(ValueError):
            store.create_game({"reveal": True})
        for settings in ({"seed": True}, {"seed": 2**53}, {"policy": "oracle"}, {"exchange": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                GameSession(**settings)


class ReplayNavigationTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((WEB_ROOT / "assets/three_winners.json").read_text())

    def test_forward_backward_and_view_switch_are_deterministic(self):
        session = ReplaySession(self.data)
        opening = session.view()
        self.assertFalse(any("concealed" in p for p in opening["observation"]["players"][1:]))
        self.assertEqual(opening["observation"]["legal_actions"], [])
        complete = session.view(12)
        self.assertEqual(complete["observation"]["scores"], [-3, 1, 1, 1])
        self.assertEqual(session.view(0), opening)
        all_hands = session.view(3, reveal=True)
        self.assertTrue(all("concealed" in p for p in all_hands["observation"]["players"]))
        self.assertNotIn("wall", all_hands["observation"])
        self.assertEqual(session.view(12), complete)
        for step in (-1, 13, True):
            with self.assertRaises(ValueError):
                session.view(step)

    def test_tampered_and_malformed_records_are_rejected_before_display(self):
        tampered = deepcopy(self.data)
        tampered["expected"]["scores"][0] = 100
        for data in (tampered, {"commands": []}, {"commands": [{}] * 2001}, []):
            with self.subTest(data_type=type(data)), self.assertRaises(ValueError):
                ReplaySession(data)

    def test_all_original_svg_assets_are_valid_and_packaged(self):
        paths = list((WEB_ROOT / "assets").rglob("*.svg"))
        self.assertEqual(len(paths), 43)
        for path in paths:
            root = ElementTree.parse(path).getroot()
            self.assertTrue(root.tag.endswith("svg"))
        self.assertTrue((WEB_ROOT / "app.js").is_file())


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        encoded = json.dumps(body) if body is not None else None
        conn.request(method, path, encoded, headers or {})
        response = conn.getresponse()
        result = (response.status, dict(response.getheaders()), response.read())
        conn.close()
        return result

    def test_assets_and_path_traversal(self):
        for path, mime in (("/", "text/html"), ("/app.js", "javascript"),
                           ("/assets/tiles/0.svg", "image/svg+xml"), ("/api/health", "application/json")):
            status, headers, content = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertIn(mime, headers["Content-Type"])
            self.assertTrue(content)
        for path in ("/../../pyproject.toml", "/%2e%2e/engine.py", "/api/missing"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_action_transport_and_live_export_rejection(self):
        status, _, content = self.request("POST", "/api/games", {"seed": 44})
        self.assertEqual(status, 201)
        view = json.loads(content)
        game_id, obs = view["id"], view["observation"]
        self.assertEqual(self.request("GET", f"/api/games/{game_id}/replay")[0], 400)
        self.assertEqual(self.request("POST", "/api/replays", {"game_id": game_id})[0], 400)
        body = {"decision_id": obs["decision_id"], "action": obs["legal_actions"][0]}
        self.assertEqual(self.request("POST", f"/api/games/{game_id}/action", body)[0], 200)
        self.assertEqual(self.request("POST", f"/api/games/{game_id}/action", body)[0], 409)
        play(self.server.store.get_game(game_id))
        status, headers, content = self.request("GET", f"/api/games/{game_id}/replay")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        status, _, content = self.request("POST", "/api/replays", {"replay": json.loads(content)})
        self.assertEqual(status, 201)
        replay_id = json.loads(content)["id"]
        status, _, content = self.request("GET", f"/api/replays/{replay_id}?step=0&reveal=true")
        self.assertEqual(status, 200)
        self.assertTrue(all("concealed" in p for p in json.loads(content)["observation"]["players"]))

    def test_riichi_match_transport_and_json_replay(self):
        from tests.test_riichi_web import finish
        from mahjong_sandbox import RiichiAgent
        status, _, content = self.request('POST', '/api/games', {'seed':21, 'variant':'riichi', 'match':'east'})
        self.assertEqual(status,201)
        game_id=json.loads(content)['id'];session=self.server.store.get_game(game_id);agent=RiichiAgent(0)
        finish(session,agent)
        v=session.view();body={'decision_id':v['observation']['decision_id'],'hands_played':v['match']['hands_played']}
        status,_,content=self.request('GET',f'/api/games/{game_id}/replay')
        self.assertEqual(status,200)
        status,_,_=self.request('POST','/api/replays',{'replay':json.loads(content)})
        self.assertEqual(status,201)
        self.assertEqual(self.request('GET',f'/api/games/{game_id}/replay?match=true')[0],400)
        self.assertEqual(self.request('POST',f'/api/games/{game_id}/next',body)[0],200)
        self.assertEqual(self.request('POST',f'/api/games/{game_id}/next',body)[0],409)
        for _ in range(50):
            finish(session,agent);v=session.view()
            if v['match']['ended']:break
            session.next_hand({'decision_id':v['observation']['decision_id'],'hands_played':v['match']['hands_played']})
        self.assertTrue(session.match.ended)
        status,_,content=self.request('GET',f'/api/games/{game_id}/replay?match=true')
        self.assertEqual(status,200)
        status,_,content=self.request('POST','/api/replays',{'replay':json.loads(content)})
        self.assertEqual(status,201)
        r=json.loads(content)
        status,_,content=self.request('GET',f"/api/replays/{r['id']}?step={r['replay']['total']}")
        self.assertEqual(status,200)
        self.assertEqual(json.loads(content)['match']['final_scores'],session.match.final_scores)

    def test_origin_invalid_requests_and_unknown_resources(self):
        self.assertEqual(self.request("POST", "/api/games", {}, {"Origin": "https://foreign.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/games", [1, 2])[0], 400)
        self.assertEqual(self.request("POST", "/api/replays", {"replay": {"commands": []}})[0], 400)
        self.assertEqual(self.request("POST", "/api/games", {}, {"Content-Length": "-1"})[0], 413)
        self.assertEqual(self.request("GET", "/api/games/missing")[0], 404)


if __name__ == "__main__":
    unittest.main()
