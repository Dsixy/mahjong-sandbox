"""Bundled checkpoint, optional dependencies and playable trained opponents."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest

from mahjong_sandbox.model import Phase
from mahjong_sandbox.replay import replay
from mahjong_sandbox.rl.opponent import MODEL_PATH, high_difficulty_opponent
from mahjong_sandbox.webserver import GameSession, make_server

HAS_RL = all(importlib.util.find_spec(name) is not None for name in ('torch', 'numpy', 'sb3_contrib'))


class TrainedOpponentSettingsTests(unittest.TestCase):
    def test_bundled_checkpoint_matches_promoted_model(self):
        metadata = json.loads(MODEL_PATH.with_suffix('.json').read_text())
        self.assertEqual(metadata['timesteps'], 6361088)
        self.assertEqual(hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), metadata['model_sha256'])
        self.assertEqual(metadata['model_sha256'], '3daa6cf1262373af425d379d6710b289e54bc70e6b562aac4fbc353d997b0241')
        self.assertTrue(metadata['rules']['exchange_three'])

    def test_incompatible_settings_are_rejected(self):
        for settings in ({'variant': 'riichi'}, {'exchange': False}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, '换三张的川麻'):
                GameSession(policy='high', **settings)

    def test_baseline_games_work_without_torch_and_high_has_actionable_error(self):
        script = '''
import sys
sys.modules['torch'] = None
from mahjong_sandbox.webserver import GameSession
assert GameSession(seed=44).view()['settings']['policy'] == 'greedy'
try:
    GameSession(seed=44, policy='high')
except ValueError as error:
    assert 'RL' in str(error), error
else:
    raise AssertionError('high difficulty must require RL dependencies')
'''
        subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                       check=True, capture_output=True, text=True)


@unittest.skipUnless(HAS_RL, 'install .[rl] for trained opponent integration')
class TrainedOpponentIntegrationTests(unittest.TestCase):
    def test_reference_observations_match_training_features(self):
        import numpy as np
        from mahjong_sandbox.bots import make_bot
        from mahjong_sandbox.engine import Engine
        from mahjong_sandbox.fast import FastEngine
        from mahjong_sandbox.rl.features import encode_batch, observation_batch
        reference, fast = Engine(seed=44), FastEngine(seed=44)
        bots = [make_bot('greedy', variant='sichuan', seed=p) for p in range(4)]
        for _ in range(32):
            player = reference.next_player()
            self.assertEqual(player, fast.next_player())
            obs = reference.observe(player)
            left = encode_batch(observation_batch(obs))
            right = encode_batch(observation_batch(fast.observe(player)))
            for key in left:
                np.testing.assert_array_equal(left[key], right[key], err_msg=key)
            action = bots[player].act(obs)
            reference.submit(player, action, reference.state.decision_id)
            fast.submit(player, action, fast.state.decision_id)

    def test_shared_cpu_model_is_deterministic_under_concurrent_inference(self):
        left, right = GameSession(seed=44, policy='high'), GameSession(seed=44, policy='high')
        agent = high_difficulty_opponent()
        self.assertIs(left.agents[1], right.agents[3])
        self.assertEqual(str(agent.agent.model.device), 'cpu')
        observations = [left.engine.observe(0), right.engine.observe(0)] * 2
        with ThreadPoolExecutor(4) as pool:
            actions = list(pool.map(agent.act, observations))
        for action, obs in zip(actions, observations):
            self.assertIn(action.to_dict(), obs['legal_actions'])
        self.assertTrue(all(action == actions[0] for action in actions))

    def test_complete_real_game_private_frames_and_replay(self):
        from mahjong_sandbox.bots import make_bot
        human = make_bot('greedy', variant='sichuan', seed=123)
        game = GameSession(seed=44, policy='high')
        for _ in range(500):
            obs = game.view()['observation']
            if game.engine.state.phase == Phase.TERMINAL:
                break
            result = game.submit({'decision_id': obs['decision_id'], 'action': human.act(obs).to_dict()})
            for frame in result['frames']:
                projected = frame['observation']
                if projected['phase'] != 'terminal':
                    self.assertTrue(all('concealed' not in p for p in projected['players'][1:] if p['can_act']))
        self.assertEqual(game.engine.state.phase, Phase.TERMINAL)
        game.engine.assert_invariants()
        self.assertEqual(sum(game.engine.state.scores), 0)
        self.assertEqual(replay(game.export()).state_hash(), game.engine.state_hash())

    def test_http_high_policy_and_unsupported_rules(self):
        server = make_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for settings, expected in [({}, 201), ({'exchange': False}, 400), ({'variant': 'riichi'}, 400)]:
                conn = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=30)
                try:
                    conn.request('POST', '/api/games', json.dumps({'seed': 44, 'policy': 'high', **settings}))
                    response = conn.getresponse()
                    data = json.loads(response.read())
                    self.assertEqual(response.status, expected, data)
                    if expected == 201:
                        self.assertEqual(data['settings']['policy'], 'high')
                        self.assertTrue(data['settings']['exchange'])
                finally:
                    conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
