import json
from copy import deepcopy
import unittest
from mahjong_sandbox import Phase, RiichiAgent
from mahjong_sandbox.webserver import GameSession, ReplaySession, WEB_ROOT
from mahjong_sandbox.replay import replay


def finish(session, agent):
    for n in range(500):
        obs=session.view()['observation']
        if obs['phase']=='terminal': return
        result=session.submit({'decision_id':obs['decision_id'],'action':agent.act(obs).to_dict()})
        for frame in result['frames']:
            if frame['observation']['phase']!='terminal':
                assert all('concealed' not in p for p in frame['observation']['players'][1:] if p['can_act'])
    raise AssertionError('no end')


class RiichiWebTests(unittest.TestCase):
    def test_real_rounds_private_frames_and_export(self):
        for seed in range(8):
            s=GameSession(seed=seed,variant='riichi',policy=('greedy','random','win-first')[seed%3])
            with self.assertRaises(ValueError): s.export()
            finish(s,RiichiAgent(seed))
            self.assertEqual(replay(s.export()).state_hash(),s.engine.state_hash())
            r=ReplaySession(s.export())
            self.assertEqual(r.view(r.total)['observation']['scores'],s.engine.state.scores)
            self.assertEqual(r.view()['observation']['ura_indicators'],[])

    def test_east_match_next_hand_and_full_match_replay(self):
        s=GameSession(seed=21,variant='riichi',match='east'); human=RiichiAgent(0)
        for _ in range(50):
            finish(s,human); v=s.view()
            if v['match']['ended']: break
            with self.assertRaises(ValueError): s.export(match=True)
            body={'decision_id':v['observation']['decision_id'],'hands_played':v['match']['hands_played']}
            prior=tuple(s.engine.state.scores); next_view=s.next_hand(body)
            self.assertEqual(s.engine.config.initial_scores,prior)
            before=s.engine.state_hash()
            with self.assertRaises(ValueError): s.next_hand(body)
            self.assertEqual(s.engine.state_hash(),before)
        self.assertTrue(s.match.ended)
        r=ReplaySession(s.export(match=True))
        self.assertGreater(len(r.hands),1)
        for hand in r.hands:
            view=r.view(hand['start']); self.assertEqual(view['observation']['phase'],'self_turn')
            self.assertFalse(any('concealed' in p for p in view['observation']['players'][1:]))
        self.assertEqual(r.view(r.total)['observation']['scores'],s.engine.state.scores)
        with self.assertRaises(ValueError): s.next_hand({'decision_id':s.engine.state.decision_id,'hands_played':len(s.match.rounds)})

    def test_variant_settings_and_offline_example(self):
        for settings in ({'variant':'sanma'},{'match':'east'},{'variant':'riichi','red':1},{'variant':'riichi','tanyao':None}):
            with self.assertRaises(ValueError): GameSession(**settings)
        s=GameSession(seed=4,variant='riichi',red=False,tanyao=False)
        self.assertEqual(s.view()['observation']['red_fives'],[])
        self.assertFalse(s.engine.config.open_tanyao)
        data=json.loads((WEB_ROOT/'assets/riichi.json').read_text())
        r=ReplaySession(data)
        self.assertEqual(r.view(r.total)['observation']['end_reason'],'ron')
        corrupted=deepcopy(data); corrupted['expected']['scores'][0]+=100
        with self.assertRaises(ValueError): ReplaySession(corrupted)
        for key, value in (('config', []), ('config', 'riichi'), ('commands', None)):
            corrupted=deepcopy(data);corrupted[key]=value
            with self.assertRaises(ValueError): ReplaySession(corrupted)
