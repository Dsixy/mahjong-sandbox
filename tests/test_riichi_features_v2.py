"""Ordered public state, hidden-information boundaries and versioned policies."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_RL=all(importlib.util.find_spec(p) for p in ('numpy','torch','sb3_contrib'))


@unittest.skipUnless(HAS_RL,'install RL dependencies')
class RiichiFeaturesV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def encode(self, obs, version='riichi_visible_v2'):
        from mahjong_sandbox.rl.features import observation_batch,encode_batch
        from mahjong_sandbox.riichi_model import RiichiConfig
        return encode_batch(observation_batch(obs),rules=RiichiConfig(),hints=False,feature_version=version)

    def test_reference_and_both_fast_modes_keep_the_same_public_journal(self):
        from mahjong_sandbox.fast import FastRiichiEngine,RuntimeOptions
        from mahjong_sandbox.riichi_engine import RiichiEngine
        from mahjong_sandbox.bots import make_bot
        from mahjong_sandbox.model import Phase
        import numpy as np
        engines=[RiichiEngine(seed=9),FastRiichiEngine(seed=9),
                 FastRiichiEngine(seed=9,runtime=RuntimeOptions(record=True))]
        bots=[make_bot('greedy',variant='riichi',seed=p) for p in range(4)]
        for _ in range(180):
            first=engines[0]
            if first.state.phase==Phase.TERMINAL:break
            p=first.next_player();observations=[e.observe(p) for e in engines]
            self.assertEqual(observations[0]['public_history'],observations[1]['public_history'])
            self.assertEqual(observations[0]['public_history'],observations[2]['public_history'])
            a=self.encode(observations[0]);b=self.encode(observations[1])
            for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            action=bots[p].act(observations[0])
            for e in engines:e.submit(p,action,e.state.decision_id)
        self.assertFalse(engines[1].events)
        obs=engines[1].observe(0);obs['public_history'][0][3]=135
        self.assertNotEqual(obs['public_history'],engines[1].observe(0)['public_history'])

    def test_early_river_order_reaches_the_policy(self):
        import numpy as np
        import torch
        from mahjong_sandbox.fast import FastRiichiEngine
        from mahjong_sandbox.bots import make_bot
        from mahjong_sandbox.model import Phase
        from mahjong_sandbox.public_history import EVENT_KINDS
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.policy import MahjongPolicy
        from sb3_contrib import MaskablePPO
        observation=None
        for seed in range(9,19):
            engine=FastRiichiEngine(seed=seed)
            bots=[make_bot('greedy',variant='riichi',seed=p) for p in range(4)]
            for _ in range(500):
                if engine.state.phase==Phase.TERMINAL:break
                for seat,hand in enumerate(engine.state.players):
                    if len(hand.river)>=10 and hand.river[0].tile//4!=hand.river[1].tile//4:
                        observation=engine.observe(engine.next_player());target=seat;break
                if observation:break
                p=engine.next_player();engine.submit(p,bots[p].act(engine.observe(p)),engine.state.decision_id)
            if observation:break
        self.assertIsNotNone(observation)
        changed=deepcopy(observation);river=changed['players'][target]['river']
        river[0]['tile'],river[1]['tile']=river[1]['tile'],river[0]['tile']
        for event in changed['public_history']:
            if event[0]==EVENT_KINDS.index('discard') and event[1]==target and event[5] in (0,1):
                event[3]=river[event[5]]['tile']
        legacy_a,legacy_b=self.encode(observation,'riichi_visible_v1'),self.encode(changed,'riichi_visible_v1')
        for key in legacy_a:np.testing.assert_array_equal(legacy_a[key],legacy_b[key],err_msg=key)
        a,b=self.encode(observation),self.encode(changed)
        self.assertFalse(np.array_equal(a['rivers'],b['rivers']))
        self.assertFalse(np.array_equal(a['public_history'],b['public_history']))
        env=RiichiVecEnv(1,hints=False,feature_version='riichi_visible_v2')
        try:
            model=MaskablePPO(MahjongPolicy,env,n_steps=2,batch_size=2,seed=17,policy_kwargs={'variant':'riichi','hidden':32})
            extractor=model.policy.features_extractor
            with torch.no_grad():
                left=extractor(model.policy.obs_to_tensor(a)[0]);right=extractor(model.policy.obs_to_tensor(b)[0])
            self.assertFalse(torch.equal(left,right))
            self.assertFalse(torch.equal(model.policy.mlp_extractor.forward_actor(left),model.policy.mlp_extractor.forward_actor(right)))
            # Global chronology also reaches the network when per-seat rivers
            # and their summaries stay identical.
            timeline=deepcopy(a)
            timeline['public_history'][:,[0,1]]=timeline['public_history'][:,[1,0]]
            with torch.no_grad():
                chronological=extractor(model.policy.obs_to_tensor(timeline)[0])
            self.assertFalse(torch.equal(left,chronological))
        finally:env.close()

    def test_meld_groups_sources_red_tiles_and_called_discard(self):
        from tests.test_riichi import real_engine
        from tests.helpers import submit,pass_rest
        from mahjong_sandbox.model import ActionKind as K
        from mahjong_sandbox.public_history import EVENT_KINDS,EVENT_DETAILS
        engine=real_engine({1:'34m'},'5m')
        submit(engine,0,K.DISCARD,tile_kind=4);submit(engine,1,K.CHI);pass_rest(engine)
        features=self.encode(engine.observe(1));meld=features['melds'][0,0,0]
        self.assertEqual(meld.tolist(),[0,3,2,3,4,34,4,4,1,34,0])
        self.assertEqual(features['rivers'][0,3,0,3],0)
        history=features['public_history'][0][features['history_mask'][0]]
        call=next(row for row in history if row[0]==EVENT_KINDS.index('chi'))
        self.assertEqual((call[1],call[2],call[3],call[4],call[8],call[11]),(0,3,4,0,EVENT_DETAILS.index('chi'),1))
        engine=real_engine({0:'111m'},'1m')
        submit(engine,0,K.CONCEALED_KONG);pass_rest(engine)
        features=self.encode(engine.observe(0))
        self.assertEqual(features['melds'][0,0,0,0],3)
        self.assertEqual(features['melds'][0,0,0,1],4)
        self.assertEqual(sum(row[0]==EVENT_KINDS.index('dora_revealed') for row in engine.public_history),2)

    def test_riichi_declaration_indexes_are_not_tile_ids(self):
        from tests.test_riichi import real_engine
        from tests.helpers import submit,pass_rest
        from mahjong_sandbox.model import ActionKind as K
        from mahjong_sandbox.public_history import EVENT_KINDS
        engine=real_engine({0:'123456m234p456s5p'},'9s')
        submit(engine,0,K.RIICHI,tile_kind=26)
        declared=self.encode(engine.observe(1))
        self.assertEqual(declared['rivers'][0,3,0,2]&8,0)  # declared but not accepted yet
        pass_rest(engine)
        features=self.encode(engine.observe(1))
        self.assertEqual(features['rivers'][0,3,0,0],26)
        self.assertEqual(features['rivers'][0,3,0,2]&8,8)
        kinds=[row[0] for row in engine.public_history]
        self.assertLess(kinds.index(EVENT_KINDS.index('riichi_declared')),kinds.index(EVENT_KINDS.index('discard')))
        self.assertGreater(kinds.index(EVENT_KINDS.index('riichi_accepted')),kinds.index(EVENT_KINDS.index('discard')))

    def test_multi_ron_winner_markers_are_relative_seats(self):
        from tests.test_riichi import real_engine
        from tests.helpers import submit,pass_rest
        from mahjong_sandbox.model import ActionKind as K
        engine=real_engine({1:'123m234p789s555z5p',2:'123m234p789s666z5p'},'5p')
        submit(engine,0,K.DISCARD,tile_kind=13)
        submit(engine,1,K.RON);submit(engine,2,K.RON);pass_rest(engine)
        self.assertEqual(engine.state.players[0].river[-1].winner_refs,[1,2])
        features=self.encode(engine.observe(3))
        self.assertEqual(features['rivers'][0,1,0,4],(1<<2)|(1<<3))

    def test_match_context_rules_tie_order_and_hand_progression(self):
        import numpy as np
        from mahjong_sandbox.riichi_match import RiichiMatch,MatchConfig
        from mahjong_sandbox.riichi_model import RiichiConfig
        from mahjong_sandbox.fast import FastRiichiEngine
        from mahjong_sandbox.bots import make_bot
        from mahjong_sandbox.model import Phase
        from mahjong_sandbox.rl.riichi_features import STATE_DIM
        rules=RiichiConfig(dealer=2,initial_scores=(31000,23000,25000,21000))
        config=MatchConfig(length='east',target=35000,bankruptcy=False,agari_yame=False,tenpai_yame=True,west_extension=False)
        match=RiichiMatch(config,rules,seed=91,engine_factory=FastRiichiEngine,record=False)
        original=match.engine.observe(0);a=self.encode(original)
        alternate=deepcopy(original);alternate['match_context']['config']['length']='south'
        self.assertFalse(np.array_equal(a['state'][:,STATE_DIM:],self.encode(alternate)['state'][:,STATE_DIM:]))
        context=a['state'][0,STATE_DIM:STATE_DIM+17]
        np.testing.assert_array_equal(context[:3],[0,1,0]);self.assertEqual(context[2+1+1],0) # bankruptcy
        self.assertEqual(original['match_context']['initial_dealer'],2)
        bots=[make_bot('greedy',variant='riichi',seed=p) for p in range(4)]
        while match.engine.state.phase!=Phase.TERMINAL:
            e=match.engine;p=e.next_player();e.submit(p,bots[p].act(e.observe(p)),e.state.decision_id)
        match.complete_hand();self.assertFalse(match.ended)
        match.next_hand();next_obs=match.engine.observe(0)
        self.assertEqual(next_obs['match_context']['hands_played'],1)
        self.assertEqual(next_obs['match_context']['initial_scores'],list(rules.initial_scores))
        self.assertEqual(original['match_context']['hands_played'],0)
        self.assertEqual(len(next_obs['public_history']),2)

    def test_hidden_tiles_do_not_change_any_input(self):
        import numpy as np
        from mahjong_sandbox.fast import FastRiichiEngine
        from mahjong_sandbox.public_history import EVENT_KINDS
        engine=FastRiichiEngine(seed=76);changed=deepcopy(engine);p=engine.next_player()
        other=(p+1)%4;head=changed.state.head
        changed.state.players[other].concealed[0],changed.state.wall[head]=changed.state.wall[head],changed.state.players[other].concealed[0]
        changed.state.dead_wall[5],changed.state.dead_wall[7]=changed.state.dead_wall[7],changed.state.dead_wall[5]
        a,b=self.encode(engine.observe(p)),self.encode(changed.observe(p))
        for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        for row in engine.public_history:
            if row[0]==EVENT_KINDS.index('draw'):self.assertEqual(row[3],-1)
        self.assertNotIn('concealed',engine.observe(p)['players'][other])

    def test_relative_seats_preserve_sources_and_all_ordered_features(self):
        import numpy as np
        from tests.test_riichi import real_engine
        from tests.helpers import submit,pass_rest
        from mahjong_sandbox.model import ActionKind as K
        engine=real_engine({1:'34m'},'5m')
        submit(engine,0,K.DISCARD,tile_kind=4);submit(engine,1,K.CHI);pass_rest(engine)
        obs=engine.observe(1);rotated=deepcopy(obs);shift=2
        for key in ('player','actor','dealer'):rotated[key]=(rotated[key]+shift)%4
        rotated['players']=[None]*4
        for player in obs['players']:
            item=deepcopy(player);item['player']=(item['player']+shift)%4
            for meld in item['melds']:
                if meld['source_player'] is not None:meld['source_player']=(meld['source_player']+shift)%4
            for discard in item['river']:
                if discard['called_by'] is not None:discard['called_by']=(discard['called_by']+shift)%4
            rotated['players'][item['player']]=item
        for key in ('scores',):
            rotated[key]=[obs[key][(p-shift)%4] for p in range(4)]
        context=rotated['match_context'];context['initial_dealer']=(context['initial_dealer']+shift)%4
        context['initial_scores']=[obs['match_context']['initial_scores'][(p-shift)%4] for p in range(4)]
        for row in rotated['public_history']:
            for col in (1,2):
                if 0<=row[col]<4:row[col]=(row[col]+shift)%4
        a,b=self.encode(obs),self.encode(rotated)
        for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)

    def test_missing_or_overflowing_history_is_rejected(self):
        from mahjong_sandbox.fast import FastRiichiEngine
        from mahjong_sandbox.public_history import MAX_PUBLIC_EVENTS,remember_public
        engine=FastRiichiEngine(seed=3);obs=engine.observe(0)
        missing=deepcopy(obs);missing.pop('public_history_version')
        with self.assertRaisesRegex(ValueError,'complete public history'):self.encode(missing)
        overflow=deepcopy(obs);overflow['public_history']*=MAX_PUBLIC_EVENTS
        with self.assertRaisesRegex(ValueError,'overflowing'):self.encode(overflow)
        engine.public_history=[engine.public_history[0]]*MAX_PUBLIC_EVENTS
        with self.assertRaisesRegex(RuntimeError,'refusing to truncate'):remember_public(engine,'draw',{'player':0,'source':'wall'})

    def test_process_and_sync_features_match_with_custom_profile(self):
        import numpy as np
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.riichi_match import MatchConfig
        from mahjong_sandbox.rl.policy import MahjongPolicy
        from mahjong_sandbox.rl.selfplay import publish_snapshot
        from mahjong_sandbox.riichi_model import RiichiConfig
        from sb3_contrib import MaskablePPO
        directory=tempfile.TemporaryDirectory()
        base=RiichiVecEnv(2,hints=False,feature_version='riichi_visible_v2')
        try:
            model=MaskablePPO(MahjongPolicy,base,n_steps=8,batch_size=8,policy_kwargs={'variant':'riichi','hidden':32})
            publish_snapshot(model,directory.name,rules=RiichiConfig(),hints=False,training={})
        finally:base.close()
        args=dict(seed=38,hints=False,feature_version='riichi_visible_v2',match_length='south',
                  match_config=MatchConfig(target=35000,agari_yame=False),
                  opponent_manifest=Path(directory.name)/'opponents.json',self_play_probability=1.)
        left,right=RiichiVecEnv(2,workers=0,**args),RiichiVecEnv(2,workers=1,**args)
        try:
            a,b=left.reset(),right.reset()
            for _ in range(12):
                for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                actions=a['mask'].argmax(1);a,r,d,_=left.step(actions);b,s,e,_=right.step(actions)
                np.testing.assert_array_equal(r,s);np.testing.assert_array_equal(d,e)
        finally:left.close();right.close();directory.cleanup()

    def test_whole_match_resets_ordered_inputs_and_preserves_rank_rewards(self):
        import numpy as np
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.riichi_features_v2 import canonical_index
        from mahjong_sandbox.bots import make_bot
        env=RiichiVecEnv(1,seed=991,opponents=('greedy',),hints=False,backend='numba',
            feature_version='riichi_visible_v2',match_length='south',reward_mode='rank')
        bot=make_bot('greedy',variant='riichi',seed=66,backend='numba')
        try:
            obs=env.reset();previous=0;seen_reset=False;total=0
            for _ in range(3000):
                count=int(obs['history_mask'].sum());seen_reset|=count<previous;previous=count
                engine=env.core.engines[0];p=env.core.players[0]
                index=canonical_index(env.raw,bot.act(engine.observe(p)))
                obs,reward,done,infos=env.step(np.array([index]));total+=float(reward[0])
                if done[0]:
                    self.assertEqual(total,2.5-infos[0]['metrics']['rank'])
                    self.assertTrue(infos[0]['terminal_observation']['history_mask'].any())
                    self.assertTrue(obs['history_mask'].any());self.assertTrue(seen_reset)
                    break
                self.assertEqual(float(reward[0]),0.)
            else:self.fail('match did not end')
        finally:env.close()

    def test_default_training_checkpoint_resume_and_legacy_models(self):
        import torch
        from mahjong_sandbox.rl.train import TrainConfig,train
        from mahjong_sandbox.rl.checkpoint import load_checkpoint,checksum
        from mahjong_sandbox.rl.agent import RLAgent
        from mahjong_sandbox.fast import FastRiichiEngine
        config=TrainConfig(variant='riichi',match_length='east',match_config={'length':'east','target':35000},
            total_timesteps=16,n_envs=2,workers=0,rollout_steps=8,batch_size=8,epochs=1,
            bc_samples=8,bc_epochs=1,validation_blocks=0,hints=False)
        with tempfile.TemporaryDirectory() as directory:
            report=train(directory,config);path=Path(directory)/'latest.zip';model,meta=load_checkpoint(path)
            self.assertEqual(meta['feature_version'],'riichi_visible_v2')
            self.assertTrue(all(torch.isfinite(x).all() for x in model.policy.parameters()))
            self.assertEqual(meta['training']['config']['match_config']['target'],35000)
            before={k:v.clone() for k,v in model.policy.state_dict().items() if 'public_encoder' in k}
            report=train(directory,config,resume=path);after,meta=load_checkpoint(path)
            self.assertEqual(report['final_timesteps'],32)
            self.assertTrue(any(not torch.equal(v,after.policy.state_dict()[k]) for k,v in before.items()))
            with self.assertRaisesRegex(ValueError,'feature versions cannot change'):
                train(Path(directory)/'wrong',TrainConfig(variant='riichi',feature_version='riichi_visible_v1',bc_samples=0,hints=False),resume=path)
        root=Path(__file__).resolve().parents[1]
        proof=json.loads((root/'docs/placement/training.json').read_text())['wheel_verification']['extracted_package_cpu_load_and_legal_actions']['riichi']
        path=root/'mahjong_sandbox/models/riichi-high/best.zip';original=checksum(path)
        agent=RLAgent(path,backend='numba');e=FastRiichiEngine(seed=79);p=e.next_player()
        self.assertEqual(agent.feature_version,'riichi_visible_v1')
        self.assertEqual(agent.act(e.observe(p)).to_dict(),proof['legal_cpu_action'])
        self.assertEqual(checksum(path),original)
