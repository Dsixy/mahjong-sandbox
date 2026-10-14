"""Known scoring examples and inventory-valid real-wall riichi scenarios."""
from copy import deepcopy
from dataclasses import replace
import random
import unittest
from mahjong_sandbox import Action, ActionKind as K, IllegalAction, Phase, RiichiEngine, RiichiConfig
from mahjong_sandbox.model import Meld, Discard
from mahjong_sandbox.riichi_scoring import RiichiContext, score_hand, riichi_waits, dora_kind, ORPHANS, _value
from mahjong_sandbox.riichi_agent import RiichiAgent, riichi_shanten
from mahjong_sandbox.riichi_match import RiichiMatch, MatchConfig, replay_match
from mahjong_sandbox.replay import export_replay, replay
from mahjong_sandbox.tiles import parse_hand
from tests.helpers import select, submit, pass_rest


def allocate(text, pool):
    result = []
    for kind, n in enumerate(parse_hand(text)):
        for _ in range(n):
            tile = next(t for t in pool if t // 4 == kind)
            pool.remove(tile); result.append(tile)
    return result


def real_engine(hands=None, draws='', config=None):
    """Deal requested 13-tile hands from a full physical 136-tile wall."""
    pool = list(range(136)); hands = hands or {}; config = config or RiichiConfig()
    requested = {p: allocate(text, pool) for p, text in hands.items()}
    future = allocate(draws, pool)
    random.Random(99).shuffle(pool)
    for p in range(4):
        requested.setdefault(p, [])
        if len(requested[p]) > 13: raise ValueError('fixture requires 13 tiles before draw')
        while len(requested[p]) < 13: requested[p].append(pool.pop())
    deck = [requested[(config.dealer+p)%4][i] for i in range(13) for p in range(4)]
    return RiichiEngine(config, wall=deck+future+pool)


def value(text, win_kind=None, *, ctx=None, meld_specs=(), config=None, indicators=(), ura=()):
    pool = list(range(136)); melds = []
    for kind, spec in meld_specs:
        melds.append(Meld(kind, tuple(allocate(spec, pool)), None if kind == 'concealed_kong' else 0))
    tiles = allocate(text, pool)
    win = next(t for t in tiles if t//4 == win_kind) if win_kind is not None else tiles[-1]
    return score_hand(tiles, melds, win, context=ctx or RiichiContext(seat_wind=28),
                      config=config or RiichiConfig(red_fives=()), indicators=indicators, ura_indicators=ura)


class ScoringTests(unittest.TestCase):
    def test_pinfu_ron_and_tsumo_fu_and_payments(self):
        v=value('123456m234p456s55p', 0)
        self.assertEqual((v.han,v.fu,v.payment),(1,30,1000))
        v=value('123456m234p456s55p', 0, ctx=RiichiContext(self_draw=True,seat_wind=28))
        self.assertEqual((v.han,v.fu,v.tsumo_dealer,v.tsumo_other),(2,20,700,400))

    def test_wait_sensitive_fu(self):
        for kind, expected in ((0,30),(1,40),(2,40)):
            v=value('123m456m234p456s55p',kind,ctx=RiichiContext(seat_wind=28,riichi=True))
            self.assertEqual(v.fu,expected)
        v=value('123456m234p456s55p',13,ctx=RiichiContext(seat_wind=28,riichi=True))
        self.assertEqual(v.fu,40)

    def test_dora_alone_is_not_yaku(self):
        self.assertIsNone(value('456m123p789s22z', meld_specs=[('chi','123m')], indicators=[20]))
        self.assertIsNone(value('234m456p678s55p',meld_specs=[('chi','234p')],config=RiichiConfig(open_tanyao=False)))

    def test_open_tanyao_fu(self):
        v=value('234m456p678s55p',meld_specs=[('chi','234p')],win_kind=1)
        self.assertEqual((v.han,v.fu,v.payment),(1,30,1000))

    def test_pair_value_double_wind_and_ron_triplet_fu(self):
        v=value('111m234p567s11z',meld_specs=[('chi','789m')],win_kind=0,ctx=RiichiContext(seat_wind=27,round_wind=27,houtei=True))
        self.assertEqual(v.fu,30) # 20 + open terminal triplet 4 + double wind pair 4
        v=value('111333m555p777s22z',0,ctx=RiichiContext(seat_wind=28))
        self.assertIn('三暗刻',[x.name for x in v.items])
        self.assertNotIn('四暗刻',[x.name for x in v.items])

    def test_seven_pairs_and_orphans(self):
        v=value('1122m3344p5566s77z')
        self.assertEqual((v.family,v.han,v.fu),('seven_pairs',2,25))
        self.assertFalse(any(s.family=='seven_pairs' for s in __import__('mahjong_sandbox.riichi_scoring',fromlist=['shapes']).shapes(parse_hand('11112233445566m'))))
        v=value('19m19p19s12345677z',33)
        self.assertEqual(v.yakuman,2)
        v=value('119m19p19s1234567z',33)
        self.assertEqual(v.yakuman,1)
        self.assertEqual(riichi_waits(parse_hand('19m19p19s1234567z')),ORPHANS)

    def test_regular_yaku_examples(self):
        cases=[('123123m456p789s55p','一杯口',()),('123123m456456p55s','二杯口',()),
               ('123456789m123p55s','一气通贯',()),('123m123p123s789s55p','三色同顺',()),
               ('111m111p111s22z','三色同刻',(('peng','777s'),)),('111m999p111s22z', '混老头',(('peng','777z'),)),
               ('123m123p789s11z','混全带幺九',(('peng','555z'),)),
               ('123m123p789s11m','纯全带幺九',(('peng','999p'),)),
               ('123456789m11122z','混一色',()),('123456789m23455m','清一色',()),
               ('123m456p55566677z','小三元',())]
        for text,name,melds in cases:
            with self.subTest(text=text,name=name):
                v=value(text,meld_specs=melds)
                self.assertIsNotNone(v); self.assertIn(name,[x.name for x in v.items])

    def test_yakuman_examples_and_stacking(self):
        cases=[('11122233344455z','大四喜'),('11122233355544z','小四喜'),
               ('111m55566677722z','大三元'),('22233344466688s','绿一色'),
               ('111999m111999p11s','清老头'),('11123456789999m','九莲宝灯'),
               ('11123455678999m','纯正九莲宝灯')]
        for text,name in cases:
            with self.subTest(name=name):
                v=value(text,4 if name.startswith('纯正') else 0 if name=='九莲宝灯' else None,ctx=RiichiContext(self_draw=True,seat_wind=28))
                self.assertIn(name,[x.name for x in v.items]); self.assertGreaterEqual(v.yakuman,1)
        v=value('11122233344455z',ctx=RiichiContext(self_draw=True,seat_wind=28))
        self.assertEqual(v.yakuman,5) # big winds double + all honors + suuankou tanki double

    def test_four_kongs_and_three_kongs(self):
        melds=[('concealed_kong',s) for s in ('1111m','2222p','3333s','4444z')]
        v=value('55p',meld_specs=melds,ctx=RiichiContext(self_draw=True,seat_wind=28))
        self.assertIn('四杠子',[x.name for x in v.items])
        v=value('456m55p',meld_specs=melds[:3])
        self.assertIn('三杠子',[x.name for x in v.items]); self.assertEqual(v.fu,100)

    def test_dora_red_ura_and_cycles(self):
        self.assertEqual([dora_kind(t*4) for t in (8,17,26,30,33)],[0,9,18,27,31])
        cfg=RiichiConfig()
        v=value('123456m234p456s55p',0,config=cfg,ctx=RiichiContext(seat_wind=28,riichi=True),indicators=[48],ura=[48])
        items={i.name:i.fan for i in v.items}
        self.assertEqual((items['宝牌'],items['里宝牌'],items['赤宝牌']),(2,2,3))
        plain=value('123456m234p456s55p',0,config=cfg,indicators=[48],ura=[48])
        self.assertNotIn('里宝牌',[i.name for i in plain.items])

    def test_limits_rounding_and_double_toggle(self):
        from mahjong_sandbox.model import FanItem
        for han,fu,points in ((1,30,1000),(2,25,1600),(3,30,3900),(4,30,7700),(3,70,8000),(5,30,8000),(6,30,12000),(8,30,16000),(11,30,24000),(13,30,32000)):
            v=_value('standard',[FanItem('test',han)],fu,RiichiContext(seat_wind=28),RiichiConfig())
            self.assertEqual(v.payment,points)
        v=value('19m19p19s12345677z',33,config=RiichiConfig(double_yakuman=False))
        self.assertEqual(v.yakuman,1)


class EngineTests(unittest.TestCase):
    def test_dead_wall_initial_draw_and_information(self):
        e=RiichiEngine(seed=9)
        self.assertEqual((e.state.remaining_tiles,len(e.state.dead_wall)),(69,14))
        self.assertEqual(len(e.indicators()),1)
        obs=e.observe(0); self.assertEqual(len(obs['players'][0]['concealed']),14)
        self.assertTrue(all('concealed' not in p for p in obs['players'][1:]))
        self.assertEqual(obs['ura_indicators'],[])
        before=e.state_hash(); obs['players'][0]['concealed'].clear(); self.assertEqual(e.state_hash(),before)

    def test_riichi_deposit_double_ippatsu_draw_lock(self):
        e=real_engine({0:'123456m234p456s5p'},'9s')
        submit(e,0,K.RIICHI,tile_kind=26)
        self.assertEqual(e.state.scores,[25000]*4); pass_rest(e)
        self.assertEqual((e.state.scores[0],e.state.riichi_sticks),(24000,1))
        hand=e.state.players[0]; self.assertTrue(hand.double_riichi); self.assertTrue(hand.ippatsu)
        for p in (1,2,3):
            submit(e,p,K.DISCARD); pass_rest(e)
        actions=e.legal_actions(0)
        self.assertEqual([a.tiles[0] for a in actions if a.kind==K.DISCARD],[hand.last_drawn])
        submit(e,0,K.DISCARD); self.assertFalse(hand.ippatsu)

    def test_riichi_minimum_points_wall_and_open_hand(self):
        for cfg in (RiichiConfig(initial_scores=(900,25000,25000,25000)),RiichiConfig(riichi_min_live=70)):
            e=real_engine({0:'123456m234p456s5p'},'9s',cfg)
            self.assertFalse(any(a.kind==K.RIICHI for a in e.legal_actions(0)))

    def test_declared_tile_ron_cancels_deposit_and_single_hand_ends(self):
        e=real_engine({0:'123456m234p456s5p',1:'123m456p789s555z5p'},'5p')
        submit(e,0,K.RIICHI,tile_kind=13)
        submit(e,1,K.RON); pass_rest(e)
        self.assertEqual(e.state.phase,Phase.TERMINAL)
        self.assertFalse(e.state.players[0].riichi); self.assertEqual(e.state.riichi_sticks,0)
        self.assertFalse(any(t.reason=='riichi_deposit' for t in e.state.transfers))

    def test_discard_furiten_checks_all_waits_and_tsumo_allowed(self):
        e=real_engine({0:'123456m234p456s5p',1:'123456m234p456s5p'},'5p')
        # Swap an owned physical 5p into seat 1's earlier river; retain inventory.
        tile=next(t for t in e.state.wall[e.state.head:e.state.tail] if t//4==13)
        index=e.state.wall.index(tile); e.state.wall.pop(index); e.state.tail-=1
        e.state.players[1].river.append(Discard(tile,False))
        e.assert_invariants(); submit(e,0,K.DISCARD,tile_kind=13)
        self.assertNotIn(Action(K.RON),e.legal_actions(1)); self.assertTrue(e.observe(1)['players'][1]['furiten'])
        e=real_engine({0:'123456m234p456s5p'},'5p')
        e.state.players[0].temporary_furiten=True
        self.assertIn(Action(K.TSUMO),e.legal_actions(0))

    def test_passed_ron_temporary_and_riichi_furiten(self):
        for riichi in (False,True):
            e=real_engine({0:'123456m234p456s5p',1:'123456m234p456s5p'},'5p')
            e.state.players[1].riichi=riichi
            submit(e,0,K.DISCARD,tile_kind=13); pass_rest(e)
            self.assertFalse(e.state.players[1].temporary_furiten) # seat 1 has drawn again
            self.assertEqual(e.state.players[1].riichi_furiten,riichi)

    def test_chi_only_next_seat_and_pon_priority(self):
        e=real_engine({1:'23m',2:'11m'},'1m')
        submit(e,0,K.DISCARD,tile_kind=0)
        self.assertTrue(any(a.kind==K.CHI for a in e.legal_actions(1)))
        self.assertFalse(any(a.kind==K.CHI for a in e.legal_actions(2)))
        submit(e,1,K.CHI); submit(e,2,K.PENG); pass_rest(e)
        self.assertEqual(e.state.actor,2); self.assertEqual(e.state.players[2].melds[-1].kind,'peng')

    def test_kuikae_and_no_yaku_open_ron(self):
        e=real_engine({1:'234m'},'1m')
        submit(e,0,K.DISCARD,tile_kind=0); submit(e,1,K.CHI); pass_rest(e)
        self.assertFalse(any(a.kind==K.DISCARD and a.tiles[0]//4 in (0,3) for a in e.legal_actions(1)))

    def test_concealed_kong_dead_wall_and_dora(self):
        e=real_engine({0:'111m'},'1m'); dead=e.state.dead_wall[0]; remaining=e.state.remaining_tiles
        submit(e,0,K.CONCEALED_KONG); pass_rest(e)
        self.assertEqual((e.state.rinshan_count,e.state.indicator_count,e.state.remaining_tiles),(1,2,remaining-1))
        self.assertEqual(e.state.players[0].last_drawn,dead); self.assertEqual(len(e.state.dead_wall),14)
        self.assertFalse(any(t.reason.endswith('kong') for t in e.state.transfers))

    def test_open_kong_dora_delayed_and_no_fifth(self):
        e=real_engine({1:'111m'},'1m')
        submit(e,0,K.DISCARD,tile_kind=0); submit(e,1,K.OPEN_KONG); pass_rest(e)
        self.assertEqual((e.state.indicator_count,e.state.delayed_indicators),(1,1))
        submit(e,1,K.DISCARD); self.assertEqual((e.state.indicator_count,e.state.delayed_indicators),(2,0))

    def test_kokushi_can_rob_concealed_kong(self):
        e=real_engine({0:'111m',1:'9m19p19s12345677z'},'1m')
        submit(e,0,K.CONCEALED_KONG)
        self.assertIn(Action(K.RON),e.legal_actions(1)); submit(e,1,K.RON); pass_rest(e)
        self.assertEqual(e.state.end_reason,'ron'); self.assertEqual(e.state.kong_count,0)
        self.assertEqual(len(e.state.robbed_tiles),1); self.assertEqual(len(e.indicators()),1)

    def test_nine_terminals_abort_and_dealer_continuation(self):
        e=real_engine({0:'19m19p19s1234567z'},'2m')
        submit(e,0,K.ABORT_NINE)
        self.assertEqual(e.state.end_reason,'nine_terminals'); self.assertTrue(e.state.settlement['dealer_continues'])

    def test_two_tenpai_draw_transfers_are_hundred_point_payments(self):
        e=real_engine({0:'123456m234p456s5p',1:'234567m123p678s6p'},'9s')
        submit(e,0,K.DISCARD,tile_kind=26)
        e.state.players[0].river.extend(Discard(t,False) for t in e.state.wall[e.state.head:e.state.tail])
        e.state.head=e.state.tail
        e.assert_invariants(); pass_rest(e)
        self.assertEqual(e.state.scores,[26500,26500,23500,23500])
        self.assertTrue(all(t.amount%100==0 for t in e.state.transfers))

    def test_four_winds_abort(self):
        e=real_engine({p:'1z' for p in range(4)})
        for p in range(4):
            submit(e,p,K.DISCARD,tile_kind=27); pass_rest(e)
        self.assertEqual(e.state.end_reason,'four_winds')

    def test_multi_ron_and_head_bump_order(self):
        for multi in (False,True):
            e=real_engine({1:'123m234p789s555z5p',2:'123m234p789s666z5p'},'5p',RiichiConfig(multi_ron=multi))
            submit(e,0,K.DISCARD,tile_kind=13)
            submit(e,2,K.RON); submit(e,1,K.RON); pass_rest(e)
            self.assertEqual([w.player for w in e.state.wins],[1,2] if multi else [1])

    def test_real_rounds_agent_replay_privacy_and_stale_rejection(self):
        for seed in range(10):
            e=RiichiEngine(seed=seed); agents=[RiichiAgent(seed*4+p) for p in range(4)]
            for n in range(1000):
                if e.state.phase==Phase.TERMINAL: break
                p=e.next_player(); obs=e.observe(p); old=deepcopy(obs)
                action=agents[p].act(obs); self.assertEqual(obs,old)
                self.assertIn(action.to_dict(),obs['legal_actions'])
                e.submit(p,action,obs['decision_id'])
                before=e.state_hash()
                with self.assertRaises(IllegalAction): e.submit(p,action,obs['decision_id'])
                self.assertEqual(e.state_hash(),before)
            self.assertEqual(e.state.phase,Phase.TERMINAL)
            self.assertEqual(sum(e.state.scores)+e.state.riichi_sticks*1000,100000)
            self.assertEqual(replay(export_replay(e)).state_hash(),e.state_hash())

    def test_shanten_honors_seven_pairs_and_orphans(self):
        for text,expected in (('123456m234p456s5p',0),('1122m3344p5566s7z',0),('19m19p19s1234567z',0),('123456m234p456s55p',-1)):
            self.assertEqual(riichi_shanten(parse_hand(text)),expected)
        self.assertGreater(riichi_shanten(parse_hand('1234567z123456m')),0)


class MatchTests(unittest.TestCase):
    def test_real_east_match_carry_and_replay(self):
        match=RiichiMatch(MatchConfig(length='east'),seed=21)
        agents=[RiichiAgent(p) for p in range(4)]
        for _ in range(60):
            e=match.engine
            while e.state.phase!=Phase.TERMINAL:
                p=e.next_player(); e.submit(p,agents[p].act(e.observe(p)),e.state.decision_id)
            match.complete_hand()
            if match.ended: break
            old_scores=tuple(e.state.scores); old_bank=e.state.riichi_sticks
            match.next_hand(); self.assertEqual(match.engine.config.initial_scores,old_scores)
            self.assertEqual(match.engine.state.riichi_sticks,old_bank)
        self.assertTrue(match.ended); self.assertEqual(sum(match.final_scores),100000)
        data=match.export(); self.assertEqual(replay_match(data).summary(),match.summary())
        import json
        self.assertEqual(replay(json.loads(json.dumps(data))).summary(),match.summary())
        exported=match.export();exported['rounds'].clear();exported['expected']['final_scores'][0]=0
        self.assertTrue(match.rounds);self.assertNotEqual(match.final_scores[0],0)
        altered=deepcopy(data); altered['rounds'][-1]['config']['honba']+=1
        with self.assertRaises(ValueError): replay_match(altered)


def position(hands=None, meld_specs=None, actor=0, last_kind=None, config=None):
    """Trusted test position; a disjoint 136-tile inventory, not a portable replay."""
    from mahjong_sandbox.riichi_model import RiichiPlayer, RiichiState
    pool=list(range(136)); players=[RiichiPlayer() for _ in range(4)]
    for p,specs in (meld_specs or {}).items():
        for kind,text,source in specs:
            tiles=tuple(allocate(text,pool)); called=None if source is None else tiles[0]
            players[p].melds.append(Meld(kind,tiles,source,called))
            if source is not None: players[source].river.append(Discard(called,False,called_by=p))
    for p,text in (hands or {}).items(): players[p].concealed=allocate(text,pool)
    random.Random(38).shuffle(pool)
    for p,hand in enumerate(players):
        count=13+int(p==actor)-3*len(hand.melds)
        if len(hand.concealed)>count: raise ValueError('too many fixture tiles')
        while len(hand.concealed)<count: hand.concealed.append(pool.pop())
        hand.concealed.sort()
    players[actor].last_drawn=next(t for t in players[actor].concealed if t//4==last_kind) if last_kind is not None else players[actor].concealed[-1]
    players[actor].draw_source='wall'
    config=config or RiichiConfig(); e=RiichiEngine(config)
    dead=[pool.pop() for _ in range(14)]
    kans=sum(m.kind.endswith('kong') for p in players for m in p.melds)
    e.state=RiichiState(players,pool,0,len(pool),Phase.SELF_TURN,actor,scores=list(config.initial_scores),dead_wall=dead,
                        rinshan_count=kans,kong_count=kans,indicator_count=kans+1,any_call=True,
                        riichi_sticks=config.riichi_sticks,initial_bank=config.riichi_sticks*1000)
    e.events.clear();e.commands.clear();e._window(Phase.SELF_TURN,actor,(actor,)); e.assert_invariants()
    return e


class BoundaryTests(unittest.TestCase):
    def test_added_kan_robbed_and_keeps_original_triplet(self):
        e=position({0:'1m234p567s555s2p',1:'23m123p789s555z11p'},
                   {0:[('peng','111m',2)]},last_kind=0)
        submit(e,0,K.ADDED_KONG); submit(e,1,K.RON); pass_rest(e)
        self.assertEqual(e.state.players[0].melds[0].kind,'peng')
        self.assertEqual(e.state.kong_count,0);self.assertEqual(len(e.state.robbed_tiles),1)
        self.assertIn('抢杠',[i.name for i in e.state.wins[0].value.items])

    def test_added_kan_confirmed_delayed_indicator(self):
        e=position({0:'1m234p567s555s2p'},{0:[('peng','111m',2)]},last_kind=0)
        submit(e,0,K.ADDED_KONG); pass_rest(e)
        self.assertEqual(e.state.players[0].melds[0].kind,'added_kong')
        self.assertEqual((e.state.rinshan_count,e.state.indicator_count,e.state.delayed_indicators),(1,1,1))

    def test_regular_hand_cannot_rob_ankan_or_become_furiten_for_it(self):
        e=position({0:'1111m',1:'23m123p789s555z11p'})
        submit(e,0,K.CONCEALED_KONG)
        self.assertNotIn(Action(K.RON),e.legal_actions(1)); pass_rest(e)
        self.assertFalse(e.state.players[1].temporary_furiten)

    def test_riichi_kan_preserves_waits_and_all_decompositions(self):
        for text,allowed in [('1111m234p567s555z2p',True),('1111222333m444p5s',False)]:
            e=position({0:text},last_kind=0);e.state.players[0].riichi=True
            available=any(a.kind==K.CONCEALED_KONG for a in e.legal_actions(0))
            self.assertEqual(available,allowed)
            if allowed: submit(e,0,K.CONCEALED_KONG);pass_rest(e);e.assert_invariants()

    def test_four_riichi_abort_and_ippatsu_canceled_by_call(self):
        e=position({0:'9s'})
        for hand in e.state.players: hand.riichi=True
        submit(e,0,K.DISCARD); pass_rest(e)
        self.assertEqual(e.state.end_reason,'four_riichi')
        e=real_engine({0:'123456m234p456s5p',1:'99s'},'9s')
        submit(e,0,K.RIICHI,tile_kind=26); submit(e,1,K.PENG);pass_rest(e)
        self.assertTrue(e.state.players[0].riichi);self.assertFalse(e.state.players[0].ippatsu)

    def test_four_kans_abort_after_replacement_and_discard(self):
        e=position({0:'4444m555p666s1z'}, {0:[('concealed_kong','1111m',None)],
                   1:[('concealed_kong','2222p',None)],2:[('concealed_kong','3333s',None)]})
        submit(e,0,K.CONCEALED_KONG);pass_rest(e)
        self.assertEqual(e.state.phase,Phase.SELF_TURN);self.assertTrue(e.state.four_kans_pending)
        self.assertFalse(any(a.kind in (K.ADDED_KONG,K.CONCEALED_KONG) for a in e.legal_actions(0)))
        submit(e,0,K.DISCARD);pass_rest(e)
        self.assertEqual(e.state.end_reason,'four_kans')

    def test_pao_yakuman_split_and_honba(self):
        cfg=RiichiConfig(honba=2)
        e=position({0:'111m22p'}, {0:[('peng','555z',1),('peng','666z',2),('peng','777z',3)]},config=cfg)
        e.state.players[0].pao['大三元']=3
        v=value('111m22p',meld_specs=[('peng','555z'),('peng','666z'),('peng','777z')],ctx=RiichiContext(seat_wind=27))
        e._pay_value(0,v,source=2); e.assert_invariants()
        self.assertEqual(e.state.scores,[73600,25000,400,1000])
        self.assertEqual(sum(t.amount for t in e.state.transfers if t.reason=='pao'),24000)

    def test_nagashi_mangan_carries_deposits(self):
        e=real_engine(config=RiichiConfig(riichi_sticks=2))
        tile=next(t for t in e.state.players[0].concealed if t//4>=27 or t//4%9 in (0,8))
        e.submit(0,Action(K.DISCARD,(tile,)),e.state.decision_id)
        # Move remaining live wall to seat 1's river; seat 0's river remains a single uncalled outside tile.
        e.state.players[1].river.extend(Discard(t,False) for t in e.state.wall[e.state.head:e.state.tail])
        e.state.head=e.state.tail;pass_rest(e)
        self.assertEqual(e.state.settlement['nagashi'],[0]);self.assertEqual(e.state.scores,[37000,21000,21000,21000])
        self.assertEqual(e.state.riichi_sticks,2)

    def test_triple_ron_abort_and_toggle(self):
        for abort in (False,True):
            # Three separate pairs waiting on the same 7p; triplets do not share physical tiles.
            e=real_engine({1:'123m123p123s555z7p',2:'456m456p456s666z7p',3:'789m789s111z222z7p'},'7p',RiichiConfig(triple_ron_abort=abort))
            submit(e,0,K.DISCARD,tile_kind=15)
            for p in (3,2,1):submit(e,p,K.RON)
            self.assertEqual(e.state.end_reason,'triple_ron' if abort else 'ron')
            self.assertEqual(len(e.state.wins),0 if abort else 3)


class MatchPolicyTests(unittest.TestCase):
    @staticmethod
    def drawn(config):
        e=position({0:'147m258p369s12344z'},config=config)
        submit(e,0,K.DISCARD)
        e.state.players[0].river.extend(Discard(t,False) for t in e.state.wall[e.state.head:e.state.tail])
        e.state.head=e.state.tail;pass_rest(e)
        return e

    def test_dealer_rotation_extension_and_finite_extra_wind(self):
        for length,wind in [('east',27),('south',28)]:
            match=RiichiMatch(MatchConfig(length=length))
            match.engine=self.drawn(RiichiConfig(dealer=3,round_wind=wind,hand_number=4))
            self.assertFalse(match.complete_hand()['ended'])
            match.next_hand()
            self.assertEqual((match.engine.config.dealer,match.engine.config.round_wind,match.engine.config.hand_number,match.engine.config.honba),(0,wind+1,1,1))
            match.engine=self.drawn(RiichiConfig(dealer=3,round_wind=wind+1,hand_number=4))
            match._resolved=False; match.complete_hand()
            self.assertTrue(match.ended);self.assertEqual(match.end_reason,'extension_end')

    def test_dealer_agari_yame_and_bankruptcy(self):
        for enabled in (False,True):
            match=RiichiMatch(MatchConfig(length='east',agari_yame=enabled))
            cfg=RiichiConfig(round_wind=27,hand_number=4,initial_scores=(32000,23000,23000,22000),red_fives=())
            match.engine=position({0:'123456m234p456s55p'},config=cfg,last_kind=0)
            submit(match.engine,0,K.TSUMO);match.complete_hand()
            self.assertEqual(match.ended,enabled)
        match=RiichiMatch();match.engine=self.drawn(RiichiConfig(initial_scores=(-100,50100,25000,25000)))
        match.complete_hand();self.assertEqual(match.end_reason,'bankruptcy')

    def test_final_pot_awarded_once_and_tie_by_initial_seat(self):
        match=RiichiMatch(MatchConfig(length='east',west_extension=False))
        match.engine=self.drawn(RiichiConfig(dealer=3,round_wind=27,hand_number=4,riichi_sticks=2))
        summary=match.complete_hand();self.assertTrue(summary['ended'])
        self.assertEqual(summary['final_scores'],[27000,25000,25000,25000])
        self.assertEqual(match.complete_hand()['final_scores'],summary['final_scores'])
