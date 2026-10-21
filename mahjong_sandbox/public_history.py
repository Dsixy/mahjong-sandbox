"""Compact public riichi journal, independent of private replay recording.

Rows contain only announced facts. In particular a draw records its actor and
source, never its tile, even when the caller owns that tile.
"""
MAX_PUBLIC_EVENTS = 512
HISTORY_VERSION = 'riichi_public_v1'
EVENT_KINDS = ('draw', 'discard', 'riichi_declared', 'riichi_accepted', 'chi',
               'peng', 'kong_proposed', 'kong_confirmed', 'dora_revealed',
               'ura_revealed', 'payment', 'win', 'kong_robbed', 'round_ended')
EVENT_DETAILS = ('none', 'initial', 'wall', 'kong', 'chi', 'peng', 'open_kong',
                 'concealed_kong', 'added_kong', 'ron', 'tsumo', 'honba', 'pao',
                 'riichi_deposit', 'riichi_award', 'nagashi_mangan', 'noten',
                 'wall_exhausted', 'nine_terminals', 'four_winds', 'four_riichi',
                 'four_kans', 'triple_ron')
HISTORY_FIELDS = ('kind', 'actor', 'source', 'tile', 'meld_index', 'river_index',
                  'flags', 'amount', 'detail', 'remaining', 'discard_count')
RULE_FLAGS = ('open_tanyao', 'kuikae_forbidden', 'multi_ron', 'triple_ron_abort',
              'abortive_draws', 'nagashi_mangan', 'kiriage_mangan', 'counted_yakuman',
              'double_yakuman', 'pao', 'delayed_open_kan_dora')


def remember_public(engine, kind, data):
    if kind not in EVENT_KINDS:
        return
    state = engine.state
    actor, source, tile, meld, river = data.get('player', -1), -1, -1, -1, -1
    flags, amount, detail = 0, 0, 'none'
    if kind == 'draw':
        detail = data['source']
    elif kind in ('discard', 'riichi_declared', 'kong_proposed', 'kong_robbed'):
        tile = data['tile']
        if kind == 'discard':
            river = len(state.players[actor].river) - 1
            flags = int(data['tsumogiri'])
        elif kind == 'riichi_declared':
            river = len(state.players[actor].river)
        elif kind == 'kong_proposed':
            meld = data['meld_index'] if data.get('meld_index') is not None else len(state.players[actor].melds)
            detail = data['kind']
    elif kind == 'riichi_accepted':
        river, flags = data['discard_index'], int(data['double'])
        tile = state.players[actor].river[river].tile
    elif kind in ('chi', 'peng', 'kong_confirmed'):
        value = data['meld']
        source = value['source_player'] if value['source_player'] is not None else -1
        tile = value['added_tile'] if value['added_tile'] is not None else value['called_tile']
        if tile is None:
            tile = value['tiles'][0]
        detail = value['kind']
        meld = next(i for i, m in enumerate(state.players[actor].melds)
                    if tuple(m.tiles) == tuple(value['tiles']))
    elif kind in ('dora_revealed', 'ura_revealed'):
        seen = engine._announced_indicators.setdefault(kind, set())
        for indicator in data['indicators']:
            if indicator not in seen:
                seen.add(indicator)
                _append(engine, kind, -1, -1, indicator, -1, -1, 0, 0, 'none')
        return
    elif kind == 'payment':
        actor, source, amount, detail = data['receiver'], data['payer'], data['amount'], data['reason']
    elif kind == 'win':
        win = data['win']
        actor, source, tile = win['player'], win['source_player'], win['tile']
        source = -1 if source is None else source
        flags, amount = int(win['self_draw']), win['value']['payment']
        detail = 'tsumo' if win['self_draw'] else 'ron'
    elif kind == 'round_ended':
        detail = data['reason']
    _append(engine, kind, actor, source, tile, meld, river, flags, amount, detail)


def _append(engine, kind, actor, source, tile, meld, river, flags, amount, detail):
    if len(engine.public_history) >= MAX_PUBLIC_EVENTS:
        raise RuntimeError('public history capacity exceeded; refusing to truncate')
    engine.public_history.append((EVENT_KINDS.index(kind), actor, source, tile, meld,
        river, flags, amount, EVENT_DETAILS.index(detail), engine.state.remaining_tiles,
        sum(p.discard_count for p in engine.state.players)))


def observation_context(engine):
    """Standalone hands explicitly announce their mode instead of guessing it."""
    context = getattr(engine, '_match_context', None)
    if context is None:
        context = {'config': None, 'initial_dealer': engine.config.dealer,
                   'initial_scores': list(engine.config.initial_scores), 'hands_played': 0}
    return {'public_history_version': HISTORY_VERSION,
            'public_history': [list(row) for row in engine.public_history],
            'match_context': context}
