"""Structured, ordered public information for new riichi policies.

State keeps a compact summary, while integer tensors retain every river entry,
individual meld and public event in the current hand. No event is truncated.
"""
import numpy as np
from ..public_history import MAX_PUBLIC_EVENTS, HISTORY_FIELDS, RULE_FLAGS
from . import riichi_features as legacy

FEATURE_VERSION = 'riichi_visible_v2'
HINT_DIM = legacy.HINT_DIM
STATE_DIM = legacy.STATE_DIM + 17 + len(RULE_FLAGS) + 4
RIVER_FIELDS = ('tile_kind','red','flags','caller_relative','winner_relative_bits')
MELD_FIELDS = ('kind','source_relative','tile_0','tile_1','tile_2','tile_3',
               'red_position_bits','called_kind','called_red','added_kind','added_red')
HISTORY_TOKEN_FIELDS = ('kind','actor_relative','source_relative','tile_kind',
    'meld_index','river_index','flags','amount','detail','remaining','discard_count','red')
EXTRA_SHAPES = {'rivers': (4,136,5), 'melds': (4,4,11),
                'public_history': (MAX_PUBLIC_EVENTS,len(HISTORY_FIELDS)+1),
                'history_mask': (MAX_PUBLIC_EVENTS,)}
canonical_mask, canonical_index = legacy.canonical_mask, legacy.canonical_index


def _relative(seat, players):
    return np.where(seat < 0, 4, np.where(seat == 4, 5, (seat-players) % 4))


def encode_batch(batch, *, rules=None, backend='auto', hints=True):
    if not np.all(batch.get('history_complete',False)):
        raise ValueError('riichi_visible_v2 requires complete public history and explicit match context')
    result = legacy.encode_batch(batch, rules=rules, backend=backend, hints=hints,
                                 correct_riichi_markers=True)
    n=len(batch['meta']);own=batch['meta'][:,1].astype(int)
    relative=(own[:,None]+np.arange(4)) % 4;rows=np.arange(n)[:,None]
    context=batch['match_context']
    match=np.concatenate((np.eye(3)[context[:,0]],context[:,1:2]/50000,
        context[:,2:6],np.eye(4)[(context[:,6]-own)%4],
        context[:,7:11][rows,relative]/50000,context[:,11:12]/128),axis=1)
    options=batch['rule_options']/np.array([1]*len(RULE_FLAGS)+[100,4,4,4])
    result['state']=np.concatenate((result['state'],match,options),axis=1).astype(np.float32)
    assert result['state'].shape==(n,STATE_DIM)

    tiles=batch['rivers'][rows,relative];flags=(batch['river_flags'][rows,relative]&7).astype(np.int16)
    declaration=batch['seats'][rows,relative,5]
    flags|=((np.arange(136)[None,None,:]==declaration[:,:,None]) & (tiles>=0)).astype(np.int16)*8
    callers=_relative(batch['river_callers'][rows,relative],own[:,None,None])
    winners=batch['river_winners'][rows,relative];winner_bits=np.zeros_like(winners)
    for seat in range(4):
        winner_bits|=(((winners >> relative[:,seat,None,None])&1) << seat).astype(np.uint8)
    red=batch['red_mask'][np.arange(n)[:,None,None],np.maximum(tiles,0)] & (tiles>=0)
    result['rivers']=np.stack((np.where(tiles>=0,tiles//4,34),red,flags,callers,winner_bits),axis=-1).astype(np.int16)

    raw=batch['melds'][rows,relative];tiles=raw[...,1:5]
    red=batch['red_mask'][np.arange(n)[:,None,None,None],np.maximum(tiles,0)] & (tiles>=0)
    red_bits=(red*(1 << np.arange(4))).sum(axis=-1)
    named=[]
    for field in (6,7):
        tile=raw[...,field]
        named.extend((np.where(tile>=0,tile//4,34),
            batch['red_mask'][np.arange(n)[:,None,None],np.maximum(tile,0)] & (tile>=0)))
    result['melds']=np.concatenate((np.where(raw[...,8]>0,raw[...,0],5)[...,None],
        _relative(raw[...,5],own[:,None,None])[...,None],np.where(tiles>=0,tiles//4,34),
        red_bits[...,None],*[x[...,None] for x in named]),axis=-1).astype(np.int16)

    history=batch['public_history'].copy();valid=np.arange(MAX_PUBLIC_EVENTS)[None,:]<batch['history_lengths'][:,None]
    tiles=history[:,:,3].copy()
    history[:,:,1]=_relative(history[:,:,1],own[:,None])
    history[:,:,2]=_relative(history[:,:,2],own[:,None])
    history[:,:,3]=np.where(tiles>=0,tiles//4,34)
    red=batch['red_mask'][np.arange(n)[:,None],np.maximum(tiles,0)] & (tiles>=0)
    tokens=np.concatenate((history,red[:,:,None]),axis=-1)
    tokens[~valid]=0
    result['public_history']=tokens.astype(np.int32);result['history_mask']=valid
    return result
