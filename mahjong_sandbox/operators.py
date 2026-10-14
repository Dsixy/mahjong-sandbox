"""Batched, typed tile operators. NumPy/Numba are optional performance extras."""
from functools import lru_cache
from .tiles import validate_counts


def numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise ImportError('batch arrays require numpy; install mahjong-sandbox[batch]') from error
    return np


@lru_cache(maxsize=1)
def native_available():
    try:
        import numba
        import numpy
        return True
    except ImportError:
        return False


def _settings(counts,meld_count,variant,seven_pairs_quads,backend):
    if type(meld_count) is not int or not 0<=meld_count<=4 or variant not in ('riichi','sichuan'):
        raise ValueError('invalid meld count or variant')
    if variant=='sichuan' and any(counts[27:]): raise ValueError('sichuan has no honors')
    if backend not in ('auto','python','numba'):raise ValueError('unknown operator backend')
    if type(seven_pairs_quads) is not bool:raise ValueError('seven_pairs_quads must be boolean')
    if sum(counts)>14-3*meld_count:raise ValueError('too many concealed tiles for meld count')


def shanten(counts, meld_count=0, *, variant='riichi', seven_pairs_quads=True, backend='auto'):
    counts=validate_counts(counts)
    _settings(counts,meld_count,variant,seven_pairs_quads,backend)
    return _shanten(counts,meld_count,variant,seven_pairs_quads,backend)


@lru_cache(maxsize=100000)
def _shanten(counts,meld_count,variant,seven_pairs_quads,backend):
    if backend=='numba' or backend=='auto' and native_available():
        from ._kernels import shanten_one
        return int(shanten_one(numpy().asarray(counts,dtype='int16'),meld_count,variant=='riichi',seven_pairs_quads))
    if variant=='riichi':
        from .riichi_agent import riichi_shanten
        return riichi_shanten(counts,meld_count)
    from .greedy import shape_distance
    return shape_distance(counts,meld_count,seven_pairs_quads=seven_pairs_quads)


def structural_waits(counts, meld_count=0, *, variant='riichi', seven_pairs_quads=True, backend='auto'):
    counts=validate_counts(counts)
    _settings(counts,meld_count,variant,seven_pairs_quads,backend)
    return _waits(counts,meld_count,variant,seven_pairs_quads,backend)


@lru_cache(maxsize=100000)
def _waits(counts,meld_count,variant,seven_pairs_quads,backend):
    if backend=='numba' or backend=='auto' and native_available():
        from ._kernels import waits_one
        mask=int(waits_one(numpy().asarray(counts,dtype='int16'),meld_count,variant=='riichi',seven_pairs_quads))
        return tuple(k for k in range(34) if mask>>k&1)
    if variant=='riichi':
        from .riichi_scoring import riichi_waits
        return riichi_waits(counts,meld_count)
    from .hands import waits
    return waits(counts,meld_count,seven_pairs_quads=seven_pairs_quads)


shanten.cache_clear=_shanten.cache_clear
structural_waits.cache_clear=_waits.cache_clear


def _inputs(counts,meld_counts,variant):
    np=numpy();values=np.asarray(counts)
    if values.ndim!=2 or values.shape[1]!=34 or values.dtype.kind not in 'iu' or np.any(values<0) or np.any(values>4):
        raise ValueError('counts must be an integer [N,34] array in 0..4')
    if variant not in ('riichi','sichuan') or variant=='sichuan' and np.any(values[:,27:]):raise ValueError('invalid variant / honors')
    melds=np.asarray(meld_counts)
    if melds.ndim==0:melds=np.full(len(values),melds,dtype=melds.dtype)
    if melds.shape!=(len(values),) or melds.dtype.kind not in 'iu' or np.any(melds<0) or np.any(melds>4):raise ValueError('meld_counts must be N integers in 0..4')
    if np.any(values.sum(axis=1)>14-3*melds):raise ValueError('too many concealed tiles for meld counts')
    return np.array(values,dtype=np.int16,copy=True,order='C'),np.ascontiguousarray(melds,dtype=np.int8)


def shanten_batch(counts,meld_counts=0,*,variant='riichi',seven_pairs_quads=True,backend='auto'):
    np=numpy();values,melds=_inputs(counts,meld_counts,variant)
    if backend not in ('auto','python','numba'):raise ValueError('unknown operator backend')
    if backend=='numba' or backend=='auto' and native_available():
        from ._kernels import shanten_many
        return shanten_many(values,melds,variant=='riichi',seven_pairs_quads)
    return np.array([shanten(tuple(map(int,c)),int(m),variant=variant,seven_pairs_quads=seven_pairs_quads,backend='python') for c,m in zip(values,melds)],dtype=np.int8)


def waits_batch(counts,meld_counts=0,*,variant='riichi',seven_pairs_quads=True,backend='auto'):
    np=numpy();values,melds=_inputs(counts,meld_counts,variant)
    if backend not in ('auto','python','numba'):raise ValueError('unknown operator backend')
    if backend=='numba' or backend=='auto' and native_available():
        from ._kernels import waits_many
        masks=waits_many(values,melds,variant=='riichi',seven_pairs_quads)
        return ((masks[:,None] >> np.arange(34,dtype=np.uint64)) & 1).astype(np.bool_)
    result=np.zeros((len(values),34),dtype=np.bool_)
    for row,(c,m) in enumerate(zip(values,melds)):
        result[row,list(structural_waits(tuple(map(int,c)),int(m),variant=variant,seven_pairs_quads=seven_pairs_quads,backend='python'))]=True
    return result


def discard_ukeire_batch(counts,known_counts,meld_counts=0,*,variant='riichi',seven_pairs_quads=True,backend='auto',best_only=False,candidate_mask=None):
    """Return structural distances and visible-information ukeire for all 34 discards.

    Absent discards have distance=127, ukeire=0. known_counts includes own tiles and
    all publicly known physical tiles exactly once; no wall or opponent hands.
    best_only computes ukeire only for minimal-distance candidate discards;
    distances are still returned for every present kind. Other ukeire is zero.
    """
    np=numpy();values,melds=_inputs(counts,meld_counts,variant);known=np.asarray(known_counts)
    if known.shape!=values.shape or known.dtype.kind not in 'iu' or np.any(known<values) or np.any(known>4):raise ValueError('known counts must include own counts and be in 0..4')
    allowed=values>0 if candidate_mask is None else np.asarray(candidate_mask)
    if allowed.shape!=values.shape or allowed.dtype.kind!='b' or type(best_only) is not bool:raise ValueError('candidate_mask must be boolean [N,34] and best_only boolean')
    if backend not in ('auto','python','numba'):raise ValueError('unknown operator backend')
    if backend=='numba' or backend=='auto' and native_available():
        from ._kernels import discard_many
        return discard_many(values,np.ascontiguousarray(known,dtype=np.int16),melds,variant=='riichi',seven_pairs_quads,best_only,np.ascontiguousarray(allowed))
    distances=np.full((len(values),34),127,dtype=np.int8);ukeire=np.zeros((len(values),34),dtype=np.int16)
    for row,c in enumerate(values):
        for discard in range(34 if variant=='riichi' else 27):
            if not c[discard]:continue
            after=c.copy();after[discard]-=1
            distances[row,discard]=shanten(tuple(map(int,after)),int(melds[row]),variant=variant,seven_pairs_quads=seven_pairs_quads,backend='python')
        best=min((int(distances[row,k]) for k in range(34) if allowed[row,k] and c[k]),default=127)
        for discard in range(34 if variant=='riichi' else 27):
            if not c[discard] or best_only and (not allowed[row,discard] or distances[row,discard]!=best):continue
            after=c.copy();after[discard]-=1;d=int(distances[row,discard])
            for incoming in range(34 if variant=='riichi' else 27):
                if known[row,incoming]>=4:continue
                after[incoming]+=1
                if shanten(tuple(map(int,after)),int(melds[row]),variant=variant,seven_pairs_quads=seven_pairs_quads,backend='python')<d:ukeire[row,discard]+=4-known[row,incoming]
                after[incoming]-=1
    return distances,ukeire
