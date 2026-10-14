"""Integer-only Numba kernels. Kept separate from dependency-free rules code.

Compile per process: recursive-call relocations in the disk cache crashed when
loaded by a fresh process on the validated runtime. In-memory JIT reuse is safe.
"""
import numpy as np
from numba import njit


@njit(cache=False,nogil=True)
def _regular(c,start,complete,partial,head,best):
    first=start
    while first<34 and c[first]==0:first+=1
    if first==34:
        return min(best,8-2*complete-min(partial,4-complete)-head)
    if best==-1:return best
    if complete<4:
        if c[first]>=3:
            c[first]-=3;best=_regular(c,first,complete+1,partial,head,best);c[first]+=3
        if first<27 and first%9<=6 and c[first+1]>0 and c[first+2]>0:
            c[first]-=1;c[first+1]-=1;c[first+2]-=1
            best=_regular(c,first,complete+1,partial,head,best)
            c[first]+=1;c[first+1]+=1;c[first+2]+=1
    if c[first]>=2:
        c[first]-=2
        if head==0:best=_regular(c,first,complete,partial,1,best)
        if partial<4-complete:best=_regular(c,first,complete,partial+1,head,best)
        c[first]+=2
    if partial<4-complete and first<27:
        for gap in range(1,3):
            if first%9+gap<9 and c[first+gap]>0:
                c[first]-=1;c[first+gap]-=1
                best=_regular(c,first,complete,partial+1,head,best)
                c[first]+=1;c[first+gap]+=1
    c[first]-=1;best=_regular(c,first,complete,partial,head,best);c[first]+=1
    return best


@njit(cache=False,nogil=True)
def shanten_one(c,melds,riichi,quads):
    result=_regular(c,0,melds,0,0,8)
    if melds==0:
        pairs=0;distinct=0;quad_pairs=0
        for k in range(34):
            pairs+=int(c[k]>=2);distinct+=int(c[k]>0);quad_pairs+=c[k]//2
        seven=6-pairs+max(0,7-distinct) if riichi or not quads else 6-quad_pairs
        result=min(result,seven)
        if riichi:
            unique=0;head=0
            for k in (0,8,9,17,18,26,27,28,29,30,31,32,33):
                unique+=int(c[k]>0);head=max(head,int(c[k]>1))
            result=min(result,13-unique-head)
    return result


@njit(cache=False,nogil=True)
def _groups(c,start):
    first=start
    while first<34 and c[first]==0:first+=1
    if first==34:return True
    if c[first]>=3:
        c[first]-=3;win=_groups(c,first);c[first]+=3
        if win:return True
    if first<27 and first%9<=6 and c[first+1]>0 and c[first+2]>0:
        c[first]-=1;c[first+1]-=1;c[first+2]-=1;win=_groups(c,first)
        c[first]+=1;c[first+1]+=1;c[first+2]+=1
        if win:return True
    return False


@njit(cache=False,nogil=True)
def is_win(c,melds,riichi,quads):
    if c.sum()!=14-3*melds:return False
    if melds==0:
        seven=True;pairs=0
        for k in range(34):
            if c[k]%2:seven=False
            pairs+=int(c[k]==2)
        if seven and (not riichi and quads or pairs==7):return True
        if riichi:
            total=0;present=True
            for k in (0,8,9,17,18,26,27,28,29,30,31,32,33):
                total+=c[k];present=present and c[k]>0
            if present and total==14:return True
    for k in range(34):
        if c[k]>=2:
            c[k]-=2;win=_groups(c,0);c[k]+=2
            if win:return True
    return False


@njit(cache=False,nogil=True)
def waits_one(c,melds,riichi,quads):
    mask=np.uint64(0)
    if c.sum()!=13-3*melds:return mask
    for k in range(34 if riichi else 27):
        if c[k]>=4:continue
        c[k]+=1
        if is_win(c,melds,riichi,quads):mask|=np.uint64(1)<<np.uint64(k)
        c[k]-=1
    return mask


@njit(cache=False,nogil=True)
def shanten_many(counts,melds,riichi,quads):
    out=np.empty(len(counts),dtype=np.int8)
    for row in range(len(counts)):out[row]=shanten_one(counts[row],melds[row],riichi,quads)
    return out


@njit(cache=False,nogil=True)
def waits_many(counts,melds,riichi,quads):
    out=np.empty(len(counts),dtype=np.uint64)
    for row in range(len(counts)):out[row]=waits_one(counts[row],melds[row],riichi,quads)
    return out


@njit(cache=False,nogil=True)
def discard_many(counts,known,melds,riichi,quads,best_only,allowed):
    distances=np.full((len(counts),34),127,dtype=np.int8);ukeire=np.zeros((len(counts),34),dtype=np.int16)
    for row in range(len(counts)):
        c=counts[row]
        best=127
        for discard in range(34 if riichi else 27):
            if c[discard]==0:continue
            c[discard]-=1;distance=shanten_one(c,melds[row],riichi,quads);c[discard]+=1
            distances[row,discard]=distance
            if allowed[row,discard]:best=min(best,distance)
        for discard in range(34 if riichi else 27):
            if c[discard]==0 or best_only and (not allowed[row,discard] or distances[row,discard]!=best):continue
            c[discard]-=1;distance=distances[row,discard]
            for incoming in range(34 if riichi else 27):
                if known[row,incoming]>=4:continue
                c[incoming]+=1
                if shanten_one(c,melds[row],riichi,quads)<distance:ukeire[row,discard]+=4-known[row,incoming]
                c[incoming]-=1
            c[discard]+=1
    return distances,ukeire
