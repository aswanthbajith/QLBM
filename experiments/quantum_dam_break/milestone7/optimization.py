"""Space/time tradeoff: recompute output dependency cones in shared work wires.
No numerical oracle is used. Every emitted gate comes from the M6 circuit.
"""
from functools import lru_cache
from milestone6.arithmetic import FluxOracle
from milestone4.solver import gate


def controls(mask):
    while mask:
        low=mask&-mask
        yield low.bit_length()-1
        mask^=low


def compact_cones(source, inputs, outputs, groups):
    """Partition compute/copy/uncompute by output targets, reusing work wires.
    Valid for M6 fresh-wire Boolean compiler: each fresh target has a contiguous
    compute definition with controls on earlier wires. No output is a control.
    """
    boundary=inputs+outputs
    copy=[g for g in source if inputs<=g[2]<boundary]
    compute=source[:(len(source)-len(copy))//2]
    assert tuple(source)==tuple(compute)+tuple(copy)+tuple(reversed(compute))
    defs={}
    for g in compute:
        assert g[2]>=boundary and g[0]==g[1]
        defs.setdefault(g[2],[]).append(g)
        assert all(c<g[2] and not inputs<=c<boundary for c in controls(g[0]))
    assert sorted(t for group in groups for t in group)==list(range(inputs,boundary))
    result=[]; peak=0; stats=[]
    for group in groups:
        group=set(group); selected=[g for g in copy if g[2] in group]
        need=set(); stack=[c for g in selected for c in controls(g[0]) if c>=boundary]
        while stack:
            t=stack.pop()
            if t in need:continue
            need.add(t)
            stack.extend(c for g in defs[t] for c in controls(g[0]) if c>=boundary)
        mapping={v:boundary+i for i,v in enumerate(sorted(need))}
        def remap(g):
            m,p,t=g;assert m==p
            mm=sum(1<<mapping.get(c,c) for c in controls(m))
            return mm,mm,mapping.get(t,t)
        forward=[remap(g) for g in compute if g[2] in need]
        middle=[remap(g) for g in selected]
        result.extend(forward+middle+forward[::-1]);peak=max(peak,len(need))
        stats.append(dict(work_qubits=len(need),gates=2*len(forward)+len(middle)))
    return tuple(result),peak,stats

class CompactFlux(FluxOracle):
    def __init__(self,scale,sign,partition='components'):
        super().__init__(scale,sign)
        self.original_ancillas=self.ancillas;self.original_gates=len(self.gates)
        if partition=='components':sizes=[self.fb]*3
        elif partition=='bits':sizes=[1]*self.outputs
        elif partition=='normal_other':sizes=[self.fb,2*self.fb]
        else:raise ValueError(partition)
        groups=[];offset=self.inputs
        for size in sizes:groups.append(range(offset,offset+size));offset+=size
        self.gates,self.ancillas,self.group_stats=compact_cones(self.gates,self.inputs,self.outputs,groups)
        self.nqubits=self.inputs+self.outputs+self.ancillas
