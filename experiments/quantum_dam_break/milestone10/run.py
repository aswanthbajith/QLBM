"""Run model screening, quantum oracle and multispeed streaming tests."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone10.kinetic import SETS,G,feasible,bounds,ritter,kinetic_strip,equilibrium2d
from milestone10.oracle import MarginalOracle,K
from milestone10.transport import permutation,apply_word_swaps
from milestone3.arithmetic import basis_apply,statevector_apply,signed
from quantum_core import apply_gates,gate_pairs
OUT=Path(__file__).parent/'results'
def csvwrite(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(exist_ok=True);h,u=np.meshgrid(np.linspace(.005,1.5,300),np.linspace(-1,1,401),indexing='ij');x=np.linspace(-1,1,4001);hr,ur=ritter(x,1);wet=(hr>1e-10)&(x>=-np.sqrt(G*1.5));rows=[];allrows=[];profiles={};report={}
    for name,v in SETS.items():
        m=feasible(v,h,u);r=feasible(v,hr,ur);row=dict(lattice=name,directions_2d=len(v)**2,speed_spacing=float(v[1]-v[0]),max_speed=float(v[-1]),sample_feasible_fraction=float(m.mean()),ritter_fan_feasible_fraction=float(r[wet].mean()));rows.append(row)
        xx,hh,rr=kinetic_strip(v);profiles[name]=hh;allrows.extend(dict(lattice=name,**a) for a in rr);report[name]=rr[-1];print(row,rr[-1],flush=True)
    xx,hh,rr=kinetic_strip(SETS['D2Q9'],signed=True);profiles['signed D2Q9 marginal']=hh;allrows.extend(dict(lattice='signed D2Q9 marginal',**a) for a in rr);report['signed D2Q9 marginal']=rr[-1]
    csvwrite('feasibility.csv',rows);csvwrite('strip_diagnostics.csv',allrows);np.savez_compressed(OUT/'profiles.npz',x=xx,**profiles)
    # exact moment test on feasible 2D states, including cross moment
    v=SETS['D2Q25'];rng=np.random.default_rng(10);H=rng.uniform(.005,1.5,10000);ux=rng.uniform(-1,1,10000);uy=rng.uniform(-1,1,10000);ok=feasible(v,H,ux)&feasible(v,H,uy);H,ux,uy=H[ok],ux[ok],uy[ok];f,_=equilibrium2d(v,H,ux,uy);err=[np.max(abs(f.sum((1,2))-H)),np.max(abs(np.einsum('nij,i->n',f,v)-H*ux)),np.max(abs(np.einsum('nij,j->n',f,v)-H*uy)),np.max(abs(np.einsum('nij,i->n',f,v*v)-H*ux**2-G*H**2/2)),np.max(abs(np.einsum('nij,j->n',f,v*v)-H*uy**2-G*H**2/2)),np.max(abs(np.einsum('nij,i,j->n',f,v,v)-H*ux*uy))];assert max(err)<1e-12 and f.min()>=0
    o=MarginalOracle();orows=[];valid=0
    for pat in range(1<<o.nin):
        H=pat&7;J=signed(pat>>o.nh,o.nj);word=basis_apply(pat,o.gates,o.nin);p,flag=o.decode(word);assert basis_apply(word,o.gates[::-1],o.nin)==pat;assert word&255==pat
        if not flag:
            valid+=1;assert p.sum()==H and p@K==J;target=J*J/H+H*H/(4*o.scale) if H else 0;orows.append(dict(depth_ticks=H,momentum_speed_index_ticks=J,**{f'f{i}_ticks':int(a) for i,a in enumerate(p)},target_second_moment_index_ticks=target,actual_second_moment_index_ticks=int(p@(np.array(K)**2)),absolute_error_index_ticks=abs(p@(np.array(K)**2)-target)))
    csvwrite('oracle.csv',orows)
    # explicit statevector evolution for smaller coherent oracle (17 qubits)
    small=MarginalOracle(2,4);s=np.zeros(1<<small.nqubits,complex);a=small.encode(3,1);b=small.encode(2,-2);s[a]=1/np.sqrt(2);s[b]=1j/np.sqrt(2);z=statevector_apply(s,small.gates,small.nin);expected=np.zeros_like(s);expected[basis_apply(a,small.gates,small.nin)]=s[a];expected[basis_apply(b,small.gates,small.nin)]=s[b];coherent_err=float(np.max(abs(z-expected)));inverse_err=float(np.max(abs(statevector_apply(z,small.gates[::-1],small.nin)-s)));assert coherent_err==inverse_err==0
    # D2Q25 transport: exact word SWAP and padded quantum basis gate test
    mask=np.zeros((4,4),bool);mask[1,2]=True;p=permutation(mask,range(-2,3));words=rng.integers(0,20,len(p));words.reshape(25,-1)[:,mask.ravel()]=0;out=apply_word_swaps(words,p);assert out.sum()==words.sum() and not out.reshape(25,-1)[:,mask.ravel()].any();assert np.array_equal(apply_word_swaps(out,np.argsort(p)),words)
    pad=np.arange(1<<(len(p)-1).bit_length());pad[:len(p)]=p;s=rng.normal(size=len(pad))+1j*rng.normal(size=len(pad));s/=np.linalg.norm(s);z=apply_gates(s,pad);assert np.allclose(z,s[np.argsort(pad)]);result=dict(feasibility=rows,strip_final=report,product_equilibrium={'tested_feasible_states':int(len(ux)),'max_moment_error':float(max(err)),'moment_errors_h_qx_qy_pxx_pyy_pxy':[float(e) for e in err]},oracle={'qubits':o.nqubits,'logical_pattern_controlled_x':len(o.gates),'input_patterns':1<<o.nin,'mass_momentum_feasible_patterns':valid,'invalid_flagged':(1<<o.nin)-valid,'max_stress_error_index_ticks':max(r['absolute_error_index_ticks'] for r in orows),'smaller_statevector_qubits':small.nqubits,'smaller_statevector_gates':len(small.gates),'coherent_error':coherent_err,'inverse_error':inverse_err},transport={'directions':25,'grid':[4,4],'padded_qubits':int((len(pad)-1).bit_length()),'pattern_controlled_x_gates':sum(1 for _ in gate_pairs(pad)),'mass_error_ticks':int(out.sum()-words.sum()),'obstacle_population_ticks':int(out.reshape(25,-1)[:,mask.ravel()].sum()),'max_statevector_error':float(np.max(abs(z-s[np.argsort(pad)])))},backend='floating kinetic candidate screening separately; actual pattern-controlled X basis and small statevector simulation for oracle and transport only; no integrated quantum fluid solver')
    (OUT/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
