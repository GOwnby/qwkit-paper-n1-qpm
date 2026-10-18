import numpy as np, itertools, json
from multiprocessing import Pool
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); from qwkit import core as q
HW = 1.23984/10

def tw(g): return [(g[0],"well"),(g[1],"barrier"),(g[2],"well"),(g[1],"barrier"),(g[0],"well")]
def ad(g): return [(g[0],"well"),(g[1],"barrier"),(g[2],"well")]

def run_t(a):
    g,F=a
    r=q.analyze_T(tw(g),F,HW,dz=0.5)
    esc=max(q.escape_broadening(r,k,F)[0] for k in range(min(3,len(r['E']))))
    return dict(g=list(g),F=F,chi=abs(r['chiT'][300.0]),chi77=abs(r['chiT'][77.0]),esc=float(esc))
def run_a(g):
    r=q.analyze_T(ad(g),0.0,HW,dz=0.5)
    return dict(g=list(g),chi=abs(r['chiT'][300.0]),chi77=abs(r['chiT'][77.0]))

if __name__=="__main__":
    with Pool(2) as p:
        A=p.map(run_a, list(itertools.product(range(58,74,2),range(6,20,2),range(24,34,2))))
        A.sort(key=lambda r:-r['chi']); print("ADQW best:",A[:3],flush=True)
        T=p.map(run_t, [(g,float(F)) for g in itertools.product(range(52,70,2),range(8,18,2),range(26,40,2)) for F in range(15,65,5)])
    T=[t for t in T if t['esc']<=1.0]; T.sort(key=lambda r:-r['chi'])
    print("TW best:",T[:5])
    json.dump(dict(A=A,T=T),open("refine.json","w"))
