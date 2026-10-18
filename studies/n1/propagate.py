"""Eq. 15.29 propagation (undepleted pump, below saturation) for SHG in a waveguide, with QPM patterns.
Modal ISB absorption = Gc * material alpha; background modal loss abg at w and 2w; Gc also scales chi.
Scans sheet density per period; chi and alpha recomputed with 300 K Fermi populations at each density."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); import numpy as np, json; from qwkit import core as q
HW=1.23984/10; G=0.010; Gc=0.5; ABG=3.0; LCOH=63.06e-4  # cm
def tw(g): return [(g[0],"well"),(g[1],"barrier"),(g[2],"well"),(g[1],"barrier"),(g[0],"well")]
def ad(g): return [(g[0],"well"),(g[1],"barrier"),(g[2],"well")]
def props(lay,F,ns):
    r=q.analyze(lay,F,HW)                                       # states (density-independent)
    Lp=(r['active'][1]-r['active'][0]+200.0)*1e-8; N=ns/Lp
    rho=q.populations(r['E'],ns,300.0)
    chi=abs(q.chi2_general(r['E'],r['Z'],rho,HW,G,N))
    # absorption with populations: sum over l<m (rho_l - rho_m) transitions
    aw=a2=0.0
    for hw,acc in ((HW,'w'),(2*HW,'2w')):
        tot=0.0
        for l in range(len(r['E'])):
            for m in range(l+1,len(r['E'])):
                Em=r['E'][m]-r['E'][l]; L=(G/np.pi)/((Em-hw)**2+G**2)/q.E_CH
                tot+=(rho[l]-rho[m])*(r['Z'][l,m]*1e-10)**2*L
        w=hw*q.E_CH/1.054571817e-34
        a=np.pi*w*N*1e6*q.E_CH**2/(3.1*2.99792458e8*q.EPS0)*tot/100
        if acc=='w': aw=a
        else: a2=a
    return chi,aw,a2
def peak(chi,aw,a2,pattern):
    dx=0.25e-4; x=np.arange(0,0.6,dx)                           # up to 6 mm
    aw_t=Gc*aw+ABG; a2_t=Gc*a2+ABG
    dk=0.0 if pattern=="perfect" else np.pi/LCOH
    seg=np.floor(x/LCOH).astype(int)
    s={"none":np.ones_like(x),"perfect":np.ones_like(x),"sign":np.where(seg%2==0,1.0,-1.0),"onoff":np.where(seg%2==0,1.0,0.0)}[pattern]
    integrand=s*Gc*chi*np.exp(-aw_t*x+a2_t*x/2+1j*dk*x)
    E2=np.exp(-a2_t*x/2)*np.cumsum(integrand)*dx
    i=np.argmax(np.abs(E2)); return float(abs(E2[i])**2), float(x[i]*1e4)
out=[]
for ns in np.logspace(9,np.log10(5e11),15):
    A=props(ad((64,10,30)),0.0,ns); T=props(tw((58,14,34)),50.0,ns)
    row=dict(ns=ns,A=A,T=T)
    for name,(p,pat) in {"adqw_none":(A,"none"),"adqw_onoff":(A,"onoff"),"adqw_perfect":(A,"perfect"),"tw_sign":(T,"sign"),"tw_none":(T,"none")}.items():
        row[name]=peak(*p,pat)
    out.append(row)
ref=out[-1]["adqw_none"][0]
for r in out:
    print(f"ns={r['ns']:.2e}  aw(ADQW)={r['A'][1]:7.1f}  " + "  ".join(f"{k}={r[k][0]/ref:8.3f}@{r[k][1]:5.0f}um" for k in ("adqw_none","adqw_onoff","tw_sign","adqw_perfect")))
json.dump(dict(rows=out,ref=ref,params=dict(Gc=Gc,abg=ABG,Lcoh_um=LCOH*1e4)),open('prop.json','w'),default=float)
