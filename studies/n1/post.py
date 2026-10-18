import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); import numpy as np, json; from qwkit import core as q
HW=1.23984/10; G=0.010
TW=(58,14,34); AD=(64,10,30)
def tw(g): return [(g[0],"well"),(g[1],"barrier"),(g[2],"well"),(g[1],"barrier"),(g[0],"well")]
def ad(g): return [(g[0],"well"),(g[1],"barrier"),(g[2],"well")]
out={}
ra=q.analyze_T(ad(AD),0.0,HW); ref=abs(ra['chiT'][300.0]); ref77=abs(ra['chiT'][77.0])
out['ref']=dict(chi300=ref,chi77=ref77,E=list((ra['E']-ra['E'][0])*1e3),N=ra['N_cm3'])
# 1) chi vs F curves
curves={}
for name,lay,Fs in [("Symmetric triple well 58/14/34",tw(TW),range(0,105,5)),
                    ("Symmetric double well 56/6",[(56,"well"),(6,"barrier"),(56,"well")],range(0,205,10)),
                    ("Single well 116",[(116,"well")],range(0,205,10))]:
    c=[]
    for F in Fs:
        try:
            r=q.analyze_T(lay,float(F),HW)
            esc=max(q.escape_broadening(r,k,F)[0] for k in range(min(3,len(r['E'])))) if F else 0
            c.append(dict(F=F,pct=abs(r['chiT'][300.0])/ref*100,esc=float(esc)))
        except Exception as ex: pass
    curves[name]=c
out['curves']=curves
# 2) design point details
r=q.analyze_T(tw(TW),50.0,HW)
E=(r['E']-r['E'][0])*1e3; Z=r['Z']
out['tw']=dict(E=list(E),z12=Z[0,1],z13=Z[0,2],z14=Z[0,3],z23=Z[1,2],z34=Z[2,3],z24=Z[1,3],rho300=list(r['rhoT'][300.0][:3]),N=r['N_cm3'],
               chi300=abs(r['chiT'][300.0]),chi77=abs(r['chiT'][77.0]),alpha=r['alpha'])
r0=q.analyze_T(tw(TW),0.0,HW); out['tw0']=dict(E=list((r0['E']-r0['E'][0])*1e3))
# 3) monolayer sensitivity (+-3 A on each layer type), at fixed 50 kV/cm and with field re-tuned 30-70
sens=[]
for i,lab in enumerate(["outer wells","barriers","centre well"]):
    for d in (-3,3):
        g=list(TW); g[i]+=d
        fixed=abs(q.analyze_T(tw(g),50.0,HW)['chiT'][300.0])/ref*100
        best=max(abs(q.analyze_T(tw(g),float(F),HW)['chiT'][300.0]) for F in range(30,75,5))/ref*100
        sens.append(dict(layer=lab,delta=d,fixed=fixed,retuned=best))
out['sens']=sens
# 4) screening: population-weighted <z> shift between F=0 and 50 kV/cm
dz_=r['z'][1]-r['z'][0]
def zmean(rr):
    zc=0.5*(rr['active'][0]+rr['active'][1]); rho=rr['rhoT'][300.0]
    return sum(rho[k]*rr['Z'][k,k] for k in range(len(rho)))
dZ=abs(zmean(r)-zmean(r0))   # A
ns=5e11*1e4; Lp=(2*58+2*14+34+200)*1e-10; eps=13.9*8.8541878e-12
dF=1.602e-19*ns*dZ*1e-10/(eps*Lp)/1e5
out['screen']=dict(dZ_A=dZ,dF_kVcm=dF,Lp_A=Lp*1e10)
# 5) coherence length: InP Sellmeier (Pettit & Turner) + resonant intersubband index at w and 2w
def nInP(l): return np.sqrt(7.255+2.316*l**2/(l**2-0.6263**2)+2.765*l**2/(l**2-32.935**2))
def dn_isb(rr,hw):
    rho=rr['rhoT'][300.0]; Ecur=rr['E']; Zc=rr['Z']; N=rr['N_cm3']*1e6
    s=0j
    for l in range(len(Ecur)):
        for m in range(len(Ecur)):
            if m==l: continue
            s+=(rho[l]-rho[m])*Zc[l,m]**2*1e-20/((Ecur[m]-Ecur[l]-hw-1j*G)*1.602e-19)
    chi1=N*(1.602e-19)**2/8.8541878e-12*s
    return chi1.real/(2*nInP(1.23984/hw)), chi1.imag
nb_w, nb_2w = nInP(10.0), nInP(5.0)
dnw,_=dn_isb(r,HW); dn2w,_=dn_isb(r,2*HW)
for k,(a,b) in {"bulk":(0,0),"with_ISB_full_overlap":(dnw,dn2w),"with_ISB_30pct_overlap":(0.3*dnw,0.3*dn2w)}.items():
    Dn=(nb_2w+b)-(nb_w+a); out.setdefault('lcoh',{})[k]=dict(dn=Dn,Lcoh_um=10.0/(4*abs(Dn)))
out['lcoh']['dn_isb']=dict(w=dnw,two_w=dn2w)
# 6) thermionic leakage estimate over the lowered downhill barrier at 300 K
U=q.MAT['barrier']['V']-50*1e-5*0.5*(r['active'][1]-r['active'][0])-r['E'][0]
J=1.602e-19*ns*1e13*np.exp(-U/0.02585)/1e4   # A/cm^2, attempt 1e13 /s
out['leak']=dict(U_eV=U,J_Acm2=J)
json.dump(out,open('post.json','w'),default=float,indent=1)
print(json.dumps({k:v for k,v in out.items() if k!='curves'},default=lambda x: round(float(x),3),indent=0))
