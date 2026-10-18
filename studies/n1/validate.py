import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); import numpy as np; from qwkit import core as q
# 1) Infinite-well limit: very high barrier, same mass, parabolic -> E_n = HB2M0/m * (n pi / L)^2
q.MAT["barrier"].update(V=50.0, m=0.043, Eg=0.75)
z,V0,m,Eg,act = q.build([(100,"well")], pad=30)
E,psi = q.solve_states(z,V0,m,Eg,nstates=3,nonparabolic=False)
Eth = q.HB2M0/0.043*(np.arange(1,4)*np.pi/100)**2
print("infinite-well check (meV):", np.round(E*1e3,2), "analytic:", np.round(Eth*1e3,2))
Z = q.dipoles(z,psi); print("z12 (A):", round(abs(Z[0,1]),3), "analytic 16L/(9pi^2)=", round(16*100/(9*np.pi**2),3))
q.MAT["barrier"].update(V=0.52, m=0.075, Eg=1.45)
# 2) Book Fig 15.14 ADQW: 64 A InGaAs / 18 A InAlAs / 33 A InGaAs
lay=[(64,"well"),(18,"barrier"),(33,"well")]
for npar in (False, True):
    z,V0,m,Eg,act = q.build(lay)
    E,psi = q.solve_states(z,V0,m,Eg,nstates=4,nonparabolic=npar)
    Z=q.dipoles(z,psi)
    print("nonparabolic" if npar else "parabolic", "E21,E31 (meV):", round((E[1]-E[0])*1e3,1), round((E[2]-E[0])*1e3,1),
          "| z12 z23 z13 (A):", np.round([abs(Z[0,1]),abs(Z[1,2]),abs(Z[0,2])],2))
# 3) symmetry check: symmetric DQW at F=+50 and -50 -> chi opposite sign, same |chi| and energies
hw=1.23984/10
for F in (50,-50,0):
    r=q.analyze([(50,"well"),(20,"barrier"),(50,"well")],F,hw)
    print("sym DQW F=%+d"%F, "E21,E31:",np.round((r['E'][1:3]-r['E'][0])*1e3,1), "chi pm/V:", np.round(r['chi'],0))
