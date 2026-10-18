# N1 study code: sign-reversal QPM with field-induced intersubband chi(2)

Python 3 + numpy + scipy. Run from this folder.

| File | What it does |
| --- | --- |
| qwchi2.py | 1D effective-mass solver (InGaAs/InAlAs on InP, Kane nonparabolicity), dipoles, resonant chi(2) with Fermi-Dirac subband populations, absorption, WKB tunnelling escape |
| validate.py | Checks: infinite-well limit, book Fig. 15.14 ADQW ladder, chi(2) = 0 at zero field and odd in F for symmetric wells |
| scan2.py | Coarse design scans: `python3 scan2.py adqw|sqw|sdqw|stqw` (writes v2_<family>.json) |
| refine.py | Finer search around the best ADQW and symmetric triple well |
| post.py | Field curves, monolayer sensitivity, screening, coherence length, leakage for the chosen design |
| propagate.py | Book Eq. 15.29 propagation vs doping for no QPM, Stark on/off QPM, sign-reversal QPM, perfect phase matching |

Assumptions are listed in each docstring. Known gaps: no Hartree/depolarisation self-consistency,
no saturation or pump depletion, parabolic 2D density of states for populations, bulk-InP coherence length.
