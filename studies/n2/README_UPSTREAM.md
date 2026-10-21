# Electrode-field model from the N2 study

N1's electrical schemes are evaluated with the electrode fields of the N2 study (archive:
https://doi.org/10.5281/zenodo.23444591, folder `studies/n2/`). These files make that model run from this folder alone.

| File | Role | Origin |
|---|---|---|
| `common_n2.py` | field model: Laplace solution of the electrode plane, three stack heights, media per bias | copied from N2 |
| `parts.py` | resumable job runner used by `chimap.py` | copied from N2 |
| `chimap.py` | bias maps: self-consistent state sets with LO-phonon rates, every 5 kV/cm | copied from N2 |
| `chimap.json` | bias map at 5e11 cm^-2 | copied from N2 (generated there by `chimap.py`) |
| `chimap_I2.json` | bias map at 1e11 cm^-2 | copied from N2 (`NS_I=2 chimap.py`) |
| `chimap_I0.json`, `chimap_I1.json`, `chimap_I3.json` | bias maps at 1e10, 3e10, 2e11 cm^-2 | generated here by `run_maps.sh` |

The N1 scripts that use them are `studies/n1_paper/realfield_*.py` (run `studies/n1_paper/rerun_realfield.sh`).
