#!/bin/sh
# Regenerates every N1 result in dependency order (about two and a half hours on two cores).
# detuned_qpm.py, twofield_*.py and stark_offscan.py also read ../nl02/hw_scan_wg.json (waveguide only, state-independent).
set -e
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd)
cd "$(dirname "$0")/n1_paper"
python3 tables.py > tables.log 2>&1
python3 curves_sc.py > curves.log 2>&1
python3 stepwell.py > stepwell.log 2>&1
python3 waveguide_design.py > wg.log 2>&1
python3 electrode_study.py > el.log 2>&1
python3 rates_study.py > rates.log 2>&1
python3 propagation_lo.py > propagation_lo.log 2>&1
python3 stark_realistic.py > stark_realistic.log 2>&1
python3 propagation_study.py > propagation_study.log 2>&1
python3 adqw_sign.py > adqw_sign.log 2>&1
python3 stark_valid.py > stark_valid.log 2>&1            # needs tables.json, rates.json
python3 linear_grating.py > linear_grating.log 2>&1      # needs electrodes.json, stark_valid.json
python3 sensitivity.py > sensitivity.log 2>&1            # needs stark_valid.json
python3 summary_lo.py > summary_lo.log 2>&1              # needs propagation_lo.json, stark_valid.json
python3 bipolar_fair.py > bipolar_fair.log 2>&1          # needs stark_valid.json
python3 stark_offscan.py > stark_offscan.log 2>&1
python3 detuned_qpm.py > detuned_qpm.log 2>&1
N1_NS_INDEX=2 python3 detuned_qpm.py > detuned_qpm_ns2.log 2>&1
python3 twofield_qpm.py > twofield_qpm.log 2>&1
python3 twofield_tw.py > twofield_tw.log 2>&1
python3 twofield_best.py > twofield_best.log 2>&1        # needs twofield_qpm.json, twofield_tw.json
python3 extras_audit.py > extras_audit.log 2>&1          # needs all of the above and ../nl02/qwkit_validation.json, numbers.json
bash ../n2/run_maps.sh                                   # bias maps at 1e10, 3e10, 2e11 (1e11, 5e11 copied from N2)
bash rerun_realfield.sh                                  # electrical schemes through the electrode fields (Tables 3-6, Fig. 5)
cd ../n1
python3 ladder.py 58 14 34 50 64 10 29 > ladder.log 2>&1
python3 errbudget.py 58 14 34 64 10 29 > errbudget.log 2>&1
python3 scan_sc.py A > sc_A.log 2>&1
python3 scan_sc.py B > sc_B.log 2>&1                      # needs sc_A.json
echo ALL-DONE
