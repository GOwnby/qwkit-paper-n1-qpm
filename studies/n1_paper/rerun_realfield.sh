#!/bin/bash
# N1 electrical schemes with the electrode fields (LATER_STAGE 51). Resumable: finished jobs are kept in
# parts_realfield/. Needs ../n2/chimap*.json (bias maps; ../n2/run_maps.sh makes the 1e10, 3e10 and 2e11 ones).
cd "$(dirname "$0")"
export PYTHONPATH=$(cd ../.. && pwd) OMP_NUM_THREADS=1 NPROC=${NPROC:-2}
for s in realfield_eff realfield_twofield realfield_detuned realfield_sens; do
  python3 -W ignore $s.py >> $s.log 2>&1 || { echo "$s failed"; exit 1; }
done
python3 -W ignore realfield_summary.py > /dev/null      # writes realfield_summary.log, realfield_tables.tex
