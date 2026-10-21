#!/bin/bash
# Bias maps of the N1 media at the three densities N2 did not need (tables.json indices 0, 1, 3: 1e10, 3e10, 2e11 cm^-2).
cd "$(dirname "$0")"
export PYTHONPATH=$(cd ../.. && pwd) OMP_NUM_THREADS=1 NPROC=${NPROC:-2}
for I in 0 1 3; do [ -f chimap_I$I.json ] || { NS_I=$I python3 -W ignore chimap.py > chimap_I$I.log 2>&1; echo "exit $?" >> chimap_I$I.log; }; done
