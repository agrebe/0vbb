#!/bin/bash

sed python-output \
  -e 's/S_txp\[3\*\(.\)+m\] \* S_tmx\[3\*m+\(.\)\]/Y[3\*\1+\2]/g' \
  -e 's/S_txp\[3\*\(.\)+n\] \* S_tmx\[3\*n+\(.\)\]/Y[3\*\1+\2]/g' \
  -e 's/S_tmx_T\[3\*m+\(.\)\] \* S_txp_T\[3\*\(.\)+m\]/Y_T[3\*\2+\1]/g' \
  -e 's/S_tmx_T\[3\*n+\(.\)\] \* S_txp_T\[3\*\(.\)+n\]/Y_T[3\*\2+\1]/g' \
  -e 's/S_tmp/Sl_xw/g' \
  -e 's/ccs \* g5s/cgs/g' \
  -e 's/g5s \* ccs/cgs/g' \
  -e 's/Trace(cgs \* Y\([^)]*\))/Trace(Y\1 * cgs)/g' \
  -e 's/Trace(Y\[3\*\(.\)+\(.\)\] \* \([^)]*\) \* Y_T\[3\*\(.\)+\(.\)\] \* \([^)]*\))/one_trace_transposed(T, \1, \2, \4, \5, \3, \6)/' \
  -e 's/Trace(Y\[3\*\(.\)+\(.\)\] \* \([^)]*\) \* Y\[3\*\(.\)+\(.\)\] \* \([^)]*\))/one_trace(T, \1, \2, \4, \5, \3, \6)/' \
  -e 's/Trace(Y\[3\*\(.\)+\(.\)\] \* \([^)]*\)) \* Trace(Y\[3\*\(.\)+\(.\)\] \* \([^)]*\))/two_traces(T, \1, \2, \4, \5, \3, \6)/' \
  -e 's/\[3\*\(.\)+\(.\)\] \* cgs/_CG[3\*\1+\2]/g' \
  > run_nnpp_3pt.inc