#!/bin/bash

sed python-output \
  -e 's/Sl_zw\[3\*\(.\)+n\] \* gnu \* pls \* Sl_xz\[3\*n+\(.\)\]/Z[3\*\1+\2]/g' \
  -e 's/Sl_xz_T\[3\*n+\(.\)\] \* pls.transpose() \* gnu.transpose() \* Sl_zw_T\[3\*\(.\)+n\]/Z_T[3\*\2+\1]/g' \
  -e 's/Sl_yw\[3\*\(.\)+m\] \* gmu \* pls \* Sl_xy\[3\*m+\(.\)\]/Y[3\*\1+\2]/g' \
  -e 's/ccs \* g5s/cgs/g' \
  -e 's/g5s \* ccs/cgs/g' \
  -e 's/Trace(cgs \* Y\([^)]*\))/Trace(Y\1 * cgs)/g' \
  -e 's/Trace(cgs \* Z\([^)]*\))/Trace(Z\1 * cgs)/g' \
  -e 's/Trace(Y\[3\*\(.\)+\(.\)\] \* \([^)]*\) \* Z_T\[3\*\(.\)+\(.\)\] \* \([^)]*\))/one_trace_transposed(T, \1, \2, \4, \5, \3, \6)/' \
  -e 's/Trace(Y\[3\*\(.\)+\(.\)\] \* \([^)]*\) \* Z\[3\*\(.\)+\(.\)\] \* \([^)]*\))/one_trace(T, \1, \2, \4, \5, \3, \6)/' \
  -e 's/Trace(Y\[3\*\(.\)+\(.\)\] \* \([^)]*\)) \* Trace(Z\[3\*\(.\)+\(.\)\] \* \([^)]*\))/two_traces(T, \1, \2, \4, \5, \3, \6)/' \
  -e 's/\[3\*\(.\)+\(.\)\] \* cgs/_CG[3\*\1+\2]/g' \
  > run_nnpp_4pt.inc
