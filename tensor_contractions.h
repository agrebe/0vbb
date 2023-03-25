#include "spin_mat.h"

Vcomplex two_traces(Vcomplex * T,
                    int c1, int c2, int c3, int c4,
                    WeylMat A, WeylMat B);
Vcomplex one_trace(Vcomplex * T,
                   int c1, int c2, int c3, int c4,
                   WeylMat A, WeylMat B);

Vcomplex one_trace_transposed(Vcomplex * T,
                              int c1, int c2, int c3, int c4,
                              WeylMat A, WeylMat B);
