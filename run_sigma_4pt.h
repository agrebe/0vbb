#include "spin_mat.h"
#include "gamma_container.h"
#include "color_tensor.h"
#include "run_4pt.h"

Vcomplex run_sigma_4pt(SpinMat * wall_prop,       // prop from source
                       Vcomplex * T,
                       int tx,                    // time of operator
                       int tp,                    // time of sink
                       int nx,                    // spatial lattice extent
                       int block_size,            // sparsening at operator
                       int xc, int yc, int zc);   // sink coordinates

Vcomplex two_traces(Vcomplex * T,
                    int c1, int c2, int c3, int c4,
                    WeylMat A, WeylMat B);
Vcomplex one_trace(Vcomplex * T,
                   int c1, int c2, int c3, int c4,
                   WeylMat A, WeylMat B);

Vcomplex one_trace_transposed(Vcomplex * T,
                              int c1, int c2, int c3, int c4,
                              WeylMat A, WeylMat B);
