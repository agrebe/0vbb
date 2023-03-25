#include "spin_mat.h"
#include "gamma_container.h"
#include "color_tensor.h"

void run_nnpp_3pt(Vcomplex * T,             // precomputed tensor
                  SpinMat * wall_prop,      // source propagator
                  Vcomplex * corr,          // 3-point correlator
                  int nt, int nx,           // size of lattice
                  int sep,                  // sink - source time
                  int tm,                   // source time
                  int t,                    // operator - source time
                  int num_currents,         // {SS, PP, VV, AA, TT}
                  int xc, int yc, int zc);  // sink coordinates
