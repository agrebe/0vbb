#include "spin_mat.h"
#include "gamma_container.h"
#include "color_tensor.h"
#include "run_4pt.h"

Vcomplex run_nnpp_4pt(SpinMat * wall_prop,       // prop from source
                      Vcomplex * T,
                      int tp,                    // time of sink
                      int nx,                    // spatial lattice extent
                      int xc, int yc, int zc);   // sink coordinates
