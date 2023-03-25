#include "spin_mat.h"
#include "gamma_container.h"
#include "color_tensor.h"

Vcomplex run_sigma_4pt(SpinMat * wall_prop,       // prop from source
                       Vcomplex * T,
                       int tp,                    // time of sink
                       int nx,                    // spatial lattice extent
                       int xc, int yc, int zc);   // sink coordinates

