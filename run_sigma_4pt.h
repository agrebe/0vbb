#include "spin_mat.h"
#include "gamma_container.h"
#include "color_tensor.h"

void run_sigma_4pt(SpinMat * wall_prop,       // prop from source
                   SpinMat * point_prop,      // prop from sink
                   SpinMat * SnuHz,           // seqprop * nu_prop
                   Vcomplex * corr_sigma_4pt, // output (4-point correlator)
                   int tx,                    // time of operator
                   int tp,                    // time of sink
                   int nx,                    // spatial lattice extent
                   int block_size,            // sparsening at operator
                   int xc, int yc, int zc);   // sink coordinates
