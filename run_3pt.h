#include "spin_mat.h"
#include "gamma_container.h"
#include "run_4pt.h"
#include <stdlib.h>

void compute_tensor_3(Vcomplex * T,
                      SpinMat * wall_prop,  // source to operator
                      SpinMat * point_prop, // sink to operator
                      int nx,               // spatial extent of lattice
                      int block_size,       // sparsening factor at operator
                      int ty);              // operator time
