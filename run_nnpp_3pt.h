#include "spin_mat.h"
#include "gamma_container.h"
#include "color_tensor.h"

void run_nnpp_3pt(SpinMat* wall_prop,      // wall prop at source
                  SpinMat* point_prop,     // point prop at sink
                  Vcomplex * corr,         // 3-point correlator
                  int gamma_index,         // index of gamma to insert
                  int block_size_sparsen,  // sparsening factor at operator
                  int nt, int nx,          // size of lattice
                  int tm,                  // source time
                  int sep,                 // sink - source time
                  int xc, int yc, int zc); // sink spatial coords
