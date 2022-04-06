#include <math.h>
#include "spin_mat.h"
#include "gamma_container.h"

void assemble_Hvec(SpinMat * Hvec,           // sequential propagator
                   SpinMat * wall_prop,      // source to operator
                   SpinMat * point_prop,     // sink to operator
                   int nx,                   // spatial extent of lattice
                   int block_size,           // sparsening factor at operator
                   int tm,                   // source time
                   int tp,                   // sink time
                   int ty);                  // operator time

static int dist_sq(int y, int z, int nx);

static double nu_prop(int y1, int y2, int y3, int y4, // first point
                      int z1, int z2, int z3, int z4, // second point
                      int nx, int nt,                 // spatial and temporal extent
                      int global_sparsening);         // global sparsening factor


void compute_SnuHz(SpinMat * SnuHz,         // seqprop * nu_prop
                   SpinMat * Hvec,          // seqprop
                   int tx, int ty,          // operator times
                   int nx, int nt,          // spatial and temporal extent
                   int block_size,          // sparsening factor at operator
                   int global_sparsening);  // global sparsening factor
