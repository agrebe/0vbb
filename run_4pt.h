#include <math.h>
#include <fftw3.h>
#include "spin_mat.h"
#include "gamma_container.h"

void assemble_Hvec(WeylMat * Hvec,           // sequential propagator
                   SpinMat * wall_prop,      // source to operator
                   SpinMat * point_prop,     // sink to operator
                   int nx,                   // spatial extent of lattice
                   int block_size,           // sparsening factor at operator
                   int tm,                   // source time
                   int tp,                   // sink time
                   int ty);                  // operator time

static int dist_sq(int y, int z, int nx);

double nu_prop(int x, int y, int z, int tau,   // separation between current insertions
                      int nx,                  // spatial extent
                      int global_sparsening);  // global sparsening factor

int tensor_index(int c1, int c2, int c3, int c4,
                 int s1, int s2, int s3, int s4);

void compute_tensor(Vcomplex * T,
                    WeylMat * Hvec_x,
                    WeylMat * Hvec_y,
                    fftw_complex * nu_F,
                    int nx, int nt,
                    int block_size,
                    int global_sparsening);

void compute_SnuHz(WeylMat * SnuHz,         // seqprop * nu_prop
                   WeylMat * Hvec,          // seqprop
                   int tx, int ty,          // operator times
                   int nx, int nt,          // spatial and temporal extent
                   int block_size,          // sparsening factor at operator
                   int global_sparsening);  // global sparsening factor
