#include <omp.h>
#include "read_prop.h"
#include "spin_mat.h"
#include "run_meson_2pt.h"
#include "run_baryon_2pt.h"
#include "color_tensor.h"
#include "gamma_container.h"

int main() {
  double dtime0 = omp_get_wtime();
  // initialize gamma matrices and epsilon tensors
  initialize_gammas();
  color_tensor_1body(color_idx_1);
  color_tensor_2bodies(color_idx_2);
  double dtime1 = omp_get_wtime();

  // read in propagator
  char filename [100] = "../qio_propagator.lime.contents/msg02.rec03.scidac-binary-data";
  int nt = 48;
  int nx = 32;
  int vol = nt * nx * nx * nx;
  SpinMat * prop = (SpinMat*) malloc(vol * 9 * sizeof(SpinMat));
  read_prop(filename, prop, nt, nx);
  double dtime2 = omp_get_wtime();
  
  // compute pion correlator (for testing)
  Vcomplex * corr = (Vcomplex *) malloc(nt * sizeof(Vcomplex));
  run_pion_correlator_wsink(prop, corr, nt, nx);
  //for (int t = 0; t < nt; t ++) printf("corr[%d] = %f\n", t, corr[t].real());
  double dtime3 = omp_get_wtime();

  // compute neutron correlator
  int block_size = 4;
  run_neutron_correlator(prop, corr, nt, nx, block_size);
  //for (int t = 0; t < nt; t ++) printf("corr[%d] = %f + %fi\n", t, corr[t].real(), corr[t].imag());
  double dtime4 = omp_get_wtime();

  printf("----------------------------------------------------------------------\n");
  printf("0. Initialization:                                   %17.10e s\n", dtime1 - dtime0);
  printf("1. Reading propagators:                              %17.10e s\n", dtime2 - dtime1);
  printf("2. Pion 2-point correlator:                          %17.10e s\n", dtime3 - dtime2);
  printf("3. Neutron 2-point correlator:                       %17.10e s\n", dtime4 - dtime3);
  printf("----------------------------------------------------------------------\n");
}
