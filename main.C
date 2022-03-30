#include <omp.h>
#include "read_prop.h"
#include "spin_mat.h"
#include "run_meson_2pt.h"
#include "run_baryon_2pt.h"
#include "run_dibaryon_2pt.h"
#include "run_sigma_3pt.h"
#include "run_nnpp_3pt.h"
#include "color_tensor.h"
#include "gamma_container.h"

int main() {
  // global variables
  int nt = 48;
  int nx = 32;
  int vol = nt * nx * nx * nx;

  // sparsening factors
  int block_size = 32;        // sparsening at sink
  int block_size_sparsen = 4; // sparsening at operator
  int global_sparsening = 1;  // ratio between nx and actual size of lattice
                              // this is the amount by which props have already been sparsened

  double dtime0 = omp_get_wtime();
  // initialize gamma matrices and epsilon tensors
  initialize_gammas();
  color_tensor_1body(color_idx_1);
  color_tensor_2bodies(color_idx_2);
  double dtime1 = omp_get_wtime();

  // read in propagators
  char wall_filename [100] = "../qio_propagator.lime.contents/msg02.rec03.scidac-binary-data";
  char point_filename [100] = "../qio_propagator_point.lime.contents/msg02.rec03.scidac-binary-data";
  SpinMat * wall_prop = (SpinMat*) malloc(vol * 9 * sizeof(SpinMat));
  SpinMat * point_prop = (SpinMat*) malloc(vol * 9 * sizeof(SpinMat));
  read_prop(wall_filename, wall_prop, nt, nx);
  read_prop(point_filename, point_prop, nt, nx);
  double dtime2 = omp_get_wtime();
  
  // compute pion correlator (for testing)
  Vcomplex corr [nt];
  run_pion_correlator_wsink(wall_prop, corr, nt, nx);
  //for (int t = 0; t < nt; t ++) printf("corr[%d] = %f\n", t, corr[t].real());
  double dtime3 = omp_get_wtime();

  // compute neutron correlator
  run_neutron_correlator(wall_prop, corr, nt, nx, block_size);
  //for (int t = 0; t < nt; t ++) printf("corr[%d] = %f + %fi\n", t, corr[t].real(), corr[t].imag());
  double dtime4 = omp_get_wtime();

  // compute dineutron correlator
  run_dineutron_correlator_PP(wall_prop, corr, nt, nx, block_size);
  //for (int t = 0; t < nt; t ++) printf("corr[%d] = %f + %fi\n", t, corr[t].real(), corr[t].imag());
  double dtime5 = omp_get_wtime();

  // compute sigma 3-point function
  int min_source = 0;
  int max_source = 0;
  int tp = 12; // sink time
  // correlator should store source-sink sep and source-op sep
  // access with corr_sigma_3pt[((tp-tm) * nt + (t-tm) * nt) + i]
  Vcomplex corr_sigma_3pt[nt * nt * 16];
  for (int i = 0; i < nt * nt * 16; i ++)
    corr_sigma_3pt[i] = Vcomplex();
  for (int tm = min_source; tm <= max_source; tm ++) {
    for (int xc = 0; xc < nx; xc += block_size) {
      for (int yc = 0; yc < nx; yc += block_size) {
        for (int zc = 0; zc < nx; zc += block_size) {
          run_sigma_3pt(wall_prop, point_prop, corr_sigma_3pt,
              block_size_sparsen, nt, nx, tm, tp, xc, yc, zc);
        }
      }
    }
    /*
    for (int t = tm + 2; t <= tp - 2; t ++) {
      printf("%d %d %d ", tp-tm, tm, t-tm);
      for (int i = 0; i < 16; i ++) {
        Vcomplex element = corr_sigma_3pt[((tp-tm) * nt + (t-tm)) * nt + i];
        printf("%.10e %.10e ", element.real(), element.imag());
      }
      printf("\n");
    }
    */
  }
  double dtime6 = omp_get_wtime();

  // compute nn->pp 3-point function
  Vcomplex corr_nnpp_3pt[nt * nt * 16];
  for (int i = 0; i < nt * nt * 16; i ++)
    corr_nnpp_3pt[i] = Vcomplex();
  for (int tm = min_source; tm <= max_source; tm ++) {
    for (int xc = 0; xc < nx; xc += block_size) {
      for (int yc = 0; yc < nx; yc += block_size) {
        for (int zc = 0; zc < nx; zc += block_size) {
          for (int gamma_index = 0; gamma_index < 16; gamma_index ++) {
            run_nnpp_3pt(wall_prop, point_prop, corr_nnpp_3pt, gamma_index,
                block_size_sparsen, nt, nx, tm, tp, xc, yc, zc);
          }
        }
      }
    }
    for (int t = tm + 2; t <= tp - 2; t ++) {
      printf("%d %d %d ", tp-tm, tm, t-tm);
      for (int i = 0; i < 16; i ++) {
        Vcomplex element = corr_nnpp_3pt[((tp-tm) * nt + (t-tm)) * nt + i];
        printf("%.10e %.10e ", element.real(), element.imag());
      }
      printf("\n");
    }
  }
  double dtime7 = omp_get_wtime();

  printf("----------------------------------------------------------------------\n");
  printf("0. Initialization:                                   %17.10e s\n", dtime1 - dtime0);
  printf("1. Reading propagators:                              %17.10e s\n", dtime2 - dtime1);
  printf("2. Pion 2-point correlator:                          %17.10e s\n", dtime3 - dtime2);
  printf("3. Neutron 2-point correlator:                       %17.10e s\n", dtime4 - dtime3);
  printf("4. Dineutron 2-point correlator:                     %17.10e s\n", dtime5 - dtime4);
  printf("5. Sigma 3-point correlator:                         %17.10e s\n", dtime6 - dtime5);
  printf("6. nn->pp 3-point correlator:                        %17.10e s\n", dtime7 - dtime6);
  printf("----------------------------------------------------------------------\n");
}
