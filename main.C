#include <omp.h>
#include "read_prop.h"
#include "spin_mat.h"
#include "run_meson_2pt.h"
#include "run_baryon_2pt.h"
#include "run_dibaryon_2pt.h"
#include "run_sigma_3pt.h"
#include "run_nnpp_3pt.h"
#include "run_4pt.h"
#include "run_sigma_4pt.h"
#include "run_nnpp_4pt.h"
#include "color_tensor.h"
#include "gamma_container.h"

int main() {
  // lattice size variables
  int nt = 48;
  int nx = 32;
  int vol = nt * nx * nx * nx;
  
  // electron mass
  double me = 3.761159784263958e-04;

  // sparsening factors
  int block_size = 32;        // sparsening at sink
  int block_size_sparsen = 4; // sparsening at operator
  int global_sparsening = 1;  // ratio between nx and actual size of lattice
                              // this is the amount by which props have already been sparsened
  
  // source and sink time ranges
  int min_source = 0;
  int max_source = 0;
  int tp = 12; // sink time

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

  // output files
  FILE* pion_2pt = fopen("../results/pion-2pt-WW", "w");
  FILE* neutron_2pt = fopen("../results/nucleon-2pt-WP", "w");
  FILE* dineutron_2pt = fopen("../results/dinucleon-2pt-WP", "w");
  FILE* sigma_3pt = fopen("../results/sigma-3pt", "w");
  FILE* nnpp_3pt = fopen("../results/nnpp-3pt", "w");
  FILE* sigma_4pt = fopen("../results/sigma-4pt", "w");
  FILE* nnpp_4pt = fopen("../results/nnpp-4pt", "w");
  
  // compute pion correlator (for testing)
  Vcomplex corr [nt];
  run_pion_correlator_wsink(wall_prop, corr, nt, nx);
  for (int t = 0; t < nt; t ++) fprintf(pion_2pt, "corr[%d] = %f\n", t, corr[t].real());
  double dtime3 = omp_get_wtime();

  // compute neutron correlator
  run_neutron_correlator(wall_prop, corr, nt, nx, block_size);
  for (int t = 0; t < nt; t ++) fprintf(neutron_2pt, "corr[%d] = %f + %fi\n", t, corr[t].real(), corr[t].imag());
  double dtime4 = omp_get_wtime();

  // compute dineutron correlator
  run_dineutron_correlator_PP(wall_prop, corr, nt, nx, block_size);
  for (int t = 0; t < nt; t ++) fprintf(dineutron_2pt, "corr[%d] = %f + %fi\n", t, corr[t].real(), corr[t].imag());
  double dtime5 = omp_get_wtime();

  // compute sigma 3-point function
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
    for (int t = tm + 2; t <= tp - 2; t ++) {
      fprintf(sigma_3pt, "%d %d %d ", tp-tm, tm, t-tm);
      for (int i = 0; i < 16; i ++) {
        Vcomplex element = corr_sigma_3pt[((tp-tm) * nt + (t-tm)) * nt + i];
        fprintf(sigma_3pt, "%.10e %.10e ", element.real(), element.imag());
      }
      fprintf(sigma_3pt, "\n");
    }
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
      fprintf(nnpp_3pt, "%d %d %d ", tp-tm, tm, t-tm);
      for (int i = 0; i < 16; i ++) {
        Vcomplex element = corr_nnpp_3pt[((tp-tm) * nt + (t-tm)) * nt + i];
        fprintf(nnpp_3pt, "%.10e %.10e ", element.real(), element.imag());
      }
      fprintf(nnpp_3pt, "\n");
    }
  }
  double dtime7 = omp_get_wtime();

  // compute sigma and nn->pp 4-point functions
  // correlator should store source-sink sep and both source-op seps
  // access with corr_sigma_3pt[((tp-tm) * nt + (ty-tm)) * nt + (tx-tm)]
  Vcomplex corr_sigma_4pt[nt * nt * nt];
  Vcomplex corr_nnpp_4pt[nt * nt * nt];
  for (int tm = min_source; tm <= max_source; tm ++) {
    for (int ty = tm + 2; ty <= tp - 2; ty ++) {
      // compute sequential propagator through one operator
      int sparse_vol = (nx / block_size_sparsen);
      sparse_vol *= sparse_vol * sparse_vol;
      // loop over source positions
      for (int zc = 0; zc < nx; zc += block_size) {
        for (int yc = 0; yc < nx; yc += block_size) {
          for (int xc = 0; xc < nx; xc += block_size) {
            SpinMat * Hvec = (SpinMat*) malloc(sparse_vol * 4 * 9 * sizeof(SpinMat));
            assemble_Hvec(Hvec, wall_prop, point_prop, nx, 
                          block_size_sparsen, tm, tp, ty);
            for (int tx = tm + 2; tx <= tp - 2; tx ++) {
              // convolve seqprop with neutrino propagator
              SpinMat * SnuHz = (SpinMat*) malloc(sparse_vol * 4 * 9 * sizeof(SpinMat));
              compute_SnuHz(SnuHz, Hvec, tx, ty, nx, block_size_sparsen, global_sparsening);
              Vcomplex corr_sigma_4pt_value
                          = run_sigma_4pt(wall_prop, point_prop, SnuHz,
                            tx, tp, nx, block_size_sparsen, xc, yc, zc);
              Vcomplex corr_nnpp_4pt_value
                          = run_nnpp_4pt(wall_prop, point_prop, SnuHz,
                            tx, tp, nx, block_size_sparsen, xc, yc, zc);
              // rescale based on electron mass
              corr_sigma_4pt_value *= exp(me * abs(ty - tx));
              corr_sigma_4pt[((tp-tm) * nt + (ty-tm)) * nt + (tx-tm)] += corr_sigma_4pt_value;
              fprintf(sigma_4pt, "%d %d %d %e %e\n", tx-tm, ty-tm, tp-tm, corr_sigma_4pt_value.real(), corr_sigma_4pt_value.imag());
              
              corr_nnpp_4pt_value *= exp(me * abs(ty - tx));
              corr_nnpp_4pt[((tp-tm) * nt + (ty-tm)) * nt + (tx-tm)] += corr_nnpp_4pt_value;
              fprintf(nnpp_4pt, "%d %d %d %e %e\n", tx-tm, ty-tm, tp-tm, corr_nnpp_4pt_value.real(), corr_nnpp_4pt_value.imag());
              free(SnuHz);
            }
            free(Hvec);
          }
        }
      }
    }
  }

  double dtime8 = omp_get_wtime();
  
  // close files
  fclose(pion_2pt);
  fclose(neutron_2pt);
  fclose(dineutron_2pt);
  fclose(sigma_3pt);
  fclose(nnpp_3pt);
  fclose(sigma_4pt);
  fclose(nnpp_4pt);

  printf("----------------------------------------------------------------------\n");
  printf("0. Initialization:                                   %17.10e s\n", dtime1 - dtime0);
  printf("1. Reading propagators:                              %17.10e s\n", dtime2 - dtime1);
  printf("2. Pion 2-point correlator:                          %17.10e s\n", dtime3 - dtime2);
  printf("3. Neutron 2-point correlator:                       %17.10e s\n", dtime4 - dtime3);
  printf("4. Dineutron 2-point correlator:                     %17.10e s\n", dtime5 - dtime4);
  printf("5. Sigma 3-point correlator:                         %17.10e s\n", dtime6 - dtime5);
  printf("6. nn->pp 3-point correlator:                        %17.10e s\n", dtime7 - dtime6);
  printf("7. Sigma and nn->pp 4-point correlators:             %17.10e s\n", dtime8 - dtime7);
  printf("----------------------------------------------------------------------\n");
}
