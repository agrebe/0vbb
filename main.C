#include <omp.h>
#include <string.h>
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
  int nx = 8;
  int vol = nt * nx * nx * nx;
  
  // electron mass
  double me = 3.761159784263958e-04;

  // sparsening factors
  int block_size = 8;         // sparsening at sink
  int block_size_sparsen = 1; // sparsening at operator
  int global_sparsening = 4;  // ratio between nx and actual size of lattice
                              // this is the amount by which props have already been sparsened
  
  // source and sink time ranges
  int min_source = 0;
  int max_source = 7;
  int tp = 15; // sink time
  
  // compute propagator sizes and quantities
  int num_sources = (max_source - min_source) + 1;
  int prop_size = vol * 9;
  int wall_sink_prop_size = nt * 9;
  int num_pt_props = nx / block_size; // number of point props in each direction

  double dtime0 = omp_get_wtime();
  // initialize gamma matrices and epsilon tensors
  initialize_gammas();
  color_tensor_1body(color_idx_1);
  color_tensor_2bodies(color_idx_2);
  double dtime1 = omp_get_wtime();

  // read in propagators
  SpinMat * wall_prop_storage = (SpinMat*) malloc(prop_size * sizeof(SpinMat) * num_sources);
  SpinMat * wall_sink_prop_storage = (SpinMat*) malloc(prop_size * sizeof(SpinMat) * num_sources);
  SpinMat * wall_prop [nt];
  SpinMat * wall_sink_prop [nt];
  for (int tm = min_source; tm <= max_source; tm ++) {
    wall_prop[tm] = wall_prop_storage + prop_size * (tm - min_source);
    wall_sink_prop[tm] = wall_sink_prop_storage + wall_sink_prop_size * (tm - min_source);
    
    char wall_sink_filename [100], wall_filename [100];
    sprintf(wall_filename, "../props/wall-source-%d.lime.contents/msg02.rec03.scidac-binary-data", tm);
    sprintf(wall_sink_filename, "../props/wall-source-%d-wall-sink.lime.contents/msg02.rec03.scidac-binary-data", tm);
    read_prop(wall_filename, wall_prop[tm], nt, nx);
    read_prop(wall_sink_filename, wall_sink_prop[tm], nt, 1);
  }
  
  SpinMat * point_prop_storage = (SpinMat*) malloc(prop_size * sizeof(SpinMat)
                                                   * num_pt_props * num_pt_props * num_pt_props);
  SpinMat * point_prop [num_pt_props][num_pt_props][num_pt_props];
  for (int xc = 0; xc < num_pt_props; xc ++) {
    for (int yc = 0; yc < num_pt_props; yc ++) {
      for (int zc = 0; zc < num_pt_props; zc ++) {
        point_prop[xc][yc][zc] = point_prop_storage + prop_size * ((xc * num_pt_props + yc) * num_pt_props + zc);
        char point_filename [100];
        sprintf(point_filename, "../props/point-prop-%d%d%d.lime.contents/msg02.rec03.scidac-binary-data", xc, yc, zc);
        read_prop(point_filename, point_prop[xc][yc][zc], nt, nx);
        rescale_prop(point_prop[xc][yc][zc], nt, nx, 0.5);
      }
    }
  }
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
  // loop over source times
  for (int tm = min_source; tm <= max_source; tm ++) {
    run_pion_correlator_wsink(wall_sink_prop[tm], corr, nt, 1);
    for (int t = 0; t < nt; t ++) 
      fprintf(pion_2pt, "%d %d %.10e\n", tm, t, corr[(t+tm)%nt].real());

    // compute neutron correlator
    run_neutron_correlator_PP(wall_prop[tm], corr, nt, nx, block_size);
    for (int t = 0; t < nt; t ++) 
      fprintf(neutron_2pt, "%d %d %.10e %.10e\n", tm, t, corr[(t+tm)%nt].real(), corr[(t+tm)%nt].imag());

    // compute dineutron correlator
    run_dineutron_correlator_PP(wall_prop[tm], corr, nt, nx, block_size);
    for (int t = 0; t < nt; t ++) 
      fprintf(dineutron_2pt, "%d %d %.10e %.10e\n", tm, t, corr[(t+tm)%nt].real(), corr[(t+tm)%nt].imag());
  }
  double dtime3 = omp_get_wtime();
  
  // compute sigma 3-point function
  // correlator should store source-sink sep and source-op sep
  // access with corr_sigma_3pt[(sep * nt + t * nt) + i]
  Vcomplex corr_sigma_3pt[nt * nt * 16];
  for (int i = 0; i < nt * nt * 16; i ++)
    corr_sigma_3pt[i] = Vcomplex();
  for (int tm = min_source; tm <= max_source; tm ++) {
    // sep = (sink time) - (operator time)
    // if sink wraps around lattice, add nt
    int sep = (nt + tp - tm) % nt;
    for (int xc = 0; xc < num_pt_props; xc ++) {
      for (int yc = 0; yc < num_pt_props; yc ++) {
        for (int zc = 0; zc < num_pt_props; zc ++) {
          run_sigma_3pt(wall_prop[tm], point_prop[xc][yc][zc], corr_sigma_3pt,
              block_size_sparsen, nt, nx, tm, sep, 
              xc * block_size, yc * block_size, zc * block_size);
        }
      }
    }
    // t = (operator time) - (source time)
    for (int t = 3; t <= sep - 3; t ++) {
      fprintf(sigma_3pt, "%d %d %d ", sep, tm, t);
      for (int i = 0; i < 16; i ++) {
        Vcomplex element = corr_sigma_3pt[(sep * nt + t) * nt + i];
        fprintf(sigma_3pt, "%.10e %.10e ", element.real(), element.imag());
      }
      fprintf(sigma_3pt, "\n");
    }
  }
  double dtime4 = omp_get_wtime();

  // compute nn->pp 3-point function
  Vcomplex corr_nnpp_3pt[nt * nt * 16];
  for (int i = 0; i < nt * nt * 16; i ++)
    corr_nnpp_3pt[i] = Vcomplex();
  for (int tm = min_source; tm <= max_source; tm ++) {
    int sep = (nt + tp - tm) % nt;
    for (int xc = 0; xc < num_pt_props; xc ++) {
      for (int yc = 0; yc < num_pt_props; yc ++) {
        for (int zc = 0; zc < num_pt_props; zc ++) {
          for (int gamma_index = 0; gamma_index < 16; gamma_index ++) {
            run_nnpp_3pt(wall_prop[tm], point_prop[xc][yc][zc], corr_nnpp_3pt, gamma_index,
                block_size_sparsen, nt, nx, tm, sep,
                xc * block_size, yc * block_size, zc * block_size);
          }
        }
      }
    }
    for (int t = 3; t <= sep - 3; t ++) {
      fprintf(nnpp_3pt, "%d %d %d ", sep, tm, t);
      for (int i = 0; i < 16; i ++) {
        Vcomplex element = corr_nnpp_3pt[(sep * nt + t) * nt + i];
        fprintf(nnpp_3pt, "%.10e %.10e ", element.real(), element.imag());
      }
      fprintf(nnpp_3pt, "\n");
    }
  }
  double dtime5 = omp_get_wtime();

  // compute sigma and nn->pp 4-point functions
  // correlator should store source-sink sep and both source-op seps
  // access with corr_sigma_3pt[((tp-tm) * nt + (ty-tm)) * nt + (tx-tm)]
  Vcomplex corr_sigma_4pt[nt * nt * nt];
  Vcomplex corr_nnpp_4pt[nt * nt * nt];
  for (int tm = min_source; tm <= max_source; tm ++) {
    int sep = (nt + tp - tm) % nt;
    for (int ty = 3; ty <= sep - 3; ty ++) {
      // compute sequential propagator through one operator
      int sparse_vol = (nx / block_size_sparsen);
      sparse_vol *= sparse_vol * sparse_vol;
      // loop over source positions
      for (int xc = 0; xc < num_pt_props; xc ++) {
        for (int yc = 0; yc < num_pt_props; yc ++) {
          for (int zc = 0; zc < num_pt_props; zc ++) {
            WeylMat * Hvec = (WeylMat*) malloc(sparse_vol * 4 * 9 * sizeof(WeylMat));
            assemble_Hvec(Hvec, wall_prop[tm], point_prop[xc][yc][zc], nx, 
                          block_size_sparsen, tm, tp, tm + ty);
            for (int tx = 3; tx <= sep - 3; tx ++) {
              // convolve seqprop with neutrino propagator
              WeylMat * SnuHz = (WeylMat*) malloc(sparse_vol * 4 * 9 * sizeof(WeylMat));
              compute_SnuHz(SnuHz, Hvec, tx + tm, ty + tm, nx, nt, block_size_sparsen, global_sparsening);
              Vcomplex corr_sigma_4pt_value
                          = run_sigma_4pt(wall_prop[tm], point_prop[xc][yc][zc], SnuHz,
                            tx + tm, tp, nx, block_size_sparsen,
                            xc * block_size, yc * block_size, zc * block_size);
              Vcomplex corr_nnpp_4pt_value
                          = run_nnpp_4pt(wall_prop[tm], point_prop[xc][yc][zc], SnuHz,
                            tx + tm, tp, nx, block_size_sparsen,
                            xc * block_size, yc * block_size, zc * block_size);
              // rescale based on electron mass
              corr_sigma_4pt_value *= exp(me * abs(ty - tx));
              corr_sigma_4pt[(sep * nt + ty) * nt + tx] += corr_sigma_4pt_value;
              
              corr_nnpp_4pt_value *= exp(me * abs(ty - tx));
              corr_nnpp_4pt[(sep * nt + ty) * nt + tx] += corr_nnpp_4pt_value;
              free(SnuHz);
            }
            free(Hvec);
          }
        }
      }
      for (int tx = 3; tx <= sep - 3; tx ++) {
        Vcomplex corr_sigma_4pt_value = corr_sigma_4pt[(sep * nt + ty) * nt + tx];
        Vcomplex corr_nnpp_4pt_value = corr_nnpp_4pt[(sep * nt + ty) * nt + tx];
        fprintf(sigma_4pt, "%d %d %d %e %e\n", tx-tm, ty-tm, tp-tm, corr_sigma_4pt_value.real(), corr_sigma_4pt_value.imag());
        fprintf(nnpp_4pt, "%d %d %d %e %e\n", tx-tm, ty-tm, tp-tm, corr_nnpp_4pt_value.real(), corr_nnpp_4pt_value.imag());
      }
    }
  }

  double dtime6 = omp_get_wtime();
  
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
  printf("2. 2-point correlators:                              %17.10e s\n", dtime3 - dtime2);
  printf("3. Sigma 3-point correlator:                         %17.10e s\n", dtime4 - dtime3);
  printf("4. nn->pp 3-point correlator:                        %17.10e s\n", dtime5 - dtime4);
  printf("5. Sigma and nn->pp 4-point correlators:             %17.10e s\n", dtime6 - dtime5);
  printf("----------------------------------------------------------------------\n");
}
