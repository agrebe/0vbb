// external libraries
#include <omp.h>
#include <fftw3.h>
#include <string.h>

// miscellaneous helper functions
#include "read_prop.h"
#include "spin_mat.h"
#include "color_tensor.h"
#include "gamma_container.h"
#include "run_meson_2pt.h"

// 2-point, 3-point, and 4-point contractions
#include "run_baryon_2pt.h"
#include "run_dibaryon_2pt.h"
#include "run_3pt.h"
#include "run_nnpp_3pt.h"
#include "run_sigma_3pt.h"
#include "run_4pt.h"
#include "run_sigma_4pt.h"
#include "run_nnpp_4pt.h"

// QPhiX linkages
#include "setup.h"
#include "sources.h"
#include "invert.h"

int main(int argc, char ** argv) {
  omp_set_nested(1);
  // lattice size variables
  int nt = 48;
  int nx = 32;
  int vol = nt * nx * nx * nx;
  
  // electron mass
  double me = 3.761159784263958e-04;

  // sparsening factors
  int block_size = 8;         // sparsening at sink
  int block_size_sparsen = 1; // sparsening at operator
  int global_sparsening = 1;  // ratio between nx and actual size of lattice
                              // this is the amount by which props have already been sparsened
  
  // source and sink time ranges
  int min_sep = 6;
  int max_sep = 24;
  int sink_offset = 0; // minimum sink time
  int sink_sep = 8;    // separation between sinks
  
  // compute propagator sizes and quantities
  int num_sources = nt;
  size_t prop_size = vol * 9;
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
  SpinMat * wall_sink_prop_storage = (SpinMat*) malloc(wall_sink_prop_size * sizeof(SpinMat) * num_sources);
  SpinMat * wall_prop [nt];
  SpinMat * wall_sink_prop [nt];
  
  for (int tm = 0; tm < nt; tm ++) {
    wall_prop[tm] = wall_prop_storage + prop_size * tm;
    wall_sink_prop[tm] = wall_sink_prop_storage + wall_sink_prop_size * tm;
  }

  // initialization of QPhiX solver
  char filename [] = "../lattices/cl3_32_48_b6p1_m0p2450-sgf.lime";
  double mass=-0.245;
  //double mass=1.0;
  double clov_coeff=1.24930970916466;
  int soalen = 8; // QPhiX parameter
  int num_vecs = vol / (2 * soalen);
  double ** source = (double **) malloc(sizeof(double*) * 2);
  double ** ferm = (double **) malloc(sizeof(double*) * 2);
  for (int j = 0; j < 2; j ++) {
    source[j] = (double*) malloc(sizeof(double) * 12 * 2 * vol);
    ferm[j] = (double*) malloc(sizeof(double) * 12 * 2 * vol);
  }
  // initialize QDP before using up all our memory
  setup_QDP(&argc, &argv);
  void * params = create_solver(mass, clov_coeff, (char*) filename);

  for (int tm = 0; tm < nt; tm ++) {
    for (int c = 0; c < 3; c ++) {
      for (int s = 0; s < 2; s ++) {
        // create wall source
        wall_source((double (**)[3][4][2][soalen]) source, tm, s, c, nx, nt);
        // project to positive parity
        project_positive((double (**)[3][4][2][soalen]) source, nx, nt);
        // invert off wall source
        invert((double (**)[3][4][2][soalen]) ferm,
               (double (**)[3][4][2][soalen]) source,
               params);
        to_spin_mat((double*) wall_prop[tm], 
                    (double (**)[3][4][2][soalen]) ferm, 
                    s, c, nx, nt);
      }
    }
    // positive parity project the wall prop
    project_prop(wall_prop[tm], nt, nx, 1);
    // rescale prop by 2 (since half the spins were zeroed out)
    rescale_prop(wall_prop[tm], nt, nx, 2);
    printf("Finished wall source at time %d\n", tm);
    fflush(stdout);
  }

  double dtime2 = omp_get_wtime();

  // output files
  FILE* pion_2pt = fopen("../results/pion-2pt-WW", "w");
  FILE* neutron_2pt = fopen("../results/nucleon-2pt-WP", "w");
  FILE* dineutron_2pt = fopen("../results/dinucleon-2pt-WP", "w");
  // compute pion correlator (for testing)
  Vcomplex corr [nt];
  // loop over source times
  for (int tm = 0; tm < nt; tm ++) {
    // compute neutron correlator
    run_neutron_correlator_PP(wall_prop[tm], corr, nt, nx, block_size);
    // flip sign (due to AP boundary conditions) if tp = t + tm > nt
    for (int t = 0; t < nt; t ++) {
      double sign = (t + tm >= nt) ? -1 : 1;
      fprintf(neutron_2pt, "%d %d %.10e %.10e\n", tm, t,
              sign * corr[(t+tm)%nt].real(), sign * corr[(t+tm)%nt].imag());
    }

    // compute dineutron correlator
    run_dineutron_correlator_PP(wall_prop[tm], corr, nt, nx, block_size);
    for (int t = 0; t < nt; t ++) 
      fprintf(dineutron_2pt, "%d %d %.10e %.10e\n", tm, t, corr[(t+tm)%nt].real(), corr[(t+tm)%nt].imag());
  }

  // close 2-point output files
  fclose(pion_2pt);
  fclose(neutron_2pt);
  fclose(dineutron_2pt);

  double dtime3 = omp_get_wtime();
  double time_reading_points = 0, time_3_point = 0, time_4_point = 0;
  
  // plan all the FFTs and compute the neutrino propagator FFT
  // precompute all neutrino propagators and their Fourier transforms
  time_4_point -= omp_get_wtime();
  int nx_blocked = nx / block_size_sparsen;
  int sparse_vol = nx_blocked * nx_blocked * nx_blocked;
  int dims [3] = {nx_blocked, nx_blocked, nx_blocked};
  double * nu = (double *) malloc(2 * sparse_vol * sizeof(double));
  fftw_complex * nu_F = (fftw_complex *) malloc(sparse_vol * sizeof(fftw_complex) * nt);
  for (int dt = 0; dt <= max_sep; dt ++) {
    for (int y3 = 0; y3 < nx; y3 += block_size_sparsen) {
      for (int y2 = 0; y2 < nx; y2 += block_size_sparsen) {
        for (int y1 = 0; y1 < nx; y1 += block_size_sparsen) {
          int idy = ((y3 / block_size_sparsen) * nx_blocked
              + (y2 / block_size_sparsen)) * nx_blocked
              + (y1 / block_size_sparsen);
          double nu_value = nu_prop(y1, y2, y3, dt,
                                    0, 0, 0, 0,
                                    nx, nt, global_sparsening);
          nu[idy * 2] = nu_value;
          nu[idy * 2 + 1] = 0;
        }
      }
    }
    fftw_plan nu_FFT = fftw_plan_many_dft(3, dims, 1,
                         (fftw_complex *) nu,
                         dims, 1, 0,
                         nu_F + sparse_vol * dt, dims, 1, 0,
                         FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(nu_FFT);
    fftw_destroy_plan(nu_FFT);
  }
  free(nu);

  int num_ffts = 4 * 9 * 4; // mu, color^2, spin^2
  fftw_plan Hvec_FFT_forward = fftw_plan_many_dft(3, dims, num_ffts,
             (fftw_complex *) 0x0,
             dims, num_ffts, 1,
             (fftw_complex *) 0x0, dims, num_ffts, 1,
             FFTW_FORWARD, FFTW_ESTIMATE);

  // We want to compute Hx(-x) * (Hy * Snu)
  // Flipping the sign of the variable is equivalent 
  // to changing FFTW_FORWARD -> FFTW_BACKWARD
  // These are all actually forward FFTs (position -> momentum)
  fftw_plan Hvec_FFT_backward = fftw_plan_many_dft(3, dims, num_ffts,
             (fftw_complex *) 0x0,
             dims, num_ffts, 1,
             (fftw_complex *) 0x0, dims, num_ffts, 1,
             FFTW_BACKWARD, FFTW_ESTIMATE);
  time_4_point += omp_get_wtime();

  // 3-point and 4-point output files
  FILE* sigma_3pt = fopen("../results/sigma-3pt", "w");
  FILE* sigma_4pt = fopen("../results/sigma-4pt", "w");
  FILE* nnpp_3pt = fopen("../results/nnpp-3pt", "w");
  FILE* nnpp_4pt = fopen("../results/nnpp-4pt", "w");

  SpinMat * point_prop = (SpinMat*) malloc(prop_size * sizeof(SpinMat));

  for (int tp = sink_offset; tp < nt; tp += sink_sep) {

    // 3-point correlator should store source-sink sep and source-op sep
    // access with corr_sigma_3pt[((sep * nt + t) * 16 + i]
    int num_currents = 5;
    Vcomplex corr_sigma_3pt[nt * nt * num_currents * 2];
    Vcomplex corr_nnpp_3pt[nt * nt * num_currents];

    // 4-point correlator should store source-sink sep and both source-op seps
    // access with corr_sigma_4pt[((tp-tm) * nt + (ty-tm)) * nt + (tx-tm)]
    Vcomplex corr_sigma_4pt[nt * nt * nt];
    Vcomplex corr_nnpp_4pt[nt * nt * nt];

    for (int xc = 0; xc < num_pt_props; xc ++) {
      for (int yc = 0; yc < num_pt_props; yc ++) {
        for (int zc = 0; zc < num_pt_props; zc ++) {
          time_reading_points -= omp_get_wtime();
          /*
          char point_filename [100];
          sprintf(point_filename, "../props/point-prop-%d-%d%d%d.lime.contents/msg02.rec03.scidac-binary-data", tp, xc, yc, zc);
          read_prop(point_filename, point_prop, nt, nx);
          */
          for (int c = 0; c < 3; c ++) {
            for (int s = 0; s < 2; s ++) {
              // create point source
              point_source((double (**)[3][4][2][soalen]) source, 
                  xc * block_size, yc * block_size, zc * block_size, tp,
                  s, c, nx, nt);
              // project to negative parity
              project_negative((double (**)[3][4][2][soalen]) source, nx, nt);
              // invert off point source
              double start = omp_get_wtime();
              invert((double (**)[3][4][2][soalen]) ferm,
                     (double (**)[3][4][2][soalen]) source,
                     params);
              double end = omp_get_wtime();
              printf("Total solve time: %f sec\n", end - start);
              to_spin_mat((double*) point_prop,
                          (double (**)[3][4][2][soalen]) ferm, 
                          s, c, nx, nt);
              double end2 = omp_get_wtime();
              printf("Conversion to spin matrix: %f sec\n", end2 - end);
            }
          }

          double start = omp_get_wtime();
          project_prop(point_prop, nt, nx, 0);
          reverse_prop(point_prop, nt, nx);
          // rescale prop by 2 (since half the spins were zeroed out)
          rescale_prop(point_prop, nt, nx, 2);
          double end = omp_get_wtime();
          printf("Total projection time: %f sec\n", end - start);
          fflush(stdout);

          time_reading_points += omp_get_wtime();


          // compute sigma 3-point function
          time_3_point -= omp_get_wtime();
          #pragma omp parallel for collapse(2)
          for (int sep = min_sep; sep <= max_sep; sep ++) {
            for (int t = 3; t <= max_sep - 3; t ++) {
              if (t > sep - 3) continue;
              // sep = (sink time) - (operator time)
              // if sink wraps around lattice, add nt
              int tm = (nt + tp - sep) % nt;
              int ty = (tm + t) % nt;
              // loop over operator insertion time
              // t = (operator time) - (source time)
              Vcomplex * T = (Vcomplex *) malloc(1296 * num_currents * sizeof(Vcomplex));
              compute_tensor_3(T, wall_prop[tm], point_prop,
                               nx, block_size_sparsen, ty);
              run_sigma_3pt(T, wall_prop[tm], corr_sigma_3pt, 
                            nt, nx, sep, tm, t,
                            xc * block_size, yc * block_size, zc * block_size);
              run_nnpp_3pt(T, wall_prop[tm], corr_nnpp_3pt, 
                           nt, nx, sep, tm, t, num_currents,
                           xc * block_size, yc * block_size, zc * block_size);
              free(T);
            }
          }
          time_3_point += omp_get_wtime();
          printf("Cumulative 3-point time: %f sec\n", time_3_point);
          fflush(stdout);
          

          // compute sigma and nn->pp 4-point functions
          time_4_point -= omp_get_wtime();
          #pragma omp parallel for num_threads(8)
          for (int sep = min_sep; sep <= max_sep; sep ++) {
            int tm = (nt + tp - sep) % nt;
            // precompute all Hvec
            size_t offset = sparse_vol * 4 * 9;
            WeylMat * Hvec = (WeylMat*) malloc(offset * sizeof(WeylMat) * max_sep);
            WeylMat * Hvec_F_forward = (WeylMat*) malloc(offset * sizeof(WeylMat) * max_sep);
            WeylMat * Hvec_F_backward = (WeylMat*) malloc(offset * sizeof(WeylMat) * max_sep);
            #pragma omp parallel for num_threads(8)
            for (int ty = 3; ty <= sep - 3; ty ++) {
              assemble_Hvec(Hvec + ty * offset,
                            wall_prop[tm], point_prop, nx, 
                            block_size_sparsen, tm, tp, (tm + ty) % nt);

              fftw_execute_dft(Hvec_FFT_forward, (fftw_complex *) (Hvec + offset * ty), 
                               (fftw_complex *) (Hvec_F_forward + offset * ty));
              fftw_execute_dft(Hvec_FFT_backward, (fftw_complex *) (Hvec + offset * ty), 
                               (fftw_complex *) (Hvec_F_backward + offset * ty));

              // take the correct linear combinations of Hvec_y (forward) and Hvec_x (backward)
              // to produce the correct electron spins
              for (int idy = 0; idy < sparse_vol; idy ++) {
                for (int c = 0; c < 9; c ++) {
                  WeylMat * Hvec_y = Hvec_F_forward + offset * ty;
                  Hvec_y[(4*idy+0)*9+c] += Hvec_y[(4*idy+1)*9+c] * Vcomplex(0,-1);
                  Hvec_y[(4*idy+2)*9+c] += Hvec_y[(4*idy+3)*9+c] * Vcomplex(0, 1);
                  WeylMat * Hvec_x = Hvec_F_backward + offset * ty;
                  Hvec_x[(4*idy+0)*9+c] += Hvec_x[(4*idy+1)*9+c] * Vcomplex(0, 1);
                  Hvec_x[(4*idy+2)*9+c] += Hvec_x[(4*idy+3)*9+c] * Vcomplex(0,-1);
                }
              }
            }
            #pragma omp parallel for collapse(2) num_threads(8)
            for (int ty = 3; ty <= sep - 3; ty ++) {
              // compute sequential propagator through one operator
              for (int tx = 3; tx <= sep - 3; tx ++) {
                WeylMat * Hvec_y = Hvec_F_forward + ty * sparse_vol * 4 * 9;
                WeylMat * Hvec_x = Hvec_F_backward + tx * sparse_vol * 4 * 9;

                // also compute rank-4 tensor containing current convolution
                Vcomplex * T = (Vcomplex*) malloc(1296 * sizeof(Vcomplex));
                compute_tensor(T, Hvec_x, Hvec_y, 
                               nu_F + sparse_vol * abs(ty - tx),
                               nx, nt, block_size_sparsen, global_sparsening);
                Vcomplex corr_sigma_4pt_value
                            = run_sigma_4pt(wall_prop[tm], T,
                              tp, nx,
                              xc * block_size, yc * block_size, zc * block_size);
                Vcomplex corr_nnpp_4pt_value
                            = run_nnpp_4pt(wall_prop[tm], T,
                              tp, nx,
                              xc * block_size, yc * block_size, zc * block_size);
                // rescale based on electron mass
                corr_sigma_4pt_value *= exp(me * abs(ty - tx));
                corr_sigma_4pt[(sep * nt + ty) * nt + tx] += corr_sigma_4pt_value;
                corr_nnpp_4pt_value *= exp(me * abs(ty - tx));
                corr_nnpp_4pt[(sep * nt + ty) * nt + tx] += corr_nnpp_4pt_value;
                free(T);
              }
            }
            free(Hvec);
            free(Hvec_F_forward);
            free(Hvec_F_backward);
          }
          time_4_point += omp_get_wtime();
          printf("Cumulative 4-point time: %f sec\n", time_4_point);
          fflush(stdout);
        }
      }
    }


    // execute the file I/O in serial after all threads finish
    for (int sep = min_sep; sep <= max_sep; sep ++) {
      // t = (operator time) - (source time)
      for (int t = 3; t <= sep - 3; t ++) {
        int tm = (nt + tp - sep) % nt;
        fprintf(sigma_3pt, "%d %d %d ", sep, tm, t);
        // loop for sigma
        for (int i = 0; i < num_currents * 2; i ++) {
          Vcomplex element = corr_sigma_3pt[(sep * nt + t) * num_currents * 2 + i];
          // flip sign if needed for AP boundary conditions
          if (tm + sep >= nt) element *= -1;
          fprintf(sigma_3pt, "%.10e %.10e ", element.real(), element.imag());
        }
        fprintf(sigma_3pt, "\n");

        fprintf(nnpp_3pt, "%d %d %d ", sep, tm, t);
        // loop for nn->pp
        for (int i = 0; i < num_currents; i ++) {
          Vcomplex element = corr_nnpp_3pt[(sep * nt + t) * num_currents + i];
          fprintf(nnpp_3pt, "%.10e %.10e ", element.real(), element.imag());
        }
        fprintf(nnpp_3pt, "\n");
      }
    }
    for (int sep = min_sep; sep <= max_sep; sep ++) {
      int tm = (nt + tp - sep) % nt;
      for (int ty = 3; ty <= sep - 3; ty ++) {
        for (int tx = 3; tx <= sep - 3; tx ++) {
          Vcomplex corr_sigma_4pt_value = corr_sigma_4pt[(sep * nt + ty) * nt + tx];
          Vcomplex corr_nnpp_4pt_value = corr_nnpp_4pt[(sep * nt + ty) * nt + tx];
          // flip sign if needed for AP boundary conditions
          if (tm + sep >= nt) corr_sigma_4pt_value *= -1;
          fprintf(sigma_4pt, "%d %d %d %.10e %.10e\n", tx, ty, sep, corr_sigma_4pt_value.real(), corr_sigma_4pt_value.imag());
          fprintf(nnpp_4pt, "%d %d %d %.10e %.10e\n", tx, ty, sep, corr_nnpp_4pt_value.real(), corr_nnpp_4pt_value.imag());
        }
      }
    }
  }

  free(nu_F);
  fftw_destroy_plan(Hvec_FFT_forward);
  fftw_destroy_plan(Hvec_FFT_backward);

  double dtime4 = omp_get_wtime();
  
  // close 3-point and 4-point files
  fclose(sigma_3pt);
  fclose(sigma_4pt);
  fclose(nnpp_3pt);
  fclose(nnpp_4pt);
  free(wall_prop_storage);
  free(wall_sink_prop_storage);
  free(point_prop);

  printf("----------------------------------------------------------------------\n");
  printf("0. Initialization:                                   %17.10e s\n", dtime1 - dtime0);
  printf("1. Reading propagators:                              %17.10e s\n", dtime2 - dtime1);
  printf("2. 2-point correlators:                              %17.10e s\n", dtime3 - dtime2);
  printf("3. 3-point and 4-point correlators:                  %17.10e s\n", dtime4 - dtime3);
  printf("\n");
  printf("  3A. Reading point props:                           %17.10e s\n", time_reading_points);
  printf("  3B. 3-point correlators:                           %17.10e s\n", time_3_point);
  printf("  3C. 4-point correlators:                           %17.10e s\n", time_4_point);
  printf("----------------------------------------------------------------------\n");
}
