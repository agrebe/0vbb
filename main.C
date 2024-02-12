// external libraries
#include <omp.h>
#include <fftw3.h>
#include <string.h>

// miscellaneous helper functions
#include "read_prop.h"
#include "spin_mat.h"
#include "color_tensor.h"
#include "gamma_container.h"

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
#include "qphix-wrapper/setup.h"
#include "qphix-wrapper/sources.h"
#include "qphix-wrapper/invert.h"

/*
 * This code is the the main runner of the contractions.
 * It calls the QPhiX inverters (or reads props from disk),
 * calls the functions to compute the 4-quark tensors for both short-distance
 * and long-distance interactions, and then passes these tensors to the
 * 3-point and 4-point functions, respectively.
 *
 * The contractions are performed between a wall source and a sparsened
 * grid of point sinks.  The wall sources are computed at every timeslice
 * on the gauge configuration, whereas the point sinks are more
 * expensive, since the different spatial positions on the sparse grid
 * must be inverted separately, so they are only measured on a subset
 * of temporal points.  Different source-sink separations are computed
 * by moving the source instead of the sink.
 * To reduce both contraction and inversion times, positive parity
 * projection operators are inserted at both source and sink.
 *
 * On the lattice in question, this codebase is designed to fit
 * in 128 GB memory, which requires some packing of the wall props
 * and recycling the memory for point propagators after each sink
 * point has been measured.
 */

int main(int argc, char ** argv) {
  // lattice size variables
  int nt = 48;
  int nx = 32;
  int vol = nt * nx * nx * nx;
  int spatial_vol = nx * nx * nx;
  
  // electron mass
  double me = 3.761159784263958e-04;

  // sparsening factors
  int block_size = 8;         // sparsening at sink
  int block_size_sparsen = 1; // sparsening at operator
  int global_sparsening = 1;  // ratio between nx and actual size of lattice
                              // if propagators have already been sparsened,
                              // e.g. if being read from disk, this is the
                              // amount by which they are already sparsened
  
  // source and sink time ranges
  int min_sep = 6;
  int max_sep = 23;
  int sink_offset = 0; // minimum sink time
  int sink_sep = 8;    // separation between sinks
  
  // compute propagator sizes and quantities
  int num_sources = nt;
  size_t prop_size = vol * 9;
  int num_pt_props = nx / block_size; // number of point props in each direction

  double dtime0 = omp_get_wtime();
  // initialize gamma matrices and epsilon tensors
  initialize_gammas();
  color_tensor_1body(color_idx_1);
  color_tensor_2bodies(color_idx_2);

  double dtime1 = omp_get_wtime();
  
  // Create storage space for wall propagators (from the source time)
  // We really only need each prop on half the lattice points,
  // since time separations of more than half the lattice are noisy
  // and have thermal pollution from backwards-going states
  // As a result, we can store two props in something of size prop_size
  // if we have one from tm to (tm + nt/2 - 1) and the other starting
  // at (tm + nt/2 - 1)
  SpinMat * wall_prop_storage = (SpinMat*) malloc(prop_size 
                                    * sizeof(SpinMat) * nt / 2);
  // array of pointers containing locations in storage buffer
  // of wall props starting at source tm
  SpinMat * wall_prop [nt];
  
  for (int tm = 0; tm < nt / 2; tm ++) {
    wall_prop[tm] = wall_prop_storage + prop_size * tm;
    wall_prop[tm + nt / 2] = wall_prop_storage + prop_size * tm;
  }

  // initialization of QPhiX solver

  // filename of lattice to read in
  char filename [] = "../lattices/cl3_32_48_b6p1_m0p2450-sgf.lime";
  double mass=-0.245;
  double clov_coeff=1.24930970916466;
  int soalen = 8; // QPhiX parameter
  int num_vecs = vol / (2 * soalen);

  // source is the vector to invert off of
  // ferm is the result of inversion, that is, (Dslash + m)^-1 source
  // These are stored as arrays with two components because of even/odd
  // preconditioning (each element in outer array is one checkerboard)
  double ** source = (double **) malloc(sizeof(double*) * 2);
  double ** ferm = (double **) malloc(sizeof(double*) * 2);
  for (int j = 0; j < 2; j ++) {
    source[j] = (double*) malloc(sizeof(double) * 12 * 2 * vol);
    ferm[j] = (double*) malloc(sizeof(double) * 12 * 2 * vol);
  }
  // initialization of QDP
  // QDP is used to compute clover term and preconditioning matrices
  setup_QDP(&argc, &argv);

  // Parameters of QPhiX solver
  // This contains pointers to the solver and matrices used by QPhiX/QDP++
  // The qphix-wrapper code hides all the QPhiX data types
  // so this is stored as a void* to avoid exposing these data types
  void * params = create_solver(mass, clov_coeff, (char*) filename);

  // additional buffer to temporarily hold wall propagators
  // before copying them into the full storage
  SpinMat * wall_prop_buffer = (SpinMat *) malloc(prop_size * sizeof(SpinMat));
  for (int tm = 0; tm < nt; tm ++) {
    // zero out wall prop buffer
    for (int i = 0; i < prop_size; i ++)
      wall_prop_buffer[i] = SpinMat();
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
        to_spin_mat((double*) wall_prop_buffer, 
                    (double (**)[3][4][2][soalen]) ferm, 
                    s, c, nx, nt);
      }
    }
    // copy buffer into wall_prop for times in [tm, tm + nt/2) mod nt
    for (int t = 0; t < nt; t ++)
      if (((t - tm + nt) % nt) < nt / 2)
        memcpy(wall_prop[tm] + spatial_vol * 9 * t,
               wall_prop_buffer + spatial_vol * 9 * t,
               spatial_vol * 9 * sizeof(SpinMat));
    printf("Finished wall source at time %d\n", tm);
    fflush(stdout);
  }
  free(wall_prop_buffer);
  for (int tm = 0; tm < nt / 2; tm ++) {
    // positive parity project the wall prop
    project_prop(wall_prop[tm], nt, nx, 1);
    // rescale prop by 2 (since half the spins were zeroed out)
    rescale_prop(wall_prop[tm], nt, nx, 2);
  }

  double dtime2 = omp_get_wtime();

  // output files
  FILE* neutron_2pt = fopen("../results/nucleon-2pt-WP", "w");
  FILE* dineutron_2pt = fopen("../results/dinucleon-2pt-WP", "w");
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
  fclose(neutron_2pt);
  fclose(dineutron_2pt);

  double dtime3 = omp_get_wtime();
  double time_reading_points = 0, time_3_point = 0, time_4_point = 0;
  double time_4_point_A = 0, time_4_point_B = 0;
  
  // Precompute the neutrino propagator and its 3-D Fourier transform
  // This is independent of the quark propagators, so it can be computed
  // outside of all the quark propagator loops
  time_4_point -= omp_get_wtime();
  int nx_blocked = nx / block_size_sparsen;
  int sparse_vol = nx_blocked * nx_blocked * nx_blocked;
  int dims [3] = {nx_blocked, nx_blocked, nx_blocked};
  double * nu = (double *) malloc(2 * sparse_vol * sizeof(double));
  fftw_complex * nu_F = (fftw_complex *) malloc(sparse_vol * sizeof(fftw_complex) * nt);
  for (int dt = 0; dt <= max_sep; dt ++) {
    #pragma omp parallel for collapse(3)
    for (int y3 = 0; y3 < nx; y3 += block_size_sparsen) {
      for (int y2 = 0; y2 < nx; y2 += block_size_sparsen) {
        for (int y1 = 0; y1 < nx; y1 += block_size_sparsen) {
          int idy = ((y3 / block_size_sparsen) * nx_blocked
              + (y2 / block_size_sparsen)) * nx_blocked
              + (y1 / block_size_sparsen);
          double nu_value = nu_prop(y1, y2, y3, dt,
                                    nx, global_sparsening);
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

  // Plan the Fourier transforms of the quark propagators
  // These plans will be later used once the propagators are computed
  // but they can be planned on dummy memory locations

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

  // We will compute a grid of point sinks at different times tp
  // This is the outer loop over the different sink times
  for (int tp = sink_offset; tp < nt; tp += sink_sep) {

    // 3-point correlator should store source-sink sep and source-op sep
    // access with corr_sigma_3pt[((sep * nt + t) * 16 + i]
    int num_currents = 9;
    Vcomplex corr_sigma_3pt[nt * nt * num_currents * 4];
    Vcomplex corr_nnpp_3pt[nt * nt * num_currents];

    // 4-point correlator should store source-sink sep and both source-op seps
    // access with corr_sigma_4pt[((tp-tm) * nt + (ty-tm)) * nt + (tx-tm)]
    Vcomplex corr_sigma_4pt[nt * nt * nt];
    Vcomplex corr_nnpp_4pt[nt * nt * nt];

    // loop over sink spatial positions
    // sink is a sparse grid of points
    for (int xc = 0; xc < num_pt_props; xc ++) {
      for (int yc = 0; yc < num_pt_props; yc ++) {
        for (int zc = 0; zc < num_pt_props; zc ++) {
          time_reading_points -= omp_get_wtime();
          /*
           * Note: If one wanted to read precomputed propagators from disk,
           * one could do so with code such as
             char point_filename [100];
             sprintf(point_filename, 
               "../props/point-prop-%d-%d%d%d.lime.contents/msg02.rec03.scidac-binary-data",
               tp, xc, yc, zc);
             read_prop(point_filename, point_prop, nt, nx);
           * Reading wall props from disk would be a similar process
           * This is feasible if the propagators are sparsened
           * (the original role of the global_sparsening variable)
           * but is expensive if one has to read/write full props
          */

          // zero out point prop
          for (int i = 0; i < prop_size; i ++)
            point_prop[i] = SpinMat();

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
          // loop over source-sink separation
          for (int sep = min_sep; sep <= max_sep; sep ++) {
            // loop over source-operator separation
            // t = (operator time) - (source time)
            for (int t = 3; t <= max_sep - 3; t ++) {
              // source-operator separation t must be at least 3 units
              // away from both source and sink
              // OpenMP cannot parallelize ragged loops,
              // so we loop up to max_sep and then skip unneeded values of t
              if (t > sep - 3) continue;

              // sep = (sink time) - (operator time)
              // if sink wraps around lattice, add nt
              int tm = (nt + tp - sep) % nt;
              int ty = (tm + t) % nt;

              // compute the 4-quark spin-color tensor for all currents
              Vcomplex * T = (Vcomplex *) malloc(1296 * num_currents 
                                                  * sizeof(Vcomplex));
              compute_tensor_3(T, wall_prop[tm], point_prop,
                               nx, block_size_sparsen, ty);
              // compute sigma and nn -> pp contractions using this tensor
              run_sigma_3pt(T, wall_prop[tm], corr_sigma_3pt, 
                            nt, nx, sep, tm, t, num_currents,
                            xc * block_size, yc * block_size, zc * block_size);
              run_nnpp_3pt(T, wall_prop[tm], corr_nnpp_3pt, 
                           nt, nx, sep, tm, t, num_currents,
                           xc * block_size, yc * block_size, zc * block_size);
              // free the tensor
              free(T);
            }
          }
          time_3_point += omp_get_wtime();
          printf("Cumulative 3-point time: %f sec\n", time_3_point);
          fflush(stdout);
          

          // compute sigma and nn->pp 4-point functions
          time_4_point -= omp_get_wtime();
          // loop over source-sink separations
          // Note: Parallelizing this loop would dramatically increase
          // the memory footprint of the code, so it is left as serial
          for (int sep = min_sep; sep <= max_sep; sep ++) {
            // compute source time
            int tm = (nt + tp - sep) % nt;
            // Compute the sequential propagator Hvec and its Fourier transform
            // This is done for all source-operator separations
            // before any actual contractions are computed
            size_t offset = sparse_vol * 4 * 9;
            WeylMat * Hvec = (WeylMat*) malloc(offset * sizeof(WeylMat) 
                                                * max_sep);
            WeylMat * Hvec_F_forward = (WeylMat*) malloc(offset 
                                                * sizeof(WeylMat) * max_sep);
            WeylMat * Hvec_F_backward = (WeylMat*) malloc(offset 
                                                * sizeof(WeylMat) * max_sep);
            time_4_point_A -= omp_get_wtime();

            // loop over operator-source separation time
            #pragma omp parallel for
            for (int ty = 3; ty <= sep - 3; ty ++) {
              assemble_Hvec(Hvec + ty * offset,
                            wall_prop[tm], point_prop, nx, 
                            block_size_sparsen, tm, tp, (tm + ty) % nt);

              fftw_execute_dft(Hvec_FFT_forward, (fftw_complex *) (Hvec + offset * ty), 
                               (fftw_complex *) (Hvec_F_forward + offset * ty));
              fftw_execute_dft(Hvec_FFT_backward, (fftw_complex *) (Hvec + offset * ty), 
                               (fftw_complex *) (Hvec_F_backward + offset * ty));

            }
            time_4_point_A += omp_get_wtime();
            time_4_point_B -= omp_get_wtime();
            // loop over the two operator insertion times
            #pragma omp parallel for collapse(2)
            for (int ty = 3; ty <= sep - 3; ty ++) {
              for (int tx = 3; tx <= sep - 3; tx ++) {
                // grab the values of Hvec_y and Hvec_x corresponding
                // to operator insertion times of ty and tx
                WeylMat * Hvec_y = Hvec_F_forward + ty * sparse_vol * 4 * 9;
                WeylMat * Hvec_x = Hvec_F_backward + tx * sparse_vol * 4 * 9;

                // compute rank-4 tensor of the four quarks convolved
                // with the neutrino propagator
                Vcomplex * T = (Vcomplex*) malloc(1296 * sizeof(Vcomplex));
                compute_tensor(T, Hvec_x, Hvec_y, 
                               nu_F + sparse_vol * abs(ty - tx),
                               nx, nt, block_size_sparsen, global_sparsening);

                // compute the sigma and nn->pp 4-point functions based
                // on this tensor
                Vcomplex corr_sigma_4pt_value
                            = run_sigma_4pt(wall_prop[tm], T,
                              tp, nx,
                              xc * block_size, yc * block_size, zc * block_size);
                Vcomplex corr_nnpp_4pt_value
                            = run_nnpp_4pt(wall_prop[tm], T,
                              tp, nx,
                              xc * block_size, yc * block_size, zc * block_size);

                // Multiply in an exponentially growing term
                // based on the fact that we have emitted an electron mass
                // at the first of the two currents
                // This is technically necessary, although in practice
                // the effect is very small since me is tiny on nuclear scales
                corr_sigma_4pt_value *= exp(me * abs(ty - tx));
                corr_sigma_4pt[(sep * nt + ty) * nt + tx] += corr_sigma_4pt_value;
                corr_nnpp_4pt_value *= exp(me * abs(ty - tx));
                corr_nnpp_4pt[(sep * nt + ty) * nt + tx] += corr_nnpp_4pt_value;
                free(T);
              }
            }
            time_4_point_B += omp_get_wtime();
            free(Hvec);
            free(Hvec_F_forward);
            free(Hvec_F_backward);
          }
          time_4_point += omp_get_wtime();
          printf("Cumulative 4-point time: %f sec\n", time_4_point);
          printf("Cumulative 4-point seqprop construction time: %f sec\n", time_4_point_A);
          printf("Cumulative 4-point contractions time: %f sec\n", time_4_point_B);
          fflush(stdout);
        }
      }
    }


    // execute the file I/O in serial after all threads finish
    // write the 3-point files first
    for (int sep = min_sep; sep <= max_sep; sep ++) {
      // t = (operator time) - (source time)
      for (int t = 3; t <= sep - 3; t ++) {
        int tm = (nt + tp - sep) % nt;
        fprintf(sigma_3pt, "%d %d %d ", sep, tm, t);
        // loop for sigma
        for (int i = 0; i < num_currents * 4; i ++) {
          Vcomplex element = corr_sigma_3pt[(sep * nt + t) * num_currents * 4 + i];
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
    // and then write the 4-point files
    for (int sep = min_sep; sep <= max_sep; sep ++) {
      int tm = (nt + tp - sep) % nt;
      for (int ty = 3; ty <= sep - 3; ty ++) {
        for (int tx = 3; tx <= sep - 3; tx ++) {
          Vcomplex corr_sigma_4pt_value = corr_sigma_4pt[(sep * nt + ty) * nt + tx];
          Vcomplex corr_nnpp_4pt_value = corr_nnpp_4pt[(sep * nt + ty) * nt + tx];
          // flip sign if needed for AP boundary conditions
          if (tm + sep >= nt) corr_sigma_4pt_value *= -1;
          fprintf(sigma_4pt, "%d %d %d %.10e %.10e\n", tx, ty, sep, 
              corr_sigma_4pt_value.real(), corr_sigma_4pt_value.imag());
          fprintf(nnpp_4pt, "%d %d %d %.10e %.10e\n", tx, ty, sep, 
              corr_nnpp_4pt_value.real(), corr_nnpp_4pt_value.imag());
        }
      }
    }

  } // end loop over sink position

  // destroy the FFT plans and free the transformed neutrino prop
  free(nu_F);
  fftw_destroy_plan(Hvec_FFT_forward);
  fftw_destroy_plan(Hvec_FFT_backward);

  double dtime4 = omp_get_wtime();
  
  // close 3-point and 4-point files
  fclose(sigma_3pt);
  fclose(sigma_4pt);
  fclose(nnpp_3pt);
  fclose(nnpp_4pt);

  // free memory
  free(wall_prop_storage);
  free(point_prop);

  // Print out timing information
  // Note: "Reading props" initially meant time reading precomputed props,
  // but now it means the time spent in QPhiX calls to actually do
  // the propagator computation on the fly
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
