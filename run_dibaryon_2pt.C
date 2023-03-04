#include "run_dibaryon_2pt.h"
#include "gamma_container.h"
#include "color_tensor.h"

/*
 * This code computes the 2-point correlation function for a pair
 * of identical nucleons (i.e. an isospin-1 state).  As with the
 * single baryon 2-point function code, it accepts a 4x4 spin matrix
 * as the propagator from source to sink and fills the correlator corr.
 * This code is not optimized (in particular, it performs all the
 * multiplications as 4x4 matrices even though all are positive parity
 * projected), but it is a sub-percent contribution to the total
 * code runtime, in part because it is only computed on a sparse
 * grid of points at the sink.
 * Unlike the single nucleon case, only the positive parity projected
 * version is given here.
 */

// 2-point correlator for a pair of nucleons projected to positive parity
void run_dineutron_correlator_PP(SpinMat * prop, Vcomplex * corr, int nt, int nx, int block_size) {
  for (int t = 0; t < nt; t ++) corr[t] = Vcomplex();
  #pragma omp parallel for
  for (int t = 0; t < nt; t ++) {
    for (int z = 0; z < nx; z += block_size) {
      for (int y = 0; y < nx; y += block_size) {
        for (int x = 0; x < nx; x += block_size) {
          int loc = ((t * nx + z) * nx + y) * nx + x;
          SpinMat * wilsonMat = prop + 9*loc;
          SpinMat SCP [9], STCP[9];
          for (int c = 0; c < 9; c ++) {
            SCP[c] = wilsonMat[c] * cg5 * pp;
            STCP[c] = wilsonMat[c].transpose() * cg5 * pp;
          }
          for(int ii=0; ii<1296; ii++) 
          {
            const int& i      = color_idx_2[13*ii+0];
            const int& j      = color_idx_2[13*ii+1];
            const int& k      = color_idx_2[13*ii+2];
            const int& l      = color_idx_2[13*ii+3];
            const int& m      = color_idx_2[13*ii+4];
            const int& n      = color_idx_2[13*ii+5];
            const int& ip     = color_idx_2[13*ii+6];
            const int& jp     = color_idx_2[13*ii+7];
            const int& kp     = color_idx_2[13*ii+8];
            const int& lp     = color_idx_2[13*ii+9];
            const int& mp     = color_idx_2[13*ii+10];
            const int& np     = color_idx_2[13*ii+11];
            const double sign = color_idx_2[13*ii+12];
            corr[t] += sign * Trace( SCP[3*j+lp] * STCP[3*k+mp] ) * Trace( SCP[3*m+ip] * STCP[3*n+jp] ) * Trace( SCP[3*i+kp] * STCP[3*l+np] );
            corr[t] += sign * Trace( SCP[3*j+ip] * STCP[3*k+jp] ) * Trace( SCP[3*m+lp] * STCP[3*n+mp] ) * Trace( SCP[3*i+kp] * STCP[3*l+np] );
            corr[t] -= sign * Trace( SCP[3*j+ip] * STCP[3*n+jp] * SCP[3*m+lp] * STCP[3*k+mp] ) * Trace( SCP[3*i+kp] * STCP[3*l+np] );
            corr[t] -= sign * Trace( SCP[3*j+lp] * STCP[3*n+mp] * SCP[3*m+ip] * STCP[3*k+jp] ) * Trace( SCP[3*i+kp] * STCP[3*l+np] );
            corr[t] += sign * Trace( SCP[3*j+lp] * STCP[3*k+mp] ) * Trace( SCP[3*m+ip] * STCP[3*n+jp] ) * Trace( SCP[3*i+np] * STCP[3*l+kp] );
            corr[t] += sign * Trace( SCP[3*j+ip] * STCP[3*k+jp] ) * Trace( SCP[3*m+lp] * STCP[3*n+mp] ) * Trace( SCP[3*i+np] * STCP[3*l+kp] );
            corr[t] -= sign * Trace( SCP[3*i+np] * STCP[3*l+kp] ) * Trace( SCP[3*j+ip] * STCP[3*n+jp] * SCP[3*m+lp] * STCP[3*k+mp] );
            corr[t] -= sign * Trace( SCP[3*i+np] * STCP[3*l+kp] ) * Trace( SCP[3*j+lp] * STCP[3*n+mp] * SCP[3*m+ip] * STCP[3*k+jp] );
            corr[t] -= sign * Trace( SCP[3*m+lp] * STCP[3*n+mp] ) * Trace( SCP[3*i+ip] * STCP[3*k+jp] * SCP[3*j+kp] * STCP[3*l+np] );
            corr[t] -= sign * Trace( SCP[3*m+lp] * STCP[3*n+mp] ) * Trace( SCP[3*i+ip] * STCP[3*k+jp] * SCP[3*j+np] * STCP[3*l+kp] );
            corr[t] -= sign * Trace( SCP[3*j+lp] * STCP[3*k+mp] ) * Trace( SCP[3*i+ip] * STCP[3*n+jp] * SCP[3*m+kp] * STCP[3*l+np] );
            corr[t] -= sign * Trace( SCP[3*j+lp] * STCP[3*k+mp] ) * Trace( SCP[3*i+ip] * STCP[3*n+jp] * SCP[3*m+np] * STCP[3*l+kp] );
            corr[t] -= sign * Trace( SCP[3*m+lp] * STCP[3*n+mp] ) * Trace( SCP[3*i+kp] * STCP[3*j+np] * SCP[3*k+jp] * STCP[3*l+ip] );
            corr[t] -= sign * Trace( SCP[3*m+ip] * STCP[3*n+jp] ) * Trace( SCP[3*i+kp] * STCP[3*j+np] * SCP[3*k+mp] * STCP[3*l+lp] );
            corr[t] -= sign * Trace( SCP[3*j+lp] * STCP[3*k+mp] ) * Trace( SCP[3*i+kp] * STCP[3*m+np] * SCP[3*n+jp] * STCP[3*l+ip] );
            corr[t] -= sign * Trace( SCP[3*j+ip] * STCP[3*k+jp] ) * Trace( SCP[3*i+kp] * STCP[3*m+np] * SCP[3*n+mp] * STCP[3*l+lp] );
            corr[t] -= sign * Trace( SCP[3*m+ip] * STCP[3*n+jp] ) * Trace( SCP[3*i+lp] * STCP[3*k+mp] * SCP[3*j+kp] * STCP[3*l+np] );
            corr[t] -= sign * Trace( SCP[3*m+ip] * STCP[3*n+jp] ) * Trace( SCP[3*i+lp] * STCP[3*k+mp] * SCP[3*j+np] * STCP[3*l+kp] );
            corr[t] -= sign * Trace( SCP[3*j+ip] * STCP[3*k+jp] ) * Trace( SCP[3*i+lp] * STCP[3*n+mp] * SCP[3*m+kp] * STCP[3*l+np] );
            corr[t] -= sign * Trace( SCP[3*j+ip] * STCP[3*k+jp] ) * Trace( SCP[3*i+lp] * STCP[3*n+mp] * SCP[3*m+np] * STCP[3*l+kp] );
            corr[t] -= sign * Trace( SCP[3*m+lp] * STCP[3*n+mp] ) * Trace( SCP[3*i+np] * STCP[3*j+kp] * SCP[3*k+jp] * STCP[3*l+ip] );
            corr[t] -= sign * Trace( SCP[3*m+ip] * STCP[3*n+jp] ) * Trace( SCP[3*i+np] * STCP[3*j+kp] * SCP[3*k+mp] * STCP[3*l+lp] );
            corr[t] -= sign * Trace( SCP[3*j+lp] * STCP[3*k+mp] ) * Trace( SCP[3*i+np] * STCP[3*m+kp] * SCP[3*n+jp] * STCP[3*l+ip] );
            corr[t] -= sign * Trace( SCP[3*j+ip] * STCP[3*k+jp] ) * Trace( SCP[3*i+np] * STCP[3*m+kp] * SCP[3*n+mp] * STCP[3*l+lp] );
            corr[t] += sign * Trace( SCP[3*i+ip] * STCP[3*k+jp] * SCP[3*j+kp] * STCP[3*m+np] * SCP[3*n+mp] * STCP[3*l+lp] );
            corr[t] += sign * Trace( SCP[3*i+ip] * STCP[3*k+jp] * SCP[3*j+lp] * STCP[3*n+mp] * SCP[3*m+kp] * STCP[3*l+np] );
            corr[t] += sign * Trace( SCP[3*i+ip] * STCP[3*k+jp] * SCP[3*j+lp] * STCP[3*n+mp] * SCP[3*m+np] * STCP[3*l+kp] );
            corr[t] += sign * Trace( SCP[3*i+ip] * STCP[3*k+jp] * SCP[3*j+np] * STCP[3*m+kp] * SCP[3*n+mp] * STCP[3*l+lp] );
            corr[t] += sign * Trace( SCP[3*i+ip] * STCP[3*n+jp] * SCP[3*m+kp] * STCP[3*j+np] * SCP[3*k+mp] * STCP[3*l+lp] );
            corr[t] += sign * Trace( SCP[3*i+ip] * STCP[3*n+jp] * SCP[3*m+lp] * STCP[3*k+mp] * SCP[3*j+kp] * STCP[3*l+np] );
            corr[t] += sign * Trace( SCP[3*i+ip] * STCP[3*n+jp] * SCP[3*m+lp] * STCP[3*k+mp] * SCP[3*j+np] * STCP[3*l+kp] );
            corr[t] += sign * Trace( SCP[3*i+ip] * STCP[3*n+jp] * SCP[3*m+np] * STCP[3*j+kp] * SCP[3*k+mp] * STCP[3*l+lp] );
            corr[t] += sign * Trace( SCP[3*i+kp] * STCP[3*j+np] * SCP[3*k+jp] * STCP[3*m+ip] * SCP[3*n+mp] * STCP[3*l+lp] );
            corr[t] += sign * Trace( SCP[3*i+kp] * STCP[3*j+np] * SCP[3*k+mp] * STCP[3*m+lp] * SCP[3*n+jp] * STCP[3*l+ip] );
            corr[t] += sign * Trace( SCP[3*i+kp] * STCP[3*m+np] * SCP[3*n+jp] * STCP[3*j+ip] * SCP[3*k+mp] * STCP[3*l+lp] );
            corr[t] += sign * Trace( SCP[3*i+kp] * STCP[3*m+np] * SCP[3*n+mp] * STCP[3*j+lp] * SCP[3*k+jp] * STCP[3*l+ip] );
            corr[t] += sign * Trace( SCP[3*i+lp] * STCP[3*k+mp] * SCP[3*j+ip] * STCP[3*n+jp] * SCP[3*m+kp] * STCP[3*l+np] );
            corr[t] += sign * Trace( SCP[3*i+lp] * STCP[3*k+mp] * SCP[3*j+ip] * STCP[3*n+jp] * SCP[3*m+np] * STCP[3*l+kp] );
            corr[t] += sign * Trace( SCP[3*i+lp] * STCP[3*k+mp] * SCP[3*j+kp] * STCP[3*m+np] * SCP[3*n+jp] * STCP[3*l+ip] );
            corr[t] += sign * Trace( SCP[3*i+lp] * STCP[3*k+mp] * SCP[3*j+np] * STCP[3*m+kp] * SCP[3*n+jp] * STCP[3*l+ip] );
            corr[t] += sign * Trace( SCP[3*i+lp] * STCP[3*n+mp] * SCP[3*m+ip] * STCP[3*k+jp] * SCP[3*j+kp] * STCP[3*l+np] );
            corr[t] += sign * Trace( SCP[3*i+lp] * STCP[3*n+mp] * SCP[3*m+ip] * STCP[3*k+jp] * SCP[3*j+np] * STCP[3*l+kp] );
            corr[t] += sign * Trace( SCP[3*i+lp] * STCP[3*n+mp] * SCP[3*m+kp] * STCP[3*j+np] * SCP[3*k+jp] * STCP[3*l+ip] );
            corr[t] += sign * Trace( SCP[3*i+lp] * STCP[3*n+mp] * SCP[3*m+np] * STCP[3*j+kp] * SCP[3*k+jp] * STCP[3*l+ip] );
            corr[t] += sign * Trace( SCP[3*i+np] * STCP[3*j+kp] * SCP[3*k+jp] * STCP[3*m+ip] * SCP[3*n+mp] * STCP[3*l+lp] );
            corr[t] += sign * Trace( SCP[3*i+np] * STCP[3*j+kp] * SCP[3*k+mp] * STCP[3*m+lp] * SCP[3*n+jp] * STCP[3*l+ip] );
            corr[t] += sign * Trace( SCP[3*i+np] * STCP[3*m+kp] * SCP[3*n+jp] * STCP[3*j+ip] * SCP[3*k+mp] * STCP[3*l+lp] );
            corr[t] += sign * Trace( SCP[3*i+np] * STCP[3*m+kp] * SCP[3*n+mp] * STCP[3*j+lp] * SCP[3*k+jp] * STCP[3*l+ip] );
          }
        }
      }
    }
  }
}
