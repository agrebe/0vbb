#include "run_sigma_3pt.h"
#include "tensor_contractions.h"

/*
 * This file is the analog of run_nnpp_3pt.C but for the Sigma^- -> Sigma^+
 * transition.  Since this transition is simpler, the contractions are
 * coded by hand and optimized less than the nn -> pp case.
 * Unlike for the nn -> pp transition, the different spinor contractions
 * are written out separately.  This is feasible since there are few
 * of them and is useful for debugging and recognizing patterns.
 */

void run_sigma_3pt(Vcomplex * T,             // precomputed tensor
                   SpinMat * wall_prop,      // source propagator
                   Vcomplex * corr,          // 3-point correlator
                   int nt, int nx,           // size of lattice
                   int sep,                  // sink - source time
                   int tm,                   // source time
                   int t,                    // operator - source time
                   int num_currents,         // {SS, PP, VV, AA, TT}
                   int xc, int yc, int zc) { // sink coordinates
  // define identity matrix
  WeylMat idW = ExtractWeyl(id) * 0.5;

  // extract propagator from source to sink
  int tp = (tm + sep) % nt;
  int loc = ((tp * nx + zc) * nx + yc) * nx + xc;
  SpinMat * S_tm_to_tp = wall_prop + 9 * loc;

  // create array of temporaries for contractions
  Vcomplex tmp[num_currents * 4];

  // compute operator time
  int tx = (t + tm) % nt;

  // precompute product of source-to-sink prop with required gamma matrices
  WeylMat CG5SsCG5 [9];
  for (int c = 0; c < 9; c ++) {
    // Note: The -1 factor at the end is needed for consistency
    // with a former version of the code (written in CPS)
    // There, we had a factor of cg5.transpose() = - cg5
    CG5SsCG5[c] = ExtractWeyl(pp * cg5 * S_tm_to_tp[c].transpose() * pp * cg5 * -1);
  }
  for(int ii=0; ii<36; ii++) 
  {
    const int& i      = color_idx_1[7*ii+0];
    const int& j      = color_idx_1[7*ii+1];
    const int& k      = color_idx_1[7*ii+2];
    const int& ip     = color_idx_1[7*ii+3];
    const int& jp     = color_idx_1[7*ii+4];
    const int& kp     = color_idx_1[7*ii+5];
    const double sign = color_idx_1[7*ii+6];

    // loop over index of {SS, PP, VV, AA, TT, ...}
    for (int index = 0; index < num_currents; index ++) {
      // We expect (2!)^2 = 4 contractions for the sigma
      // If the two gamma matrices at the operator are the same (SS, VV, ...),
      // there is a symmetry that reduces this to 2 contractions
      // In the general case (SV, AP, ...) we need all 4 contractions
      // Thus, for SS, PP, ..., temp[0] = temp[2] and temp[1] = temp[3]
      // but this is not true for SV, AP, ...
      tmp[0 + index * 4] += sign 
        * two_traces(T + index * 1296, j, ip, i, kp, CG5SsCG5[3*k+jp], idW);
      tmp[1 + index * 4] += sign 
        * one_trace (T + index * 1296, i, ip, j, kp, CG5SsCG5[3*k+jp], idW);
      tmp[2 + index * 4] += sign 
        * two_traces(T + index * 1296, j, ip, i, kp, idW, CG5SsCG5[3*k+jp]);
      tmp[3 + index * 4] += sign 
        * one_trace (T + index * 1296, i, ip, j, kp, idW, CG5SsCG5[3*k+jp]);
    }

  } // colors

  // Add the temporary variable into the full array
  #pragma omp critical
  for (int i = 0; i < 4 * num_currents; i ++)
    corr[(sep * nt + t) * 4 * num_currents + i] += tmp[i];
}
