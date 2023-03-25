#include "run_sigma_4pt.h"
#include "tensor_contractions.h"

/*
 * This file is the analog of run_nnpp_4pt.C but for the Sigam^- -> Sigma^+
 * transition.  It contains the (hard-coded) contractions necessary
 * to compute the sigma 4-point function.
 */

Vcomplex run_sigma_4pt(SpinMat * wall_prop,       // prop from source
                       Vcomplex * T,              // big ol' tensor
                       int tp,                    // time of sink
                       int nx,                    // spatial lattice extent
                       int xc, int yc, int zc) {  // sink coordinates
  // identity matrix
  WeylMat idW = ExtractWeyl(id) * 0.5;
  Vcomplex corr_sigma_4pt = Vcomplex();

  // extract propagator from source to sink
  int loc = ((tp * nx + zc) * nx + yc) * nx + xc;
  SpinMat * Ss_xw = wall_prop + 9 * loc;

  Vcomplex tmp;

  // extract propagator from source to operator
  SpinMat * Sl_xy = wall_prop + 9 * loc;

  // precompute transposed source-to-sink propagator
  SpinMat T_CG5Ss [9];
  for (int c = 0; c < 9; c ++)
    T_CG5Ss[c] = (cg5 * Ss_xw[c]).transpose();

  // precompute source-to-sink prop times gamma matrices
  WeylMat CG5SsCG5_xw[9], Hay[9], SnuHbz[9];
  for (int c = 0; c < 9; c ++) {
    CG5SsCG5_xw[c] = ExtractWeyl(pp * cg5 * T_CG5Ss[c] * pp);
  }
  // loop over colors
  for(int ii=0; ii<36; ii++)
  {
    const int& i      = color_idx_1[7*ii+0];
    const int& j      = color_idx_1[7*ii+1];
    const int& k      = color_idx_1[7*ii+2];
    const int& ip     = color_idx_1[7*ii+3];
    const int& jp     = color_idx_1[7*ii+4];
    const int& kp     = color_idx_1[7*ii+5];
    const double sign = color_idx_1[7*ii+6];
    // The original contractions, written in terms of sequential props
    // Hay is a seqprop through one operator
    // SnuHbz is the other seqprop convolved with the neutrino prop
    // These are included for reference, as these map one-to-one
    // onto the actual contractions that use the precomputed tensor T
    /*
    tmp += sign * Trace(SnuHbz[3*i+kp]) * Trace(Hay[3*j+ip] * CG5SsCG5_xw[3*k+jp]);
    tmp += sign * Trace(Hay[3*i+kp]) * Trace(SnuHbz[3*j+ip] * CG5SsCG5_xw[3*k+jp]);
    tmp -= sign * Trace(Hay[3*i+ip] * CG5SsCG5_xw[3*k+jp] * SnuHbz[3*j+kp]);
    tmp -= sign * Trace(SnuHbz[3*i+ip] * CG5SsCG5_xw[3*k+jp] * Hay[3*j+kp]);
    */

    // The contractions actually performed using the precomputed tensor
    tmp += sign * two_traces(T, j, ip, i, kp, CG5SsCG5_xw[3*k+jp], idW);
    tmp += sign * two_traces(T, i, kp, j, ip, idW, CG5SsCG5_xw[3*k+jp]);
    tmp -= sign * one_trace (T, i, ip, j, kp, CG5SsCG5_xw[3*k+jp], idW);
    tmp -= sign * one_trace (T, j, kp, i, ip, idW, CG5SsCG5_xw[3*k+jp]);
  }
  corr_sigma_4pt += tmp;
  return (corr_sigma_4pt);
}
