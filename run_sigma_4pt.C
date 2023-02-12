#include "run_sigma_4pt.h"

Vcomplex run_sigma_4pt(SpinMat * wall_prop,       // prop from source
                       Vcomplex * T,              // big ol' tensor
                       int tx,                    // time of operator
                       int tp,                    // time of sink
                       int nx,                    // spatial lattice extent
                       int block_size,            // sparsening at operator
                       int xc, int yc, int zc) {  // sink coordinates
  Vcomplex corr_sigma_4pt = Vcomplex();

  // extract propagator from source to sink
  int loc = ((tp * nx + zc) * nx + yc) * nx + xc;
  SpinMat * Ss_xw = wall_prop + 9 * loc;
  // loop over operator insertions
  int nx_blocked = nx / block_size;
  Vcomplex tmp;
  SpinMat * Sl_xy = wall_prop + 9 * loc;
  SpinMat T_CG5Ss [9];
  for (int c = 0; c < 9; c ++)
    T_CG5Ss[c] = (cg5 * Ss_xw[c]).transpose();

  WeylMat CG5SsCG5_xw[9], Hay[9], SnuHbz[9];
  for (int c = 0; c < 9; c ++) {
    CG5SsCG5_xw[c] = ExtractWeyl(pp * cg5 * T_CG5Ss[c] * pp);
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
    // loop over spins
    for (int s1 = 0; s1 < 2; s1 ++)
      for (int s2 = 0; s2 < 2; s2 ++)
        for (int s3 = 0; s3 < 2; s3 ++)
          for (int s4 = 0; s4 < 2; s4 ++) {
            // tmp += sign * Trace(SnuHbz[3*i+kp]) * Trace(Hay[3*j+ip] * CG5SsCG5_xw[3*k+jp]);
            Vcomplex tensor_value = T[tensor_index(j, ip, i, kp, s1, s2, s3, s4)];
            // trace over spins of SnuHbz (s3 and s4)
            if (s3 == s4) {
              // trace spins of Hay (s1 and s2) against CG5SsCG5_xw
              Vcomplex CG5SsCG5_xw_value = *(((Vcomplex *) (CG5SsCG5_xw + 3*k + jp)) + 2*s2 + s1);
              tmp += sign * tensor_value * CG5SsCG5_xw_value;
            }
            tensor_value = T[tensor_index(i, kp, j, ip, s1, s2, s3, s4)];
            // trace spins of Hay
            if (s1 == s2) {
              // trace spins of SnuHbz against CG5SsCG5_xw
              Vcomplex CG5SsCG5_xw_value = *(((Vcomplex *) (CG5SsCG5_xw + 3*k + jp)) + 2*s4 + s3);
              tmp += sign * tensor_value * CG5SsCG5_xw_value;
            }
            tensor_value = T[tensor_index(i, ip, j, kp, s1, s2, s3, s4)];
            if (s4 == s1) {
              Vcomplex CG5SsCG5_xw_value = *(((Vcomplex *) (CG5SsCG5_xw + 3*k + jp)) + 2*s2 + s3);
              tmp -= sign * tensor_value * CG5SsCG5_xw_value;
            }
            tensor_value = T[tensor_index(j, kp, i, ip, s1, s2, s3, s4)];
            if (s2 == s3) {
              Vcomplex CG5SsCG5_xw_value = *(((Vcomplex *) (CG5SsCG5_xw + 3*k + jp)) + 2*s4 + s1);
              tmp -= sign * tensor_value * CG5SsCG5_xw_value;
            }
          }
    /*
    tmp -= sign * Trace(Hay[3*i+ip] * CG5SsCG5_xw[3*k+jp] * SnuHbz[3*j+kp]);
    tmp -= sign * Trace(SnuHbz[3*i+ip] * CG5SsCG5_xw[3*k+jp] * Hay[3*j+kp]);
    */
  }
  corr_sigma_4pt += tmp;
  return (corr_sigma_4pt);
}
