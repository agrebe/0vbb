#include "run_sigma_4pt.h"

// compute the tensor contraction
// Tr[S_zxw A] Tr[S_zyw B] c
// where the two sequential propagators are encoded in T
Vcomplex two_traces(Vcomplex * T,
                    int c1, int c2, int c3, int c4,
                    WeylMat A, WeylMat B) {
  T += tensor_index(c1, c2, c3, c4, 0, 0, 0, 0);
  Vcomplex sum = Vcomplex();
  // loop over spins
  for (int s1 = 0; s1 < 2; s1 ++)
    for (int s2 = 0; s2 < 2; s2 ++)
      for (int s3 = 0; s3 < 2; s3 ++)
        for (int s4 = 0; s4 < 2; s4 ++) {
          Vcomplex tensor_value = *(T++);
          sum += tensor_value * ((Vcomplex *) &A)[s2*2+s1] 
                              * ((Vcomplex *) &B)[s4*2+s3];
        }
  return sum;
}

// compute the tensor contraction
// Tr[S_zxw A S_zyw B] c
Vcomplex one_trace(Vcomplex * T,
                   int c1, int c2, int c3, int c4,
                   WeylMat A, WeylMat B) {
  T += tensor_index(c1, c2, c3, c4, 0, 0, 0, 0);
  Vcomplex sum = Vcomplex();
  for (int s1 = 0; s1 < 2; s1 ++)
    for (int s2 = 0; s2 < 2; s2 ++)
      for (int s3 = 0; s3 < 2; s3 ++)
        for (int s4 = 0; s4 < 2; s4 ++) {
          Vcomplex tensor_value = *(T++);
          sum += tensor_value * ((Vcomplex *) &A)[s2*2+s3] 
                              * ((Vcomplex *) &B)[s4*2+s1];
        }
  return sum;
}

// compute the tensor contraction
// Tr[S_zxw A S_zyw^T B] c
Vcomplex one_trace_transposed(Vcomplex * T,
                              int c1, int c2, int c3, int c4,
                              WeylMat A, WeylMat B) {
  T += tensor_index(c1, c2, c3, c4, 0, 0, 0, 0);
  Vcomplex sum = Vcomplex();
  for (int s1 = 0; s1 < 2; s1 ++)
    for (int s2 = 0; s2 < 2; s2 ++)
      for (int s3 = 0; s3 < 2; s3 ++)
        for (int s4 = 0; s4 < 2; s4 ++) {
          Vcomplex tensor_value = *(T++);
          sum += tensor_value * ((Vcomplex *) &A)[s2*2+s4] 
                              * ((Vcomplex *) &B)[s3*2+s1];
        }
  return sum;
}

Vcomplex run_sigma_4pt(SpinMat * wall_prop,       // prop from source
                       Vcomplex * T,              // big ol' tensor
                       int tx,                    // time of operator
                       int tp,                    // time of sink
                       int nx,                    // spatial lattice extent
                       int block_size,            // sparsening at operator
                       int xc, int yc, int zc) {  // sink coordinates
  Vcomplex one = Vcomplex(1, 0);
  WeylMat idW = ExtractWeyl(id) * 0.5;
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
    /*
    tmp += sign * Trace(SnuHbz[3*i+kp]) * Trace(Hay[3*j+ip] * CG5SsCG5_xw[3*k+jp]);
    tmp += sign * Trace(Hay[3*i+kp]) * Trace(SnuHbz[3*j+ip] * CG5SsCG5_xw[3*k+jp]);
    tmp -= sign * Trace(Hay[3*i+ip] * CG5SsCG5_xw[3*k+jp] * SnuHbz[3*j+kp]);
    tmp -= sign * Trace(SnuHbz[3*i+ip] * CG5SsCG5_xw[3*k+jp] * Hay[3*j+kp]);
    */
    tmp += sign * two_traces(T, j, ip, i, kp, CG5SsCG5_xw[3*k+jp], idW);
    tmp += sign * two_traces(T, i, kp, j, ip, idW, CG5SsCG5_xw[3*k+jp]);
    tmp -= sign * one_trace (T, i, ip, j, kp, CG5SsCG5_xw[3*k+jp], idW);
    tmp -= sign * one_trace (T, j, kp, i, ip, idW, CG5SsCG5_xw[3*k+jp]);
  }
  corr_sigma_4pt += tmp;
  return (corr_sigma_4pt);
}
