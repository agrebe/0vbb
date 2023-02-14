#include "run_sigma_3pt.h"

void run_sigma_3pt(Vcomplex * T,             // precomputed tensor
                   SpinMat * wall_prop,      // source propagator
                   Vcomplex * corr,          // 3-point correlator
                   int nt, int nx,           // size of lattice
                   int sep,                  // sink - source time
                   int tm,                   // source time
                   int t,                    // operator - source time
                   int xc, int yc, int zc) { // sink coordinates
  WeylMat idW = ExtractWeyl(id) * 0.5;
  // extract propagator from source to sink
  int tp = (tm + sep) % nt;
  int loc = ((tp * nx + zc) * nx + yc) * nx + xc;
  SpinMat * S_tm_to_tp = wall_prop + 9 * loc;
  Vcomplex tmp[10];
  // compute operator time
  int tx = (t + tm) % nt;
  WeylMat CG5SsCG5 [9];
  for (int c = 0; c < 9; c ++) {
    // Note: The -1 factor at the end is needed for consistency with CPS code
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

    // loop over index of {SS, PP, VV, AA, TT}
    for (int index = 0; index < 5; index ++) {
      tmp[0 + index * 2] += sign 
        * two_traces(T + index * 1296, j, ip, i, kp, CG5SsCG5[3*k+jp], idW);
      tmp[1 + index * 2] += sign 
        * one_trace (T + index * 1296, i, ip, j, kp, CG5SsCG5[3*k+jp], idW);
    }

  } // colors
  #pragma omp critical
  for (int i = 0; i < 10; i ++)
    corr[(sep * nt + t) * 10 + i] += tmp[i];
}
