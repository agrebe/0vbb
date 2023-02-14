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

    /*
    for(int c1=0; c1<3; c1++){
    for(int c2=0; c2<3; c2++){
      // commonly used products of matrices
      SpinMat prod0 = S_tmx[3*c1+ip] * CG5SsCG5[3*k+jp] * g5 * S_tpx_T[3*j+c1];
      SpinMat prod1 = S_tmx[3*c1+ip] * CG5SsCG5[3*k+jp] * g5 * S_tpx_T[3*j+c2];
      SpinMat prod2 = S_tmx[3*c2+kp] * g5 * S_tpx_T[3*i+c2];
      SpinMat prod3 = S_tmx[3*c2+kp] * g5 * S_tpx_T[3*i+c1];

      // SS
      tmp[0] += sign * Trace( g5 * prod0 ) * Trace( g5 * prod2 ); 
      tmp[1] += sign * Trace( g5 * prod1 * g5 * prod3);
      tmp[2] += sign * Trace( g5 * prod1 ) * Trace( g5 * prod3 ); 
      tmp[3] += sign * Trace( g5 * prod0 * g5 * prod2);
      
      // PP
      tmp[4] += sign * Trace( prod0 ) * Trace( prod2 );                  
      tmp[5] += sign * Trace( prod1 * prod3);                            
      tmp[6] += sign * Trace( prod1 ) * Trace( prod3 );       
      tmp[7] += sign * Trace( prod0 * prod2);
      for(int mu=0; mu<4; mu++)
      {
        SpinMat gmu;
        if (mu == 0){ gmu = gx; }
        else if(mu == 1){ gmu = gy; }
        else if(mu == 2){ gmu = gz; }
        else if(mu == 3){ gmu = gt; }

        SpinMat gvs = g5 * gmu;
        SpinMat gas = g5 * gmu * g5;

        // VV
        tmp[8]  += sign * Trace( gvs * prod0 ) * Trace( gvs * prod2 ); 
        tmp[9]  += sign * Trace( gvs * prod1 * gvs * prod3);   
        tmp[10] += sign * Trace( gvs * prod1 ) * Trace( gvs * prod3 ); 
        tmp[11] += sign * Trace( gvs * prod0 * gvs * prod2);

        // AA
        tmp[12] += sign * Trace( gas * prod0 ) * Trace( gas * prod2 ); 
        tmp[13] += sign * Trace( gas * prod1 * gas * prod3);   
        tmp[14] += sign * Trace( gas * prod1 ) * Trace( gas * prod3 );
        tmp[15] += sign * Trace( gas * prod0 * gas * prod2);
      } //mu
    }} // c1, c2
    */
  } // colors
  #pragma omp critical
  for (int i = 0; i < 10; i ++)
    corr[(sep * nt + t) * 10 + i] += tmp[i];
}
