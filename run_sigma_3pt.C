#include "run_sigma_3pt.h"

void run_sigma_3pt(SpinMat* wall_prop,      // wall prop at source
                   SpinMat* point_prop,     // point prop at sink
                   Vcomplex * corr,         // 3-point correlator
                   int block_size_sparsen,  // sparsening factor at operator
                   int nt, int nx,          // size of lattice
                   int tm,                  // source time
                   int sep,                 // sink - source time
                   int xc, int yc, int zc) {// sink spatial coords
  // extract propagator from source to sink
  int tp = (tm + sep) % nt;
  int loc = ((tp * nx + zc) * nx + yc) * nx + xc;
  SpinMat * S_tm_to_tp = wall_prop + 9 * loc;
  // loop over operator insertions
  #pragma omp parallel for collapse(4)
  for (int t = 3; t <= sep - 3; t ++) {
    for (int z = 0; z < nx; z += block_size_sparsen) {
      for (int y = 0; y < nx; y += block_size_sparsen) {
        for (int x = 0; x < nx; x += block_size_sparsen) {
          Vcomplex tmp[16];
          // compute operator time
          int tx = (t + tm) % nt;
          int loc = ((tx * nx + z) * nx + y) * nx + x;
          SpinMat * S_tm_to_x = wall_prop + 9 * loc;
          SpinMat * S_tp_to_x = point_prop + 9 * loc;
          SpinMat CG5SsCG5 [9], S_tmx [9], S_tpx [9], S_tmx_T [9], S_tpx_T [9];
          for (int c = 0; c < 9; c ++) {
            // Note: The -1 factor at the end is needed for consistency with CPS code
            // There, we had a factor of cg5.transpose() = - cg5
            CG5SsCG5[c] = pp * cg5 * S_tm_to_tp[c].transpose() * pp * cg5 * -1;
            S_tmx[c] = S_tm_to_x[c] * pp;
            S_tpx[c] = S_tp_to_x[c] * pp;
          }
          for (int c1 = 0; c1 < 3; c1 ++) {
            for (int c2 = 0; c2 < 3; c2 ++) {
              S_tmx_T[3*c1+c2] = pm * S_tm_to_x[3*c2+c1].hconj();
              S_tpx_T[3*c1+c2] = pm * S_tp_to_x[3*c2+c1].hconj();
            }
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
          } // colors
          #pragma omp critical
          for (int i = 0; i < 16; i ++)
            corr[(sep * nt + t) * nt + i] += tmp[i];
        }
      }
    } // z, y, x
  } // t
}
