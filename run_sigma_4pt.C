#include "run_sigma_4pt.h"

void run_sigma_4pt(SpinMat * wall_prop,       // prop from source
                   SpinMat * point_prop,      // prop from sink
                   SpinMat * SnuHz,           // seqprop * nu_prop
                   Vcomplex * corr_sigma_4pt, // output (4-point correlator)
                   int tx,                    // time of operator
                   int tp,                    // time of sink
                   int nx,                    // spatial lattice extent
                   int block_size,            // sparsening at operator
                   int xc, int yc, int zc) {  // sink coordinates
  // extract propagator from source to sink
  int loc = ((tp * nx + zc) * nx + yc) * nx + xc;
  SpinMat * Ss_xw = wall_prop + 9 * loc;
  // loop over operator insertions
  int nx_blocked = nx / block_size;
  for (int z = 0; z < nx; z += block_size) {
    for (int y = 0; y < nx; y += block_size) {
      for (int x = 0; x < nx; x += block_size) {
        Vcomplex tmp;
        int loc = ((tx * nx + z) * nx + y) * nx + x;
        int idx = ((z / block_size) * nx_blocked 
            + (y / block_size)) * nx_blocked 
            + (x / block_size);
        SpinMat * Sl_xy = wall_prop + 9 * loc;
        SpinMat * Sl_wy = point_prop + 9 * loc;
        SpinMat T_CG5Ss [9];
        for (int c = 0; c < 9; c ++)
          T_CG5Ss[c] = (cg5 * Ss_xw[c]).transpose();
        for(int mu=0; mu<4; mu++)
        {

          SpinMat gmu;
          if     (mu == 0){ gmu = gx; }
          else if(mu == 1){ gmu = gy; }
          else if(mu == 2){ gmu = gz; }
          else if(mu == 3){ gmu = gt; }

          SpinMat Ha[9];
          for (int c1 = 0; c1 < 3; c1 ++)
            for (int c2 = 0; c2 < 3; c2 ++)
              for (int c3 = 0; c3 < 3; c3 ++)
                Ha[3*c1+c2] += g5 * Sl_wy[3*c3+c1].hconj() * gmu * pl * Sl_xy[3*c3+c2];

          WeylMat CG5SsCG5_xw[9], Hay[9], SnuHbz[9];
          for (int c = 0; c < 9; c ++) {
            CG5SsCG5_xw[c] = ExtractWeyl(pp * cg5 * T_CG5Ss[c] * pp);
            Hay[c] = ExtractWeyl(pp * Ha[c] * pp);
            SnuHbz[c] = ExtractWeyl(pp * SnuHz[(4*idx+mu)*9+c] * pp);
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
            tmp += sign * Trace(SnuHbz[3*i+kp]) * Trace(Hay[3*j+ip] * CG5SsCG5_xw[3*k+jp]);
            tmp += sign * Trace(Hay[3*i+kp]) * Trace(SnuHbz[3*j+ip] * CG5SsCG5_xw[3*k+jp]);
            tmp -= sign * Trace(Hay[3*i+ip] * CG5SsCG5_xw[3*k+jp] * SnuHbz[3*j+kp]);
            tmp -= sign * Trace(SnuHbz[3*i+ip] * CG5SsCG5_xw[3*k+jp] * Hay[3*j+kp]);
          }
        }
        *corr_sigma_4pt += tmp;
      }
    }
  }
}
