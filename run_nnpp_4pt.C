#include "run_sigma_4pt.h"

Vcomplex run_nnpp_4pt(SpinMat * wall_prop,       // prop from source
                      SpinMat * point_prop,      // prop from sink
                      SpinMat * SnuHz,           // seqprop * nu_prop
                      int tx,                    // time of operator
                      int tp,                    // time of sink
                      int nx,                    // spatial lattice extent
                      int block_size,            // sparsening at operator
                      int xc, int yc, int zc) {  // sink coordinates
  Vcomplex corr_nnpp_4pt = Vcomplex();

  WeylMat cgs = ExtractWeyl(pp * cg5 * pp);

  // extract propagator from source to sink
  int loc = ((tp * nx + zc) * nx + yc) * nx + xc;
  SpinMat * Sl_xw_prop = wall_prop + 9 * loc;
  WeylMat Sl_xw [9], Sl_xw_T [9], Sl_xw_CG [9], Sl_xw_T_CG [9];
  for (int c = 0; c < 9; c ++) {
    Sl_xw[c] = ExtractWeyl(pp * Sl_xw_prop[c] * pp);
    Sl_xw_T[c] = Sl_xw[c].transpose();
    Sl_xw_CG[c] = Sl_xw[c] * cgs;
    Sl_xw_T_CG[c] = Sl_xw_T[c] * cgs;
  }

  // loop over operator insertions
  int nx_blocked = nx / block_size;
  #pragma omp parallel for collapse(3)
  for (int z = 0; z < nx; z += block_size) {
    for (int y = 0; y < nx; y += block_size) {
      for (int x = 0; x < nx; x += block_size) {
        Vcomplex tmp [576];
        int loc = ((tx * nx + z) * nx + y) * nx + x;
        int idx = ((z / block_size) * nx_blocked 
            + (y / block_size)) * nx_blocked 
            + (x / block_size);
        SpinMat * Sl_xy = wall_prop + 9 * loc;
        SpinMat * Sl_wy = point_prop + 9 * loc;
        for(int mu=0; mu<4; mu++)
        {

          SpinMat gmu;
          if     (mu == 0){ gmu = gx; }
          else if(mu == 1){ gmu = gy; }
          else if(mu == 2){ gmu = gz; }
          else if(mu == 3){ gmu = gt; }

          WeylMat Sl_xyw[9], Sl_xyw_CG[9];
          for (int c1 = 0; c1 < 3; c1 ++)
            for (int c2 = 0; c2 < 3; c2 ++)
              for (int c3 = 0; c3 < 3; c3 ++)
                Sl_xyw[3*c1+c2] += ExtractWeyl(pp * g5 * Sl_wy[3*c3+c1].hconj() * gmu * pl * Sl_xy[3*c3+c2] * pp);
          for (int c = 0; c < 9; c ++)
            Sl_xyw_CG[c] = Sl_xyw[c] * cgs;

          WeylMat SnuHbz[9], SnuHbz_T[9], SnuHbz_CG[9], SnuHbz_T_CG[9];
          for (int c = 0; c < 9; c ++) {
            SnuHbz[c] = ExtractWeyl(pp * SnuHz[(4*idx+mu)*9+c] * pp);
            SnuHbz_T[c] = SnuHbz[c].transpose();
            SnuHbz_CG[c] = SnuHbz[c] * cgs;
            SnuHbz_T_CG[c] = SnuHbz_T[c] * cgs;
          }
          #include "autogen/run_nnpp_4pt_1.inc"
          for(int ii=0; ii<1296; ii++) 
          {
            const int& a      = color_idx_2[13*ii+0];
            const int& b      = color_idx_2[13*ii+1];
            const int& c      = color_idx_2[13*ii+2];
            const int& d      = color_idx_2[13*ii+3];
            const int& e      = color_idx_2[13*ii+4];
            const int& f      = color_idx_2[13*ii+5];
            const int& g      = color_idx_2[13*ii+6];
            const int& h      = color_idx_2[13*ii+7];
            const int& i      = color_idx_2[13*ii+8];
            const int& j      = color_idx_2[13*ii+9];
            const int& k      = color_idx_2[13*ii+10];
            const int& l      = color_idx_2[13*ii+11];
            const double sign = color_idx_2[13*ii+12];
            #include "autogen/run_nnpp_4pt_2.inc"
          }
        }
        #include "autogen/run_nnpp_4pt_3.inc"
        Vcomplex total = Vcomplex();
        for (int c = 0; c < 576; c ++)
          total += tmp[c];
        #pragma omp critical
        corr_nnpp_4pt += total;
      }
    }
  }
  return (corr_nnpp_4pt);
}
