#include "run_nnpp_4pt.h"
#include "run_sigma_4pt.h"

Vcomplex run_nnpp_4pt(SpinMat * wall_prop,       // prop from source
                      Vcomplex * T,              // big ol' tensor
                      int tp,                    // time of sink
                      int nx,                    // spatial lattice extent
                      int xc, int yc, int zc) {  // sink coordinates
  WeylMat idW = ExtractWeyl(id) * 0.5;
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
  for(int ii=0; ii<1296; ii++) 
  {
    Vcomplex tmp;
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
    #include "run_nnpp_4pt.inc"
    corr_nnpp_4pt += tmp * 2 * sign;
  }
  return (corr_nnpp_4pt);
}
