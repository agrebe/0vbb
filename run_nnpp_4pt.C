#include "run_nnpp_4pt.h"
#include "tensor_contractions.h"

/*
 * Analogously to run_nnpp_3pt.C, this computes 4-point correlation
 * functions for the nn -> pp transition from the precomputed tensor T.
 * Unlike the 3-point code, however, this code computes the contraction
 * at a given set of separations between the operator, source, and sink
 * times and therefore returns a single complex number
 * rather than writing into a whole array.
 */

Vcomplex run_nnpp_4pt(SpinMat * wall_prop,       // prop from source
                      Vcomplex * T,              // precomputed tensor
                      int tp,                    // time of sink
                      int nx,                    // spatial lattice extent
                      int xc, int yc, int zc) {  // sink coordinates
  // define identity matrix (will be needed in some contractions)
  WeylMat idW = ExtractWeyl(id) * 0.5;

  // Zero out return value
  Vcomplex corr_nnpp_4pt = Vcomplex();

  // precompute positive-projected version of C * g5
  WeylMat cgs = ExtractWeyl(pp * cg5 * pp);

  // extract propagator from source to sink
  int loc = ((tp * nx + zc) * nx + yc) * nx + xc;
  SpinMat * Sl_xw_prop = wall_prop + 9 * loc;

  // precompute transpose of source-to-sink prop
  // as well as prop times C * g5 (C = charge conjugation matrix)
  // since this pattern appears in source/sink interpolators
  WeylMat Sl_xw [9], Sl_xw_T [9], Sl_xw_CG [9], Sl_xw_T_CG [9];
  for (int c = 0; c < 9; c ++) {
    Sl_xw[c] = ExtractWeyl(pp * Sl_xw_prop[c] * pp);
    Sl_xw_T[c] = Sl_xw[c].transpose();
    Sl_xw_CG[c] = Sl_xw[c] * cgs;
    Sl_xw_T_CG[c] = Sl_xw_T[c] * cgs;
  }
  // loop over colors
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
    // include auto-generated file of all nn->pp contractions
    #include "run_nnpp.inc"
    // there is a symmetry that lets us only compute half the contractions
    // full sum is contractions we have computed times 2
    corr_nnpp_4pt += tmp * 2 * sign;
  }
  return (corr_nnpp_4pt);
}
