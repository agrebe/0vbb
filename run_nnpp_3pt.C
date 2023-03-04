#include "run_nnpp_4pt.h"
#include "tensor_contractions.h"

/*
 * This code combines the 4-index tensor T computed from two seqprops
 * at a single point with the source and sink dinucleon interpolating
 * operators to give a 3-point function.  This function is computed for
 * all the different gamma matrix insertions at the 4-quark operator
 * needed to form the complete basis of dimension-9 short-distance operators
 */

void run_nnpp_3pt(Vcomplex * T,             // precomputed tensor
                  SpinMat * wall_prop,      // source propagator
                  Vcomplex * corr,          // 3-point correlator
                  int nt, int nx,           // size of lattice
                  int sep,                  // sink - source time
                  int tm,                   // source time
                  int t,                    // operator - source time
                  int num_currents,         // {SS, PP, VV, AA, TT, etc.}
                  int xc, int yc, int zc) { // sink coordinates
  // define identity matrix (will be needed in some contractions)
  WeylMat idW = ExtractWeyl(id) * 0.5;

  // compute sink time
  int tp = (tm + sep) % nt;

  // create array of temporaries for the different current choices
  Vcomplex corr_nnpp_3pt[num_currents];

  // extract propagator from source to sink
  int loc = ((tp * nx + zc) * nx + yc) * nx + xc;
  SpinMat * Sl_xw_prop = wall_prop + 9 * loc;

  // precompute transpose of source-to-sink prop
  // as well as prop times C * g5 (C = charge conjugation matrix)
  // since this pattern appears in source/sink interpolators
  WeylMat cgs = ExtractWeyl(pp * cg5 * pp);
  WeylMat Sl_xw [9], Sl_xw_T [9], Sl_xw_CG [9], Sl_xw_T_CG [9];
  for (int c = 0; c < 9; c ++) {
    Sl_xw[c] = ExtractWeyl(pp * Sl_xw_prop[c] * pp);
    Sl_xw_T[c] = Sl_xw[c].transpose();
    Sl_xw_CG[c] = Sl_xw[c] * cgs;
    Sl_xw_T_CG[c] = Sl_xw_T[c] * cgs;
  }

  // loop over current (SS, PP, VV, AA, TT, ...)
  for (int index = 0; index < num_currents; index ++) {
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
      corr_nnpp_3pt[index] += tmp * 2 * sign;
    }
    // increment index of T by 6^4 and move on to next current
    T += 1296;
  }
  // Add the temporary variable into the array being returned
  // There is an outer loop over the sink position, 
  // so the value computed here is summed with what is already
  // in the array from the previous sink times
  // This array is indexed by source-sink and source-operator separations,
  // as well as by which set of gamma matrices was inserted at the operator
  for (int i = 0; i < num_currents; i ++)
    corr[(sep * nt + t) * num_currents + i] += corr_nnpp_3pt[i];
}
