#include "run_sigma_3pt.h"
#include <stdio.h>

void run_nnpp_3pt(SpinMat* wall_prop,      // wall prop at source
                  SpinMat* point_prop,     // point prop at sink
                  Vcomplex * corr,         // 3-point correlator
                  int gamma_index,         // index of gamma to insert
                  int block_size_sparsen,  // sparsening factor at operator
                  int nt, int nx,          // size of lattice
                  int tm,                  // source time
                  int sep,                 // sink - source time
                  int xc, int yc, int zc) {// sink spatial coords
  
  // gamma matrix to insert
  SpinMat gs = id;
  switch(gamma_index) {
    case 0: gs = id; break;
    case 1: gs = g5; break;
    case 2: gs = gx; break;
    case 3: gs = gy; break;
    case 4: gs = gz; break;
    case 5: gs = gt; break;
    case 6: gs = g5 * gx; break;
    case 7: gs = g5 * gy; break;
    case 8: gs = g5 * gz; break;
    case 9: gs = g5 * gt; break;
    case 10: gs = gx * gy; break;
    case 11: gs = gx * gz; break;
    case 12: gs = gx * gt; break;
    case 13: gs = gy * gz; break;
    case 14: gs = gy * gt; break;
    case 15: gs = gz * gt; break;
  }
  
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
          Vcomplex tmp[288];
          int tx = (tm + t) % nt;
          int loc = ((tx * nx + z) * nx + y) * nx + x;
          SpinMat * S_tmx = wall_prop + 9 * loc;
          SpinMat * S_tpx = point_prop + 9 * loc;
          SpinMat S_txp[9];
          for (int c1 = 0; c1 < 3; c1 ++)
            for (int c2 = 0; c2 < 3; c2 ++)
              S_txp[3*c1+c2] = g5 * S_tpx[3*c2+c1].hconj() * g5;
          // construction of 2x2 Weyl matrices
          WeylMat cgs = ExtractWeyl(pp * cg5 * pp);
          WeylMat S_tmp[9], S_tmp_T[9], S_tmp_CG[9], S_tmp_T_CG[9],
                  S_tmxp[9], S_tmxp_T[9], S_tmxp_CG[9], S_tmxp_T_CG[9];
          for (int c = 0; c < 9; c ++) {
            S_tmp[c] = ExtractWeyl(pp * S_tm_to_tp[c] * pp);
            S_tmp_T[c] = S_tmp[c].transpose();
            S_tmp_CG[c] = S_tmp[c] * cgs;
            S_tmp_T_CG[c] = S_tmp_T[c] * cgs;
          }
					for (int c1 = 0; c1 < 3; c1 ++) {
						for (int c2 = 0; c2 < 3; c2 ++) {
							for (int m = 0; m < 3; m ++) {
								S_tmxp[3*c1+c2] += ExtractWeyl(pp * S_txp[3*c1+m] * gs * S_tmx[3*m+c2] * pp);
							}
							S_tmxp_T[3*c1+c2] = S_tmxp[3*c1+c2].transpose();
							S_tmxp_CG[3*c1+c2] = S_tmxp[3*c1+c2] * cgs;
							S_tmxp_T_CG[3*c1+c2] = S_tmxp_T[3*c1+c2] * cgs;
						}
					}
          // include precomputed pairs and triples
          #include "autogen/run_nnpp_3pt_1.inc"

          // loop over colors in epsilon tensors
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
            // compute the necessary contractions
            #include "autogen/run_nnpp_3pt_2.inc"
          } // colors
          #include "autogen/run_nnpp_3pt_3.inc"
          Vcomplex sum = Vcomplex();
          for (int i = 0; i < 288; i ++) sum += tmp[i];
          #pragma omp critical
          corr[(sep * nt + t) * nt + gamma_index] += sum;
        }
      }
    } // z, y, x
  } // t
}
