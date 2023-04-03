#include "run_3pt.h"

/*
 * The 3-point functions are computed by first assembling a tensor T
 * with 4 spin-color indices and then contracting this tensor with
 * source/sink interpolators.
 * This file contains the routines needed to build this tensor.
 */

// Given two sequential propagators, compute the contribution to the tensor T.
// T_shift gives the index into the tensor.
// gamma_1 and gamma_2 give the indices of the two gamma matrices
// used in constructing the two sequential propagators.
void combine_seqprops(Vcomplex * T, int T_shift,
                      WeylMat * seqprop,
                      int gamma_1, int gamma_2) {
  int tensor_size = 6 * 6 * 6 * 6; // 4 spin-color indices
  int seqprop_size = 3 * 3; // 2 color indices
  Vcomplex * T_shifted = T + T_shift * tensor_size;
  WeylMat * seqprop_shifted_1 = seqprop + gamma_1 * seqprop_size;
  WeylMat * seqprop_shifted_2 = seqprop + gamma_2 * seqprop_size;
  // loop over all the colors
  for (int c1 = 0; c1 < 3; c1 ++) {
    for (int c2 = 0; c2 < 3; c2 ++) {
      for (int c3 = 0; c3 < 3; c3 ++) {
        for (int c4 = 0; c4 < 3; c4 ++) {
          // loop over spins
          for (int s1 = 0; s1 < 2; s1 ++) {
            for (int s2 = 0; s2 < 2; s2 ++) {
              for (int s3 = 0; s3 < 2; s3 ++) {
                for (int s4 = 0; s4 < 2; s4 ++) {
                  *(T_shifted++)
                    += ((Vcomplex*) (seqprop_shifted_1 + c1 * 3 + c2))[s1*2+s2]
                     * ((Vcomplex*) (seqprop_shifted_2 + c3 * 3 + c4))[s3*2+s4];
                }
              }
            }
          }

        }
      }
    }
  }
}

// This is identical to combine_seqprops above,
// except that it subtracts the product of propagators
// from T instead of adding the product to T.
// This is needed for the AT contractions that require one
// component to have a negative sign relative to the others.
void combine_seqprops_negative(Vcomplex * T, int T_shift,
                      WeylMat * seqprop,
                      int gamma_1, int gamma_2) {
  int tensor_size = 6 * 6 * 6 * 6; // 4 spin-color indices
  int seqprop_size = 3 * 3; // 2 color indices
  Vcomplex * T_shifted = T + T_shift * tensor_size;
  WeylMat * seqprop_shifted_1 = seqprop + gamma_1 * seqprop_size;
  WeylMat * seqprop_shifted_2 = seqprop + gamma_2 * seqprop_size;
  // loop over all the colors
  for (int c1 = 0; c1 < 3; c1 ++) {
    for (int c2 = 0; c2 < 3; c2 ++) {
      for (int c3 = 0; c3 < 3; c3 ++) {
        for (int c4 = 0; c4 < 3; c4 ++) {
          // loop over spins
          for (int s1 = 0; s1 < 2; s1 ++) {
            for (int s2 = 0; s2 < 2; s2 ++) {
              for (int s3 = 0; s3 < 2; s3 ++) {
                for (int s4 = 0; s4 < 2; s4 ++) {
                  *(T_shifted++)
                    -= ((Vcomplex*) (seqprop_shifted_1 + c1 * 3 + c2))[s1*2+s2]
                     * ((Vcomplex*) (seqprop_shifted_2 + c3 * 3 + c4))[s3*2+s4];
                }
              }
            }
          }

        }
      }
    }
  }
}

// This is the main method to construct the 4-index tensor.
// It begins by computing the 16 seqprops with different gamma matrices
// and then it assembles the necessary pairs into the tensor.
// The pairs of seqprops used are those needed to form the various
// operators mentioned in Eq. (7) of https://arxiv.org/pdf/1806.02780.pdf.
// After applying spin and color Fiertz identities, the necessary combinations
// of gamma matrices in seqprop pairs are {SS, PP, VV, AA, TT, VS, AP, VT, AT}.
// The tensor will have an outer index to specify which of these pairs
// was used in its construction and four spin-color indices, with all spin
// indices running faster than all color indices.
// Thus, the total size of T will be 9 * (2*3)^4 complex numbers.
void compute_tensor_3(Vcomplex * T,
                      SpinMat * wall_prop,  // source to operator
                      SpinMat * point_prop, // sink to operator
                      int nx,               // spatial extent of lattice
                      int block_size,       // sparsening factor at operator
                      int ty) {             // operator time
  int num_currents = 9; // {SS, PP, VV, AA, TT, VS, AP, VT, AT}
  int tensor_size = 6 * 6 * 6 * 6; // 4 spin-color indices

  // zero out T
  for (int i = 0; i < tensor_size * num_currents; i ++)
    T[i] = Vcomplex();

  int nx_blocked = nx / block_size;
  SpinMat gs [16] = {id, g5, gx, gy, gz, gt,
                     g5 * gx, g5 * gy, g5 * gz, g5 * gt,
                     gx * gy, gx * gz, gx * gt,
                     gy * gz, gy * gt, gz * gt};
  // split each SpinMat into four WeylMats
  WeylMat gs_W [64];
  for (int i = 0; i < 16; i ++) {
    gs_W[4*i] = ExtractSpecificWeyl(gs[i], 0, 0);
    gs_W[4*i+1] = ExtractSpecificWeyl(gs[i], 0, 1);
    gs_W[4*i+2] = ExtractSpecificWeyl(gs[i], 1, 0);
    gs_W[4*i+3] = ExtractSpecificWeyl(gs[i], 1, 1);
  }
  for (int z = 0; z < nx; z += block_size) {
    for (int y = 0; y < nx; y += block_size) {
      for (int x = 0; x < nx; x += block_size) {
        int loc = ((ty * nx + z) * nx + y) * nx + x;
        SpinMat * Sl_xz = wall_prop + 9 * loc;
        SpinMat * Sl_wz = point_prop + 9 * loc;
        // compute the sequential propagator through idx
        // compute this for all 16 Gamma matrices at idx
        WeylMat seqprop [9 * 16];
        /*
         * We want to compute the upper left component of Sl_wz * gs * Sl_xz
         * In terms of 2x2 block Weyl matrices, we need the first row of Sl_wz,
         * all four blocks of gs, and the first column of Sl_xz.
         * We will extract the necessary blocks.
         */
        WeylMat Sl_wz_W[9*2], Sl_xz_W[9*2];
        for (int c = 0; c < 9; c ++) {
          Sl_wz_W[c*2]   = ExtractSpecificWeyl(Sl_wz[c], 0, 0);
          Sl_wz_W[c*2+1] = ExtractSpecificWeyl(Sl_wz[c], 0, 1);
          Sl_xz_W[c*2]   = ExtractSpecificWeyl(Sl_xz[c], 0, 0);
          Sl_xz_W[c*2+1] = ExtractSpecificWeyl(Sl_xz[c], 1, 0);
        }
        for (int c1 = 0; c1 < 3; c1 ++) {
          for (int cA = 0; cA < 3; cA ++) {
            for (int gamma_index = 0; gamma_index < 16; gamma_index ++) {
              // Compute the block WeylMats in the upper row of Sl_wz * gs
              // This is all we need to extract the upper left corner
              WeylMat temp0 = Sl_wz_W[2*(3*c1+cA)] 
                              * gs_W[4*gamma_index]
                            + Sl_wz_W[2*(3*c1+cA)+1]
                              * gs_W[4*gamma_index+2];
              WeylMat temp1 = Sl_wz_W[2*(3*c1+cA)] 
                              * gs_W[4*gamma_index+1]
                            + Sl_wz_W[2*(3*c1+cA)+1]
                              * gs_W[4*gamma_index+3];
              for (int c2 = 0; c2 < 3; c2 ++) {
                // Multiply upper row of temp = Sl_wz * gs
                // by left column of Sl_xz
                WeylMat temp2 = Sl_xz_W[2*(3*cA+c2)];
                WeylMat temp3 = Sl_xz_W[2*(3*cA+c2)+1];
                seqprop[gamma_index * 9 + ((c1 * 3) + c2)]
                  += (temp0 * temp2 + temp1 * temp3) * 2;
              }

            }
          }
        }
        // Compute the diagonal components of T,
        // corresponding to a pair of identical seqprops
        // These give the terms SS, PP, VV, AA, and TT
        for (int gamma_index = 0; gamma_index < 16; gamma_index ++) {
          Vcomplex * T_shifted;
          int shift;
          // increment T at start of PP, VV, AA, and TT
          if (gamma_index >= 10) shift = 4;
          else if (gamma_index >= 6) shift = 3;
          else if (gamma_index >= 2) shift = 2;
          else if (gamma_index >= 1) shift = 1;
          else shift = 0;
          combine_seqprops(T, shift, seqprop, gamma_index, gamma_index);
        }

        // VS and AP contractions
        // these will multiply the temporal index (5 and 9) of V and A
        // with S and P (indices 0 and 1), respectively

        combine_seqprops(T, 5, seqprop, 0, 5);
        combine_seqprops(T, 6, seqprop, 1, 9);

        // VT contractions
        // These have the form V^{nu} T^{mu nu}
        // We are interested in the mu = 0 component, so this is
        // V^1 T^{01} + V^2 T^{02} + V^3 T^{03}
        // indices of components:
        // V^{1,2,3}: 2, 3, 4
        // T^{01, 02, 03}: 12, 14, 15

        combine_seqprops(T, 7, seqprop, 2, 12);
        combine_seqprops(T, 7, seqprop, 3, 14);
        combine_seqprops(T, 7, seqprop, 4, 15);

        // AT contractions
        // These have the form epsilon^{mu nu rho sigma} A^{nu} T^{rho sigma}
        // We are interested in the mu = 0 component, so this is
        // A^1 T^{23} - A^2 T^{13} + A^3 T^{12}
        // indices of components:
        // A^{1,2,3}: 6, 7, 8
        // T^{23, 13, 12}: 13, 11, 10

        combine_seqprops(T, 8, seqprop, 6, 13);
        // IMPORTANT: This one gets a minus sign
        combine_seqprops_negative(T, 8, seqprop, 7, 11);
        combine_seqprops(T, 8, seqprop, 8, 10);
      }
    }
  }
  return;
}
