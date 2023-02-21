#include "run_3pt.h"

// compute rank-4 tensor T
// Here T will have an extra index of {SS, PP, VV, AA, TT}
// This index will run slower than anything else
// so T will be of size (2*3)^4 * 4
void compute_tensor_3(Vcomplex * T,
                      SpinMat * wall_prop,  // source to operator
                      SpinMat * point_prop, // sink to operator
                      int nx,               // spatial extent of lattice
                      int block_size,       // sparsening factor at operator
                      int ty) {             // operator time
  int num_currents = 5; // {SS, PP, VV, AA, TT}

  // zero out T
  for (int i = 0; i < 1296 * num_currents; i ++)
    T[i] = Vcomplex();

  int nx_blocked = nx / block_size;
  SpinMat gs [16] = {id, g5, gx, gy, gz, gt,
                     g5 * gx, g5 * gy, g5 * gz, g5 * gt,
                     gx * gy, gx * gz, gx * gt,
                     gy * gz, gy * gt, gz * gt};
  // split each SpinMat into four WeylMats
  WeylMat gs_W [64];
  for (int i = 0; i < 16; i ++) {
    gs_W[4*i] = ExtractWeyl(gs[i], 0, 0);
    gs_W[4*i+1] = ExtractWeyl(gs[i], 0, 1);
    gs_W[4*i+2] = ExtractWeyl(gs[i], 1, 0);
    gs_W[4*i+3] = ExtractWeyl(gs[i], 1, 1);
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
        WeylMat Sl_wz_W[9*2], Sl_xz_W[9*2];
        for (int c = 0; c < 9; c ++) {
          Sl_wz_W[c*2]   = ExtractWeyl(Sl_wz[c], 0, 0);
          Sl_wz_W[c*2+1] = ExtractWeyl(Sl_wz[c], 0, 1);
          Sl_xz_W[c*2]   = ExtractWeyl(Sl_xz[c], 0, 0);
          Sl_xz_W[c*2+1] = ExtractWeyl(Sl_xz[c], 1, 0);
        }
        for (int c1 = 0; c1 < 3; c1 ++) {
          for (int cA = 0; cA < 3; cA ++) {
            for (int gamma_index = 0; gamma_index < 16; gamma_index ++) {
              // compute the block WeylMats in the upper row of Sl_wz * gs
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
                // multiply upper row of temp = Sl_wz * gs
                // by left column of Sl_xz
                WeylMat temp2 = Sl_xz_W[2*(3*cA+c2)];
                WeylMat temp3 = Sl_xz_W[2*(3*cA+c2)+1];
                seqprop[gamma_index * 9 + ((c1 * 3) + c2)]
                  += (temp0 * temp2 + temp1 * temp3) * 2;
              }

            }
          }
        }
        WeylMat * seqprop_shifted;
        for (int gamma_index = 0; gamma_index < 16; gamma_index ++) {
          seqprop_shifted = seqprop + gamma_index * 9;
          Vcomplex * T_shifted;
          int shift;
          // increment T at start of PP, VV, AA, and TT
          if (gamma_index >= 10) shift = 4;
          else if (gamma_index >= 6) shift = 3;
          else if (gamma_index >= 2) shift = 2;
          else if (gamma_index >= 1) shift = 1;
          else shift = 0;
          shift *= 1296;
          T_shifted = T + shift;
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
                            += ((Vcomplex*) (seqprop_shifted + c1 * 3 + c2))[s1*2+s2]
                             * ((Vcomplex*) (seqprop_shifted + c3 * 3 + c4))[s3*2+s4];
                        }
                      }
                    }
                  }

                }
              }
            }
          }
        }

      }
    }
  }
  return;
}
