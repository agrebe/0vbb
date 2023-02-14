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
  for (int z = 0; z < nx; z += block_size) {
    for (int y = 0; y < nx; y += block_size) {
      for (int x = 0; x < nx; x += block_size) {
        int loc = ((ty * nx + z) * nx + y) * nx + x;
        SpinMat * Sl_xz = wall_prop + 9 * loc;
        SpinMat * Sl_wz = point_prop + 9 * loc;
        // compute the sequential propagator through idx
        // compute this for all 16 Gamma matrices at idx
        WeylMat seqprop [9 * 16];
        for (int c1 = 0; c1 < 3; c1 ++) {
          for (int c2 = 0; c2 < 3; c2 ++) {
            for (int cA = 0; cA < 3; cA ++) {
              for (int gamma_index = 0; gamma_index < 16; gamma_index ++) {
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
                seqprop[gamma_index * 9 + ((c1 * 3) + c2)]
                  += ExtractWeyl(Sl_wz[3*c1+cA] * gs * Sl_xz[3*cA+c2]);
              }

            }
          }
        }
        Vcomplex * T_shifted = T;
        for (int gamma_index = 0; gamma_index < 16; gamma_index ++) {
          // increment T at start of PP, VV, AA, and TT
          if (gamma_index == 1 || gamma_index == 2
              || gamma_index == 6 || gamma_index == 10)
            T_shifted += 1296;
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
                          T_shifted[tensor_index(c1, c2, c3, c4, s1, s2, s3, s4)]
                            += ((Vcomplex*) (seqprop + gamma_index * 9 + c1 * 3 + c2))[s1*2+s2]
                             * ((Vcomplex*) (seqprop + gamma_index * 9 + c3 * 3 + c4))[s3*2+s4];
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
