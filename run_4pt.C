#include "run_4pt.h"
#include <stdlib.h>
#include <string.h>
#include <mkl.h>

/*
 * This file is the analog of run_3pt.C but for the 4-point functions.
 * In particular, it contains routines to create sequential propagators
 * and to combine them into the four-index tensor T that will be used
 * in constructing the 4-point functions.
 * It also contains the functions to construct the neutrino propagator.
 */

// compute sequential propagator through one insertion time
void assemble_Hvec(WeylMat * Hvec,           // sequential propagator
                   SpinMat * wall_prop,      // source to operator
                   SpinMat * point_prop,     // sink to operator
                   int nx,                   // spatial extent of lattice
                   int block_size,           // sparsening factor at operator
                   int tm,                   // source time
                   int tp,                   // sink time
                   int ty) {                 // operator time
  int nx_blocked = nx / block_size;
  SpinMat gmu [4] = {gx, gy, gz, gt};
  SpinMat gmu_pl [4];
  for (int mu = 0; mu < 4; mu ++) gmu_pl[mu] = gmu[mu] * pl;
  for (int z = 0; z < nx; z += block_size) {
    for (int y = 0; y < nx; y += block_size) {
      for (int x = 0; x < nx; x += block_size) {
        int idx = ((z / block_size) * nx_blocked 
            + (y / block_size)) * nx_blocked 
            + (x / block_size);
        int loc = ((ty * nx + z) * nx + y) * nx + x;
        SpinMat * Sl_xz = wall_prop + 9 * loc;
        SpinMat * Sl_wz = point_prop + 9 * loc;
        for (int mu = 0; mu < 4; mu ++) { 
          for (int c = 0; c < 9; c ++) 
            Hvec[(4 * idx + mu) * 9 + c] = WeylMat();
          for (int c1 = 0; c1 < 3; c1 ++) {
            for (int c3 = 0; c3 < 3; c3 ++) {
              // As with the 3-point functions, we don't need the full SpinMat
              // but only the 2x2 upper left-hand corner.
              // Here, the multiplication is written explicitly
              // in terms of the indices used in the operation.
              SpinMat temp = SpinMat();
              // manual multiply to get temp
              // s1 only needs to go from 0 to 1
              for (int s1 = 0; s1 < 2; s1 ++)
                for (int s2 = 0; s2 < 4; s2 ++)
                  for (int s3 = 0; s3 < 4; s3 ++)
                    temp.data[s1*4+s3] += Sl_wz[3*c1+c3].data[s1*4+s2] * gmu_pl[mu].data[s2*4+s3];
              for (int c2 = 0; c2 < 3; c2 ++) {
                // manual multiply to get final result
                // s1, s3 only need to go from 0 to 1
                Vcomplex result [4];
                for (int s1 = 0; s1 < 2; s1 ++)
                  for (int s2 = 0; s2 < 4; s2 ++)
                    for (int s3 = 0; s3 < 2; s3 ++)
                      result[s1*2+s3] += temp.data[s1*4+s2] * Sl_xz[3*c3+c2].data[s2*4+s3];
                WeylMat result_mat = *(WeylMat*) result;
                result_mat = result_mat * 2;
                Hvec[((idx * 4 + mu) * 3 + c1) * 3 + c2] += result_mat;
                // The net result of this is equivalent to
                // Hvec[((idx * 4 + mu) * 3 + c1) * 3 + c2]
                //    += ExtractWeyl(temp * Sl_xz[3*c3+c2]);
                // but it avoids computing the unnecessary entries
              }
            }
          }
        }
      }
    }
  }
}

// Take the norm of the vector in a periodic box.
// Since one can measure a vector length going around the box
// in either direction, we should take the minimum of these distances.
#define MIN(a, b) ( (a<b) ? a : b )
double periodic_norm(int x, int y, int z, int L) {
  x = MIN(x, L-x);
  y = MIN(y, L-y);
  z = MIN(z, L-z);
  return sqrt(x * x + y * y + z * z);
}

// The finite-volume, zero-mode subtracted neutrino propagator from
// Eq. (13) of https://arxiv.org/pdf/2012.02083.pdf.
// In the paper, the momentum sum is infinite and thus singular in the UV.
// We will regulate the sum by capping momenta at pi/a in each direction.
// This means that the momentum will be restricted to the first Brillouin zone
// of the lattice.
double nu_prop(int x, int y, int z, int tau,   // separation between currents
                      int nx,                  // spatial extent
                      int global_sparsening) { // global sparsening factor
  int nx_full = nx * global_sparsening;
  double denom = 4 * M_PI * nx_full * nx_full;
  double sum = 0;
  // loop over momentum
  for (int qz = 0; qz < nx_full; qz ++) {
    for (int qy = 0; qy < nx_full; qy ++) {
      for (int qx = 0; qx < nx_full; qx ++) {
        // compute norm of momentum
        double normq = periodic_norm(qx, qy, qz, nx_full);
        // remove zero mode
        if (normq == 0) continue;

        double phase = z * qz + y * qy + x * qx;
        phase *= 2 * M_PI / nx;

        // add contribution to total
        sum += exp(-normq * tau * 2 * M_PI / (nx_full))
               * cos(phase) / (denom * normq);
      }
    }
  }
  return sum;
}

// Given a set of color and spin indices, 
// return the overall index into the tensor T.
// Color runs slower than spin here.
int tensor_index(int c1, int c2, int c3, int c4,
                 int s1, int s2, int s3, int s4) {
  return ((((((c1 * 3 + c2) * 3 + c3) * 3 + c4) * 2 + s1) * 2 + s2) * 2 + s3) * 2 + s4;
}

// This is the main method to compute the rank-4 tensor T.
// The inputs are the Fourier-transformed sequential quark propagators
// and the Fourier-transformed neutrino propagator.
// The quark propagators carry spin-color indices
// and also include a loop over the gamma matrix gamma[mu] at the operator.
void compute_tensor(Vcomplex * T,             // tensor to be computed
                    WeylMat * Hvec_x_F,       // quark propagator through x
                    WeylMat * Hvec_y_F,       // quark propagator through y
                    fftw_complex * nu_F,      // neutrino propagator
                    int nx, int nt,           // size of lattice
                    int block_size,           // sparsening at operator
                    int global_sparsening) {  // global sparsening factor
  // spatial volume of lattice
  int nx_blocked = nx / block_size;
  int volume = nx_blocked * nx_blocked * nx_blocked;

  // zero out T
  for (int i = 0; i < 1296; i ++)
    T[i] = Vcomplex();

  // Precompute Hvec_x times neutrino propagator
  // Also pull out only gamma indices mu=0 and mu=2
  // since these are already linear combinations containing other terms
  WeylMat * Hvec_x_nu = (WeylMat*) malloc(volume * 2 * 9 * sizeof(WeylMat));
  WeylMat * Hvec_y_packed = (WeylMat*) malloc(volume * 2 * 9 * sizeof(WeylMat));
  for (int idp = 0; idp < volume; idp ++) {
    double nu_value = *(double*) (nu_F + idp);
    for (int mu = 0; mu < 2; mu ++) {
      for (int c = 0; c < 9; c ++) {
        Hvec_x_nu[(idp * 2 + mu) * 9 + c] = Hvec_x_F[(idp * 4 + mu * 2) * 9 + c] * nu_value;
        Hvec_y_packed[(idp * 2 + mu) * 9 + c] = Hvec_y_F[(idp * 4 + mu * 2) * 9 + c];
      }
    }
  }

  // Integrate over triple product Hvec_x * Hvec_y * nu in momentum space.
  // The tensors Hvec_x_nu and Hvec_y_packed are of dimension (volume * mu)
  // by (spin-color)^2.  We want to sum over volume and mu and leave the
  // spin-color pairs as free indices in the final tensor.
  // This corresponds to a product of two rectangular matrices
  // (with the volume/mu indices as the inner ones being contracted),
  // so we can write this as a matrix product and call MKL BLAS.
  Vcomplex alpha (1,0);
  Vcomplex beta (0,0);
  int m = 36, k = volume * 2, n = 36;
  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k,
      &alpha, Hvec_x_nu, m, Hvec_y_packed, n, &beta, T, n);
  
  free(Hvec_x_nu);
  free(Hvec_y_packed);

  // The Fourier transforms introduce an extra factor of volume,
  // so we divide this out here
  for (int i = 0; i < 1296; i ++)
    T[i] *= (1.0 / volume);

  // Reorder T so that all color indices run slower than all sink indices
  // This will allow easier indexing when computing contractions
  Vcomplex * buffer = (Vcomplex *) malloc(4 * 4 * 9 * sizeof(Vcomplex));
  for (int c1 = 0; c1 < 9; c1 ++) {
    memcpy(buffer, T, 4 * 4 * 9 * sizeof(Vcomplex));
    for (int c = 0; c < 9; c ++) {
      for (int s = 0; s < 4; s ++) {
        memcpy(T + (c * 4 + s) * 4, buffer + (s * 9 + c) * 4, 4 * sizeof(Vcomplex));
      }
    }
    T += 4 * 4 * 9;
  }
  free(buffer);
}
