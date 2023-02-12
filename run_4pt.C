#include "run_4pt.h"
#include <stdlib.h>
#include <string.h>
#include <mkl.h>

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
  #pragma omp parallel for collapse(3)
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
              //SpinMat temp = Sl_wz[3*c1+c3] * gmu_pl[mu];
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
                /*
                Hvec[((idx * 4 + mu) * 3 + c1) * 3 + c2]
                    += ExtractWeyl(temp * Sl_xz[3*c3+c2]);
                */
              }
            }
          }
        }
      }
    }
  }
}

static int dist_sq(int y, int z, int nx) {
  int d1 = abs(y - z); 
  d1 = (d1 < nx - d1) ? d1 : nx - d1; 
  return d1 * d1; 
}

// neutrino propagator
double nu_prop(int y1, int y2, int y3, int y4, // first point
                      int z1, int z2, int z3, int z4, // second point
                      int nx, int nt,                 // spatial and temporal extent
                      int global_sparsening) {        // global sparsening factor
  int distance_sq;
  distance_sq = dist_sq(y1, z1, nx);
  distance_sq += dist_sq(y2, z2, nx);
  distance_sq += dist_sq(y3, z3, nx);
  // inflate spatial coordinates by global sparsening factor
  distance_sq *= (global_sparsening * global_sparsening);

  distance_sq += dist_sq(y4, z4, nt);

  double prop = (distance_sq == 0) ? 1.0/16 : (1 - exp(-distance_sq * M_PI * M_PI / 4)) / (4 * M_PI * M_PI * distance_sq);
  return prop;
}

// return tensor index from colors and spins
// color runs slower than spin
int tensor_index(int c1, int c2, int c3, int c4,
                 int s1, int s2, int s3, int s4) {
  //return ((((((c1 * 3 + c2) * 3 + c3) * 3 + c4) * 2 + s1) * 2 + s2) * 2 + s3) * 2 + s4;
  return ((((((c1 * 3 + c2) * 2 + s1) * 2 + s2) * 3 + c3) * 3 + c4) * 2 + s3) * 2 + s4;
}

// compute rank-4 tensor T
void compute_tensor(Vcomplex * T,
                    WeylMat * Hvec_x_F,
                    WeylMat * Hvec_y_F,
                    fftw_complex * nu_F,
                    int nx, int nt,
                    int block_size,
                    int global_sparsening) {
  int nx_blocked = nx / block_size;

  // Fourier transform Hvec
  int volume = nx_blocked * nx_blocked * nx_blocked;
  int num_ffts = 4 * 9 * 4; // mu, color^2, spin^2
  const int dims [3] = {nx_blocked, nx_blocked, nx_blocked};

  // zero out T
  for (int i = 0; i < 1296; i ++)
    T[i] = Vcomplex();

  // precompute Hvec_x times neutrino propagator
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

  // integrate over triple product Hvec_x * Hvec_y * nu in momentum space
  // We can write this as a matrix product and call MKL BLAS
  Vcomplex alpha (1,0);
  Vcomplex beta (0,0);
  int m = 36, k = volume * 2, n = 36;
  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k,
      &alpha, Hvec_x_nu, m, Hvec_y_packed, n, &beta, T, n);
  
  free(Hvec_x_nu);
  free(Hvec_y_packed);

  // somehow we picked up a factor of V with all the FFTs
  for (int i = 0; i < 1296; i ++)
    T[i] *= (1.0 / volume);

  return;
}
