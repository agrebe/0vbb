#include "run_4pt.h"

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
        SpinMat gmu;
        for (int mu = 0; mu < 4; mu ++) { 
          if      (mu == 0) { gmu = gx; }
          else if (mu == 1) { gmu = gy; }
          else if (mu == 2) { gmu = gz; }
          else if (mu == 3) { gmu = gt; }
          for (int c = 0; c < 9; c ++) 
            Hvec[(4 * idx + mu) * 9 + c] = WeylMat();
          for (int c1 = 0; c1 < 3; c1 ++)
            for (int c2 = 0; c2 < 3; c2 ++)
              for (int c3 = 0; c3 < 3; c3 ++)
                Hvec[((idx * 4 + mu) * 3 + c1) * 3 + c2]
                    += ExtractWeyl(pp * g5 * Sl_wz[3*c3+c1].hconj() * gmu * pl * Sl_xz[3*c3+c2] * pp);
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
static double nu_prop(int y1, int y2, int y3, int y4, // first point
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

// convolve sequential quark propagator with neutrino
void compute_SnuHz(WeylMat * SnuHz,         // seqprop * nu_prop
                   WeylMat * Hvec,          // seqprop
                   int tx, int ty,          // operator times
                   int nx, int nt,          // spatial and temporal extent
                   int block_size,          // sparsening factor at operator
                   int global_sparsening) { // global sparsening factor
  int nx_blocked = nx / block_size;
  // loop over first operator insertion position
  #pragma omp parallel for collapse(3)
  for (int y3 = 0; y3 < nx; y3 += block_size) {
    for (int y2 = 0; y2 < nx; y2 += block_size) {
      for (int y1 = 0; y1 < nx; y1 += block_size) {
        int idy = ((y3 / block_size) * nx_blocked 
            + (y2 / block_size)) * nx_blocked 
            + (y1 / block_size);
        // zero out SnuHz
        for (int mu = 0; mu < 4; mu ++) { 
          for (int c = 0; c < 9; c ++) 
            SnuHz[(4 * idy + mu) * 9 + c] = WeylMat();
        }

        // loop over second operator insertion position
        for (int x3 = 0; x3 < nx; x3 += block_size) {
          for (int x2 = 0; x2 < nx; x2 += block_size) {
            for (int x1 = 0; x1 < nx; x1 += block_size) {
              int idx = ((x3 / block_size) * nx_blocked 
                  + (x2 / block_size)) * nx_blocked 
                  + (x1 / block_size);
              double nu_value = nu_prop(y1, y2, y3, ty,
                                        x1, x2, x3, tx,
                                        nx, nt, global_sparsening);
              __m512d nu_value_vec = _mm512_set1_pd(nu_value);
              for (int c = 0; c < 9; c ++) {
                for (int mu = 0; mu < 4; mu ++) {
                  SnuHz[(4*idy+mu)*9+c].data = _mm512_fmadd_pd(
                      Hvec[(4*idx+mu)*9+c].data,
                      nu_value_vec,
                      SnuHz[(4*idy+mu)*9+c].data);
                }
                /*
                 * original (unoptimized) algorithm
                 * add Hvec * pl * nu_value to SnuHz
                SnuHz[(4*idy+0)*9+c] += Hvec[(4*idx+0)*9+c] * nu_value; 
                SnuHz[(4*idy+1)*9+c] += Hvec[(4*idx+1)*9+c] * nu_value; 
                SnuHz[(4*idy+2)*9+c] += Hvec[(4*idx+2)*9+c] * nu_value; 
                SnuHz[(4*idy+3)*9+c] += Hvec[(4*idx+3)*9+c] * nu_value; 
                SnuHz[(4*idy+0)*9+c] += Hvec[(4*idx+1)*9+c] * nu_value * Vcomplex(0,-1);
                SnuHz[(4*idy+1)*9+c] += Hvec[(4*idx+0)*9+c] * nu_value * Vcomplex(0,1);
                SnuHz[(4*idy+2)*9+c] += Hvec[(4*idx+3)*9+c] * nu_value * Vcomplex(0,1);
                SnuHz[(4*idy+3)*9+c] += Hvec[(4*idx+2)*9+c] * nu_value * Vcomplex(0,-1);
                */
              }
            }
          }
        }
        // take appropriate linear combinations of matrices
        // this encodes the spin projection to put the electrons in opposite spins
        for (int c = 0; c < 9; c ++) {
          SnuHz[(4*idy+0)*9+c] += SnuHz[(4*idy+1)*9+c] * Vcomplex(0,-1);
          SnuHz[(4*idy+1)*9+c]  = SnuHz[(4*idy+0)*9+c] * Vcomplex(0, 1);
          SnuHz[(4*idy+2)*9+c] += SnuHz[(4*idy+3)*9+c] * Vcomplex(0, 1);
          SnuHz[(4*idy+3)*9+c]  = SnuHz[(4*idy+2)*9+c] * Vcomplex(0,-1);
        }
      }
    }
  }
}
