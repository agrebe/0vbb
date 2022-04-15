#include "run_baryon_2pt.h"
#include "gamma_container.h"
#include "color_tensor.h"

// 2-point correlator for a nucleon
void run_neutron_correlator(SpinMat * prop, Vcomplex * corr, int nt, int nx, int block_size) {
  for (int t = 0; t < nt; t ++) corr[t] = Vcomplex();
#pragma omp parallel for
  for (int t = 0; t < nt; t ++) {
    for (int z = 0; z < nx; z += block_size) {
      for (int y = 0; y < nx; y += block_size) {
        for (int x = 0; x < nx; x += block_size) {
          int loc = ((t * nx + z) * nx + y) * nx + x;
          SpinMat * wilsonMat = prop + 9*loc;
          for(int ii=0; ii<36; ii++) 
          {
            const int& a      = color_idx_1[7*ii+0];
            const int& b      = color_idx_1[7*ii+1];
            const int& c      = color_idx_1[7*ii+2];
            const int& d      = color_idx_1[7*ii+3];
            const int& e      = color_idx_1[7*ii+4];
            const int& f      = color_idx_1[7*ii+5];
            const double sign = color_idx_1[7*ii+6];
            corr[t] -= sign * Trace(cg5 * wilsonMat[3*e+b] * cg5 * wilsonMat[3*d+a].transpose()) * Trace(pp * wilsonMat[3*f+c]);
            corr[t] += sign * Trace(cg5 * wilsonMat[3*e+b] * cg5 * wilsonMat[3*f+a].transpose() * pp * wilsonMat[3*d+c].transpose());
          }
        }
      }
    }
  }
}

// 2-point nucleon correlator with all quarks projected to positive parity
void run_neutron_correlator_PP(SpinMat * prop, Vcomplex * corr, int nt, int nx, int block_size) {
  for (int t = 0; t < nt; t ++) corr[t] = Vcomplex();
#pragma omp parallel for
  for (int t = 0; t < nt; t ++) {
    for (int z = 0; z < nx; z += block_size) {
      for (int y = 0; y < nx; y += block_size) {
        for (int x = 0; x < nx; x += block_size) {
          int loc = ((t * nx + z) * nx + y) * nx + x;
          SpinMat * wilsonMat = prop + 9*loc;
          for(int ii=0; ii<36; ii++) 
          {
            const int& a      = color_idx_1[7*ii+0];
            const int& b      = color_idx_1[7*ii+1];
            const int& c      = color_idx_1[7*ii+2];
            const int& d      = color_idx_1[7*ii+3];
            const int& e      = color_idx_1[7*ii+4];
            const int& f      = color_idx_1[7*ii+5];
            const double sign = color_idx_1[7*ii+6];
            corr[t] -= sign * Trace(pp * cg5 * wilsonMat[3*e+b] * pp * cg5 * wilsonMat[3*d+a].transpose()) * Trace(pp * wilsonMat[3*f+c]);
            corr[t] += sign * Trace(pp * cg5 * wilsonMat[3*e+b] * pp * cg5 * wilsonMat[3*f+a].transpose() * pp * wilsonMat[3*d+c].transpose());
          }
        }
      }
    }
  }
}
