#include "run_meson_2pt.h"
#include "gamma_container.h"

// point sink
void run_pion_correlator(SpinMat * prop, Vcomplex * corr, int nt, int nx) {
  for (int t = 0; t < nt; t ++) corr[t] = Vcomplex();
  for (int t = 0; t < nt; t ++) {
    for (int i = 0; i < nx * nx * nx; i ++) {
      for (int c1 = 0; c1 < 3; c1 ++) {
        for (int c2 = 0; c2 < 3; c2 ++) {
          SpinMat mat = prop[(((t * nx * nx * nx) + i) * 3 + c1) * 3 + c2];
          corr[t] += Trace(mat * mat.hconj());
        }
      }
    }
  }
}

// wall sink
void run_pion_correlator_wsink(SpinMat * prop, Vcomplex * corr, int nt, int nx) {
  for (int t = 0; t < nt; t ++) corr[t] = Vcomplex();
  for (int t = 0; t < nt; t ++) {
    SpinMat sink [9];
    for (int i = 0; i < nx * nx * nx; i ++) {
      for (int c = 0; c < 9; c ++) {
        sink[c] = sink[c] + prop[((t * nx * nx * nx) + i) * 9 + c];
      }
    }
    for (int c = 0; c < 9; c ++)
      corr[t] += Trace(sink[c] * sink[c].hconj());
  }
}
