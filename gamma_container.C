#include "gamma_container.h"

SpinMat gx, gy, gz, gt, g5, pp, pm, cg5;

void initialize_gammas() {
  gx.data[3] = Vcomplex(0, -1);
  gx.data[6] = Vcomplex(0, -1);
  gx.data[9] = Vcomplex(0, 1);
  gx.data[12] = Vcomplex(0, 1);

  gy.data[3] = Vcomplex(-1, 0);
  gy.data[6] = Vcomplex(1, 0);
  gy.data[9] = Vcomplex(1, 0);
  gy.data[12] = Vcomplex(-1, 0);

  gz.data[2] = Vcomplex(0, -1);
  gz.data[7] = Vcomplex(0, 1);
  gz.data[8] = Vcomplex(0, 1);
  gz.data[13] = Vcomplex(0, -1);

  gt.data[2] = Vcomplex(1, 0);
  gt.data[7] = Vcomplex(1, 0);
  gt.data[8] = Vcomplex(1, 0);
  gt.data[13] = Vcomplex(1, 0);

  g5.data[0] = Vcomplex(1, 0);
  g5.data[5] = Vcomplex(1, 0);
  g5.data[10] = Vcomplex(-1, 0);
  g5.data[15] = Vcomplex(-1, 0);

  cg5.data[1] = Vcomplex(0, -1);
  cg5.data[4] = Vcomplex(0, 1);
  cg5.data[11] = Vcomplex(0, -1);
  cg5.data[14] = Vcomplex(0, 1);

  for (int i = 0; i < 4; i ++)
    for (int j = 0; j < 4; j ++) 
      pp.data[4*i+j] = Vcomplex(0.5 * ((i+j+1) % 2), 0);
  
  pm = g5 * pp * g5;
}
