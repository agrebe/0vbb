#include "read_prop.h"

void read_prop(char * filename, SpinMat * prop, int nt, int nx) {
  FILE * file = fopen(filename, "rb");
  int vol = nt * nx * nx * nx;
  uint64_t * buffer = (uint64_t*) malloc(288 * vol * sizeof(uint64_t));
  fread(buffer, sizeof(uint64_t), 288 * vol, file);
  #pragma omp parallel for
  for (int i = 0; i < 288 * vol; i ++)
    buffer[i] = _bswap64(buffer[i]);
  #pragma omp parallel for
  for (int loc = 0; loc < vol; loc ++) {
    SpinMat * wilsonMat = prop + 9*loc;
    for (int s1 = 0; s1 < 4; s1 ++) {
      for (int s2 = 0; s2 < 4; s2 ++) {
        for (int c1 = 0; c1 < 3; c1 ++) {
          for (int c2 = 0; c2 < 3; c2 ++) {
            wilsonMat[3*c1 + c2].data[4*s1 + s2] = *(Vcomplex*) (buffer + 288 * loc + (((4 * s1 + s2) * 3 + c1) * 3 + c2) * 2);
          }
        }
      }
    }
  }
  free (buffer);
}

void rescale_prop(SpinMat * prop, int nt, int nx, double factor) {
  int vol = nt * nx * nx * nx;
  for (int i = 0; i < 9 * vol; i ++)
    prop[i] *= factor;
}

// reverse propagator using gamma5-hermiticity
// this will reverse the propagator in place
// for point props, we only need the reversed version
void reverse_prop(SpinMat * prop, int nt, int nx) {
  int vol = nt * nx * nx * nx;
  size_t mat_size = 288 * sizeof(double);
  #pragma omp parallel for
  for (int loc = 0; loc < vol; loc ++) {
    SpinMat * buffer = (SpinMat *) malloc(mat_size);
    SpinMat * wilsonMat = prop + 9 * loc;
    memcpy(buffer, wilsonMat, mat_size);
    for (int c1 = 0; c1 < 3; c1 ++)
      for (int c2 = 0; c2 < 3; c2 ++)
        wilsonMat[3*c1+c2] = g5 * buffer[3*c2+c1].hconj() * g5;
    free(buffer);
  }
}

// parity project a propagator at its source
// positive is binary switch to determine
// if (positive) use pp; else use pm
// source is defined based on point inverted off of
// so "source" is tm or tp but not operator time
void project_prop(SpinMat * prop, int nt, int nx, int positive) {
  int vol = nt * nx * nx * nx;
  SpinMat proj;
  if (positive) proj = pp; else proj = pm;
  #pragma omp parallel for
  for (int loc = 0; loc < vol; loc ++)
    for (int c = 0; c < 9; c ++)
      prop[loc * 9 + c] = prop[loc * 9 + c] * proj;
}
