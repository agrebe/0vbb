#include "read_prop.h"

/*
 * This file contains routines to read propagators from disk and pre-process them.
 */

// This method reads a propagator stored from chroma and packages it into SpinMat form.
// The chroma convention is for color indices to run fastest in propagators.
// For the contraction code in this project, however, we need spin to run fastest.
// Note: This is not used in the current iteration of the code
// since QPhiX is called directly, but it is useful for testing or
// if one needs to read in propagators that have already been computed.
void read_prop(char * filename, SpinMat * prop, int nt, int nx) {
  FILE * file = fopen(filename, "rb");
  int vol = nt * nx * nx * nx;
  // read the propagator into a buffer
  uint64_t * buffer = (uint64_t*) malloc(288 * vol * sizeof(uint64_t));
  fread(buffer, sizeof(uint64_t), 288 * vol, file);

  // flip the endianness of the propagator
  #pragma omp parallel for
  for (int i = 0; i < 288 * vol; i ++)
    buffer[i] = _bswap64(buffer[i]);

  // package the propagator into SpinMat form
  #pragma omp parallel for
  for (int loc = 0; loc < vol; loc ++) {
    SpinMat * wilsonMat = prop + 9*loc;
    for (int s1 = 0; s1 < 4; s1 ++) {
      for (int s2 = 0; s2 < 4; s2 ++) {
        for (int c1 = 0; c1 < 3; c1 ++) {
          for (int c2 = 0; c2 < 3; c2 ++) {
            wilsonMat[3*c1 + c2].data[4*s1 + s2] 
              = *(Vcomplex*) (buffer + 288 * loc + (((4 * s1 + s2) * 3 + c1) * 3 + c2) * 2);
          }
        }
      }
    }
  }
  free (buffer);
}

// This routine multiplies a propagator by a constant.
// Note that this is also not currently used, but it can be used to
// correct for spurious normalization factors introduced by parity projection.
void rescale_prop(SpinMat * prop, int nt, int nx, double factor) {
  int vol = nt * nx * nx * nx;
  for (int i = 0; i < 9 * vol; i ++)
    prop[i] *= factor;
}

// This routine time-reverses the propagator using gamma5-hermiticity.
// This will reverse the propagator in place.
// For point props, we only need the reversed version.
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

// This routine parity projects a propagator at its source.
// The argument "positive" is a binary switch to determine which parity.
// if (positive) use pp; else use pm
// The source is defined based on point inverted off of,
// so "source" is tm or tp but not the operator time.
void project_prop(SpinMat * prop, int nt, int nx, int positive) {
  int vol = nt * nx * nx * nx;
  SpinMat proj;
  if (positive) proj = pp; else proj = pm;
  #pragma omp parallel for
  for (int loc = 0; loc < vol; loc ++)
    for (int c = 0; c < 9; c ++)
      prop[loc * 9 + c] = prop[loc * 9 + c] * proj;
}
