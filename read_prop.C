#include "read_prop.h"

void read_prop(char * filename, SpinMat * prop, int nt, int nx) {
  FILE * file = fopen(filename, "rb");
  int vol = nt * nx * nx * nx;
  for (int loc = 0; loc < vol; loc ++) {
    uint64_t * buffer = (uint64_t*) malloc(288 * sizeof(uint64_t));
    fread(buffer, sizeof(uint64_t), 288, file);
    for (int i = 0; i < 288; i ++)
      buffer[i] = _bswap64(buffer[i]);
    SpinMat * wilsonMat = prop + 9*loc;
    for (int s1 = 0; s1 < 4; s1 ++) {
      for (int s2 = 0; s2 < 4; s2 ++) {
        for (int c1 = 0; c1 < 3; c1 ++) {
          for (int c2 = 0; c2 < 3; c2 ++) {
            wilsonMat[3*c1 + c2].data[4*s1 + s2] = *(Vcomplex*) (buffer + (((4 * s1 + s2) * 3 + c1) * 3 + c2) * 2);
          }
        }
      }
    }
    free (buffer);
  }
}
