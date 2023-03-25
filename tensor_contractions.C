#include "tensor_contractions.h"
#include "run_4pt.h"

/*
 * This file contains routines to perform spin traces against the precomputed
 * tensor T.  Each routine takes specific color indices and pulls out the
 * necessary spin components of the tensor and contracts them with other
 * spin matrices.
 * 
 * Since T has four spin indices and a spin matrix has two free spin indices,
 * we will need to contract two spin matrices with T to obtain a spin singlet.
 *
 * There are three patterns for this contraction that appear in the code.
 * In the following routines, T is thought of as two seqprops, S_zxw and S_zyw
 * through the operator insertion points x and y, and the spin indices of T
 * are thought of as the ones coming from these two seqprops
 */

// compute the tensor contraction
// Tr[S_zxw A] Tr[S_zyw B]
// where the two sequential propagators are encoded in T
Vcomplex two_traces(Vcomplex * T,
                    int c1, int c2, int c3, int c4,
                    WeylMat A, WeylMat B) {
  T += tensor_index(c1, c2, c3, c4, 0, 0, 0, 0);
  Vcomplex sum = Vcomplex();
  // loop over spins
  for (int s1 = 0; s1 < 2; s1 ++)
    for (int s2 = 0; s2 < 2; s2 ++)
      for (int s3 = 0; s3 < 2; s3 ++)
        for (int s4 = 0; s4 < 2; s4 ++) {
          Vcomplex tensor_value = *(T++);
          sum += tensor_value * ((Vcomplex *) &A)[s2*2+s1] 
                              * ((Vcomplex *) &B)[s4*2+s3];
        }
  return sum;
}

// compute the tensor contraction
// Tr[S_zxw A S_zyw B]
Vcomplex one_trace(Vcomplex * T,
                   int c1, int c2, int c3, int c4,
                   WeylMat A, WeylMat B) {
  T += tensor_index(c1, c2, c3, c4, 0, 0, 0, 0);
  Vcomplex sum = Vcomplex();
  for (int s1 = 0; s1 < 2; s1 ++)
    for (int s2 = 0; s2 < 2; s2 ++)
      for (int s3 = 0; s3 < 2; s3 ++)
        for (int s4 = 0; s4 < 2; s4 ++) {
          Vcomplex tensor_value = *(T++);
          sum += tensor_value * ((Vcomplex *) &A)[s2*2+s3] 
                              * ((Vcomplex *) &B)[s4*2+s1];
        }
  return sum;
}

// compute the tensor contraction
// Tr[S_zxw A S_zyw^T B]
Vcomplex one_trace_transposed(Vcomplex * T,
                              int c1, int c2, int c3, int c4,
                              WeylMat A, WeylMat B) {
  T += tensor_index(c1, c2, c3, c4, 0, 0, 0, 0);
  Vcomplex sum = Vcomplex();
  for (int s1 = 0; s1 < 2; s1 ++)
    for (int s2 = 0; s2 < 2; s2 ++)
      for (int s3 = 0; s3 < 2; s3 ++)
        for (int s4 = 0; s4 < 2; s4 ++) {
          Vcomplex tensor_value = *(T++);
          sum += tensor_value * ((Vcomplex *) &A)[s2*2+s4] 
                              * ((Vcomplex *) &B)[s3*2+s1];
        }
  return sum;
}
