#ifndef SPIN_MAT_H
#define SPIN_MAT_H

/*
 * This code defines the arithmetic operations for the three structures
 * used most frequently in this code
 * - Vcomplex: a complex double-precision number
 * - SpinMat: a 4x4 complex double-precision matrix with two free spin indices,
 *            each of which is a 4-component spinor
 * - WeylMat: a 2x2 complex double-precision matrix with two free spin indices,
 *            each of which is a 2-component spinor
 * 
 * The arithmetic operations of these structures appear repeatedly,
 * so they have been optimized to use vector registers.  In particular,
 * these assume that the machine has AVX-512 vector intrinsics available
 * (true of Intel Knights Landing and Skylake processors).
 *
 * In the original design of the code, the dominant computational cost
 * was the repeated multiplication of WeylMats inside many nested loops,
 * so the WeylMat multiplication was heavily optimized for the target
 * architecture.  Since a 2x2 matrix of complex double-precision numbers
 * fits exactly in one AVX-512 register, the WeylMat arithmetic operations
 * can be written in terms of AVX-512 intrinsics, leading to a significant
 * performance boost.
 *
 * The code has been redesigned since this file has been written, and some
 * of the optimizations are less important than they were in the original
 * design.  In particular, if this codebase needed to be run on a machine
 * without AVX-512 instructions, WeylMat could probably be rewritten without
 * vectorization (or with only AVX-2 intrinsics) at a relatively mild
 * performance cost.
 *
 * In the choice of variable names, the arithmetic operations are assumed
 * to have C as output and A, B as inputs (e.g. C = A * B).  If one of
 * the inputs is the variable stored in the struct, that variable is
 * referred to as A.  In the comments below, a0, a1, ..., refer to
 * components of A in the vector register.
 */

#include <immintrin.h>

// structure for complex numbers to take advantage of intrinsics
struct Vcomplex {
  // store real in low bits and imag in high bits
  __m128d data;
  Vcomplex() {data = _mm_setzero_pd();}
  Vcomplex(__m128d dat) : data{dat} {}
  Vcomplex(double real, double imag) : data{_mm_setr_pd(real, imag)} {}
  double real() {
    return *((double*) &data);
  }
  double imag() {
    return *((double*) &data + 1);
  }
  Vcomplex operator+(const Vcomplex B) const {
    return Vcomplex(_mm_add_pd(data, B.data));
  }
  Vcomplex operator+=(const Vcomplex B) {
    data = _mm_add_pd(data, B.data);
    return *this;
  }
  Vcomplex operator-(const Vcomplex B) const {
    return Vcomplex(_mm_sub_pd(data, B.data));
  }
  Vcomplex operator-=(const Vcomplex B) {
    data = _mm_sub_pd(data, B.data);
    return *this;
  }
  // complex-complex multiplication
  Vcomplex operator*(const Vcomplex B) const {
    // a1 times (b1, b0)
    __m128d C = _mm_mul_pd(
      _mm_permute_pd(data, 3),
      _mm_permute_pd(B.data, 1));
    // a0 times (b0, b1)
    C = _mm_fmaddsub_pd(
      _mm_permute_pd(data, 0),
      B.data, C);
    return Vcomplex(C);
  }
  Vcomplex operator*=(const Vcomplex B) {
    // a1 times (b1, b0)
    __m128d C = _mm_mul_pd(
      _mm_permute_pd(data, 3),
      _mm_permute_pd(B.data, 1));
    // a0 times (b0, b1)
    data = _mm_fmaddsub_pd(
      _mm_permute_pd(data, 0),
      B.data, C);
    return *this;
  }
  // complex-real multiplication
  Vcomplex operator*(double r) const {
    return Vcomplex(_mm_mul_pd(_mm_set1_pd(r), data));
  }
  Vcomplex operator*=(double r) {
    data = _mm_mul_pd(_mm_set1_pd(r), data);
    return *this;
  }
};

// other direction of real-complex multiplication
static Vcomplex operator*(double r, Vcomplex z) {  
  return z * r;
}

// Structure for 4x4 spin matrices
// Note: Most of the multiplications involve 2x2 matrices
// rather than the full 4x4 matrices.
// As a result, these routines are substantially less optimized
// than the routines for the 2x2 WeylMats below.
// In particular, multiplication of two SpinMats is likely to
// be more than 2^3 = 8x slower than multiplication of two WeylMats.
struct SpinMat {
  SpinMat() {
    for (int i = 0; i < 16; i ++) data[i] = Vcomplex();
  }
  Vcomplex data [16];
  Vcomplex operator()(int a, int b) { return data[4*a + b]; }
  SpinMat operator*(const SpinMat& B) const {
    SpinMat C;
    for (int i = 0; i < 16; i ++) C.data[i] = Vcomplex();
    for (int a = 0; a < 4; a ++)
      for (int b = 0; b < 4; b ++)
        for (int c = 0; c < 4; c ++)
          C.data[4*a+c] += data[4*a+b] * B.data[4*b+c];
    return C;
  }

  SpinMat operator+(const SpinMat& B) const{
    SpinMat C;
    for (int i = 0; i < 16; i ++)
      C.data[i] = data[i] + B.data[i];
    return C;
  }

  SpinMat operator+=(const SpinMat& B){
    for (int i = 0; i < 16; i ++)
      data[i] += B.data[i];
    return *this;
  }

  SpinMat operator-(const SpinMat& B) const{
    SpinMat C;
    for (int i = 0; i < 16; i ++)
      C.data[i] = data[i] - B.data[i];
    return C;
  }

  SpinMat operator*(const double r) const{
    SpinMat C;
    for (int i = 0; i < 16; i ++)
      C.data[i] = data[i] * r;
    return C;
  }

  SpinMat operator*(const Vcomplex z) const{
    SpinMat C;
    for (int i = 0; i < 16; i ++)
      C.data[i] = data[i] * z;
    return C;
  }

  SpinMat operator*=(const double r){
    for (int i = 0; i < 16; i ++)
      data[i] *= r;
    return *this;
  }

  SpinMat operator*=(const Vcomplex z){
    for (int i = 0; i < 16; i ++)
      data[i] *= z;
    return *this;
  }

  SpinMat transpose() {
    SpinMat A;
    for (int a = 0; a < 4; a ++)
      for (int b = 0; b < 4; b ++)
        A.data[4*a+b] = data[4*b+a];
    return A;
  }

  // Compute the Hermitian conjugate
  // This is the combined action of complex conjugation and transposition
  SpinMat hconj() {
    SpinMat A;
    for (int a = 0; a < 4; a ++)
      for (int b = 0; b < 4; b ++)
        A.data[4*a+b] = data[4*b+a];
    double * complexes = (double*) A.data;
    for (int i = 0; i < 16; i ++)
      complexes[2*i+1] *= -1;
    return A;
  }
};

static Vcomplex Trace(SpinMat mat) {
  return mat.data[0] + mat.data[5] + mat.data[10] + mat.data[15];
}

struct WeylMat {
  WeylMat() {data = _mm512_setzero_pd();}
  WeylMat(__m512d dat) : data{dat} {}
  __m512d data;
  WeylMat operator*(const WeylMat& B) const {
    __m512i perm0 = {0, 1, 2, 3, 0, 1, 2, 3};
    __m512i perm1 = {4, 5, 6, 7, 4, 5, 6, 7};
    // a0 = (a0, a0, a0, a0, a4, a4, a4, a4)
    __m512d a0 = _mm512_permutex_pd(data, 0b00000000);
    __m512d a1 = _mm512_permutex_pd(data, 0b01010101);
    __m512d a2 = _mm512_permutex_pd(data, 0b10101010);
    __m512d a3 = _mm512_permutex_pd(data, 0b11111111);
    // b0 = (b0, b1, b2, b3, b0, b1, b2, b3)
    __m512d b0 = _mm512_permutexvar_pd(perm0, B.data);
    __m512d b1 = _mm512_permute_pd(b0, 0b01010101);
    // b2 = (b4, b5, b6, b7, b4, b5, b6, b7)
    __m512d b2 = _mm512_permutexvar_pd(perm1, B.data);
    __m512d b3 = _mm512_permute_pd(b2, 0b01010101);
    // C = a0 * b0 -/+ a1 * b1 + a2 * b2 -/+ a3 * b3
    __m512d C;
    C = _mm512_mul_pd(a1, b1);
    C = _mm512_fmadd_pd(a3, b3, C);
    C = _mm512_fmaddsub_pd(a0, b0, C);
    C = _mm512_fmadd_pd(a2, b2, C);
    return WeylMat(C);
  }
  WeylMat operator+(const WeylMat& B) const {
    return WeylMat(_mm512_add_pd(data, B.data));
  }
  WeylMat operator+=(const WeylMat& B) {
    data = _mm512_add_pd(data, B.data);
    return *this;
  }
  WeylMat operator*(const double r) {
    __m512d factor = _mm512_set1_pd(r);
    return WeylMat(_mm512_mul_pd(data, factor));
  }
  // Note: This is completely unvectorized (and likely slow)
  WeylMat operator*(const Vcomplex z) {
    WeylMat C;
    Vcomplex * data_array = (Vcomplex*)(&data);
    Vcomplex * data_array_new = (Vcomplex*)(&(C.data));
    for (int i = 0; i < 4; i ++) 
      data_array_new[i] = data_array[i] * z;
    return C;
  }
  WeylMat transpose() {
    __m512i trans = {0, 1, 4, 5, 2, 3, 6, 7}; 
    return WeylMat(_mm512_permutexvar_pd(trans, data)); 
  };
};

static WeylMat ExtractWeyl(SpinMat S)
{
  Vcomplex data [4] = {2.0 * S(0,0), 2.0 * S(0,1), 2.0 * S(1,0), 2.0 * S(1,1)};
  return WeylMat(*(__m512d*) &data);
}

// extract a WeylMat in some quadrant of a SpinMat
// Note: This does not multiply by 2
// The original ExtractWeyl is equal to 2 * ExtractWeyl(S, 0, 0)
static WeylMat ExtractSpecificWeyl(SpinMat S, int r, int c)
{
  Vcomplex data [4] = {S(2*r,2*c), S(2*r,2*c+1), S(2*r+1,2*c), S(2*r+1,2*c+1)};
  return WeylMat(*(__m512d*) &data);
}

// compute the trace (sum of diagonal entries)
static Vcomplex Trace(WeylMat mat) {
  Vcomplex * complex_array = (Vcomplex *) &mat;
  return complex_array[0] + complex_array[3];
}

// Horizontal sum of 512-bit register
// This is a helper method in the multiply-and-trace routine below
static Vcomplex horizontal_sum(__m512d A) {
  __m256d temp_256 = _mm512_extractf64x4_pd(A, 0);
  temp_256 = _mm256_add_pd(temp_256, _mm512_extractf64x4_pd(A, 1));
  __m128d temp_128 = _mm256_extractf128_pd(temp_256, 0);
  temp_128 = _mm_add_pd(temp_128, _mm256_extractf128_pd(temp_256, 1));
  return temp_128;
}

// Multiply and trace at once
// In principle, this could be made faster than separately multiplying and tracing
// It was not used in this codebase but could be used as an optimization
static Vcomplex Trace(WeylMat A, WeylMat B) {
  // real part: element-wise multiplication of AB^T
  __m512d B_T = B.transpose().data;
  __m512d real_contributions = _mm512_mul_pd(A.data, B_T);
  // horizontal sum of real_contributions
  //double real = horizontal_sum_difference(real_contributions);
  // permute pairs in B to compute real * imag
  B_T = _mm512_permutex_pd(B_T, 0b10110001);
  __m512d imag_contributions = _mm512_mul_pd(A.data, B_T);
  // permute real and imaginary to get (RIRIRIRI) twice
  // XOR trick negates real_contributions in first line
  __m512d real_imag = _mm512_unpackhi_pd(
                        _mm512_sub_pd(_mm512_setzero_pd(), real_contributions),
                        imag_contributions);
  real_imag = _mm512_add_pd(real_imag, _mm512_unpacklo_pd(real_contributions, 
                                                          imag_contributions));
  return horizontal_sum(real_imag);
}

#endif
