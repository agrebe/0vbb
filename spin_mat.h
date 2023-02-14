#ifndef SPIN_MAT_H
#define SPIN_MAT_H

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
  // TODO: Vectorize this
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
static WeylMat ExtractWeyl(SpinMat S, int r, int c)
{
  Vcomplex data [4] = {S(2*r,2*c), S(2*r,2*c+1), S(2*r+1,2*c), S(2*r+1,2*c+1)};
  return WeylMat(*(__m512d*) &data);
}

static Vcomplex Trace(WeylMat mat) {
  Vcomplex * complex_array = (Vcomplex *) &mat;
  return complex_array[0] + complex_array[3];
}

// horizontal sum of 512-bit register
static Vcomplex horizontal_sum(__m512d A) {
  __m256d temp_256 = _mm512_extractf64x4_pd(A, 0);
  temp_256 = _mm256_add_pd(temp_256, _mm512_extractf64x4_pd(A, 1));
  __m128d temp_128 = _mm256_extractf128_pd(temp_256, 0);
  temp_128 = _mm_add_pd(temp_128, _mm256_extractf128_pd(temp_256, 1));
  return temp_128;
}

// multiply and trace at once
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
  __m512d real_imag = _mm512_unpackhi_pd(_mm512_sub_pd(_mm512_setzero_pd(), real_contributions), imag_contributions);
  real_imag = _mm512_add_pd(real_imag, _mm512_unpacklo_pd(real_contributions, imag_contributions));
  return horizontal_sum(real_imag);
}

#endif
