#include "matcommon.h"

#ifndef cmplx_h
#define cmplx_h


struct a_cmplx_t {
  a_float_t re; a_float_t im;
  
  _CL_CUDA_DEVICE_ a_cmplx_t() {}
  _CL_CUDA_DEVICE_ a_cmplx_t(a_float_t x, a_float_t y) { re = x; im = y; }
  _CL_CUDA_DEVICE_ a_cmplx_t(a_float_t x)  { re = x; im = 0; }
  _CL_CUDA_DEVICE_ a_cmplx_t(const a_cmplx_t& c)  { re = c.re; im = c.im; }
  
  a_cmplx_t& operator=(const a_float_t& x) { re  = x; im = 0; }
  a_cmplx_t& operator=(const a_cmplx_t& c) { re = c.re; im = c.im; }
};


inline _CL_CUDA_DEVICE_ a_float_t real(a_cmplx_t a) { return a.re;}

inline _CL_CUDA_DEVICE_ a_float_t imag(a_cmplx_t a ) { return a.im;}

inline _CL_CUDA_DEVICE_ a_cmplx_t conj(a_cmplx_t a ) { return a_cmplx_t(a.re,-a.im);} 

inline _CL_CUDA_DEVICE_ a_float_t fabsc(a_cmplx_t a){ return _SQRT_((a.re*a.re)+(a.im*a.im)); }

inline _CL_CUDA_DEVICE_ a_float_t fabsc_sqr(a_cmplx_t a) { return (a.re*a.re)+(a.im*a.im); }

inline _CL_CUDA_DEVICE_ a_cmplx_t operator+(a_cmplx_t a, a_cmplx_t b) { return a_cmplx_t(a.re + b.re, a.im + b.im); } 

inline _CL_CUDA_DEVICE_ a_cmplx_t operator+(a_float_t a, a_cmplx_t b) { return a_cmplx_t(a + b.re, b.im); }

inline _CL_CUDA_DEVICE_ a_cmplx_t operator+(a_cmplx_t a) { return a_cmplx_t(+a.re, +a.im); }

inline _CL_CUDA_DEVICE_ a_cmplx_t operator-(a_cmplx_t a, a_cmplx_t b) { return a_cmplx_t(a.re - b.re, a.im - b.im); } 

inline _CL_CUDA_DEVICE_ a_cmplx_t operator-(a_float_t a, a_cmplx_t b) { return a_cmplx_t(a - b.re, -b.im); }

inline _CL_CUDA_DEVICE_ a_cmplx_t operator-(a_cmplx_t a) { return a_cmplx_t(-a.re, -a.im); }

inline _CL_CUDA_DEVICE_ a_cmplx_t operator*(a_cmplx_t a, a_cmplx_t b) { return a_cmplx_t((a.re * b.re) - (a.im * b.im), (a.re * b.im) + (a.im * b.re)); }

inline _CL_CUDA_DEVICE_ a_cmplx_t operator*(a_cmplx_t a, a_float_t s) {return a_cmplx_t(a.re * s, a.im * s); }

inline _CL_CUDA_DEVICE_ a_cmplx_t operator*(a_float_t s, a_cmplx_t a) {return a_cmplx_t(a.re * s, a.im * s); }

inline _CL_CUDA_DEVICE_ a_cmplx_t operator/(a_cmplx_t a, a_cmplx_t b) { 
  a_float_t t=(1./(b.re*b.re+b.im*b.im)); 
  return a_cmplx_t( ( (a.re * b.re) + (a.im * b.im))*t, (-(a.re * b.im) + (a.im * b.re))*t ); 
}

inline _CL_CUDA_DEVICE_ a_cmplx_t operator/(a_cmplx_t a, a_float_t s) {return a * (1. / s); }

inline _CL_CUDA_DEVICE_ a_cmplx_t operator/(a_float_t s, a_cmplx_t a) { 
  a_float_t inv = s*(1./(a.re*a.re+a.im*a.im));
  return a_cmplx_t(inv*a.re,-inv*a.im);
}

#endif // cmplx_h
