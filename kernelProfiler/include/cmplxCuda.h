typedef int a_int_t;
typedef float a_float_t;
#define cmplx a_cmplx_t

#ifndef cmplx_h
#define cmplx_h

struct a_cmplx_t {
  a_float_t re; 
	a_float_t im;
  
	__host__ __device__ a_cmplx_t() {}
	__host__ __device__ a_cmplx_t(a_float_t x, a_float_t y) { re = x; im = y; }

	__host__ __device__ a_cmplx_t(a_float_t x)  { re = x; im = 0; }
	__host__ __device__ a_cmplx_t(const a_cmplx_t& c)  { re = c.re; im = c.im; }
 
  __host__ __device__ a_cmplx_t& operator=(const a_float_t& x) { re  = x; im = 0; return *this;}
  __host__ __device__ a_cmplx_t& operator=(const a_cmplx_t& c) { re = c.re; im = c.im; return *this;}
};

inline __host__ __device__ a_cmplx_t mkcmplx(a_float_t x, a_float_t y) { return cmplx(x,y);}

inline __host__ __device__ a_float_t real(a_cmplx_t a) { return a.re;}

inline __host__ __device__ a_float_t imag(a_cmplx_t a ) { return a.im;}
 
inline __host__ __device__ a_cmplx_t conj(a_cmplx_t a ) { return a_cmplx_t(a.re,-a.im);} 
 
inline __host__ __device__ a_float_t fabsc(a_cmplx_t a){ return sqrt((a.re*a.re)+(a.im*a.im)); }
 
inline __host__ __device__ a_float_t fabsc_sqr(a_cmplx_t a) { return (a.re*a.re)+(a.im*a.im); }
 
inline __host__ __device__ a_cmplx_t operator+(a_cmplx_t a, a_cmplx_t b) { return a_cmplx_t(a.re + b.re, a.im + b.im); } 
 
inline __host__ __device__ a_cmplx_t operator+(a_float_t a, a_cmplx_t b) { return a_cmplx_t(a + b.re, b.im); }

inline __host__ __device__ a_cmplx_t operator+(a_cmplx_t a) { return a_cmplx_t(+a.re, +a.im); }
 
inline __host__ __device__ a_cmplx_t operator-(a_cmplx_t a, a_cmplx_t b) { return a_cmplx_t(a.re - b.re, a.im - b.im); } 

inline __host__ __device__ a_cmplx_t operator-(a_float_t a, a_cmplx_t b) { return a_cmplx_t(a - b.re, -b.im); }
 
inline __host__ __device__ a_cmplx_t operator-(a_cmplx_t a) { return a_cmplx_t(-a.re, -a.im); }

inline __host__ __device__ a_cmplx_t operator*(a_cmplx_t a, a_cmplx_t b) { return a_cmplx_t((a.re * b.re) - (a.im * b.im), (a.re * b.im) + (a.im * b.re)); }

inline __host__ __device__ a_cmplx_t operator*(a_cmplx_t a, a_float_t s) {return a_cmplx_t(a.re * s, a.im * s); }

inline __host__ __device__ a_cmplx_t operator*(a_float_t s, a_cmplx_t a) {return a_cmplx_t(a.re * s, a.im * s); }

inline __host__ __device__ a_cmplx_t operator/(a_cmplx_t a, a_cmplx_t b) { 
  a_float_t t=(1./(b.re*b.re+b.im*b.im)); 
  return a_cmplx_t( ( (a.re * b.re) + (a.im * b.im))*t, (-(a.re * b.im) + (a.im * b.re))*t ); 
}

inline __host__ __device__ a_cmplx_t operator/(a_cmplx_t a, a_float_t s) {return a * (1. / s); }

inline __host__ __device__ a_cmplx_t operator/(a_float_t s, a_cmplx_t a) { 
  a_float_t inv = s*(1./(a.re*a.re+a.im*a.im));
  return a_cmplx_t(inv*a.re,-inv*a.im);
}

#endif // cmplx_h
