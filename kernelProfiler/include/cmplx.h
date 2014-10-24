typedef int a_int_t;
typedef float a_float_t;
#define cmplx a_cmplx_t

#ifndef cmplx_h
#define cmplx_h

struct a_cmplx_t {
  a_float_t re; 
	a_float_t im;
  
	a_cmplx_t() {}
	a_cmplx_t(a_float_t x, a_float_t y) { //C++ overloaded function / constructor
		re = x; 
		im = y; 
	}
	
	a_cmplx_t(a_float_t x)  {  //C++ overloaded function / constructor
		re = x; 
		im = 0; 
	}
	
	a_cmplx_t(const a_cmplx_t& c)  { //C++ overloaded function / constructor
		re = c.re; 
		im = c.im; 
	}
  
  a_cmplx_t& operator=(const a_float_t& x) { //C++ overloaded operator
		re  = x; 
		im = 0; 
		return *this;
	}
	
  a_cmplx_t& operator=(const a_cmplx_t& c) { //C++ overloaded operator
		re = c.re; 
		im = c.im; 
		return *this;
	}
	
};


inline a_float_t real(a_cmplx_t a) { //C++ overloaded operator
	return a.re;
}

inline a_float_t imag(a_cmplx_t a ) {  //C++ overloaded operator
	return a.im;
}

inline a_cmplx_t conj(a_cmplx_t a ) {  //C++ overloaded operator
	return a_cmplx_t(a.re,-a.im);
} 

inline a_float_t fabsc(a_cmplx_t a){  //C++ overloaded operator
	return sqrt((a.re*a.re)+(a.im*a.im)); 
}

inline a_float_t fabsc_sqr(a_cmplx_t a) {  //C++ overloaded operator
	return (a.re*a.re)+(a.im*a.im); 
}

inline a_cmplx_t operator+(a_cmplx_t a, a_cmplx_t b) {  //C++ overloaded operator
	return a_cmplx_t(a.re + b.re, a.im + b.im); 
}

inline a_cmplx_t operator+(a_float_t a, a_cmplx_t b) {  //C++ overloaded operator
	return a_cmplx_t(a + b.re, b.im); 
}

inline a_cmplx_t operator+(a_cmplx_t a) {  //C++ overloaded operator
	return a_cmplx_t(+a.re, +a.im); 
}

inline a_cmplx_t operator-(a_cmplx_t a, a_cmplx_t b) {  //C++ overloaded operator
	return a_cmplx_t(a.re - b.re, a.im - b.im); 
}

inline a_cmplx_t operator-(a_float_t a, a_cmplx_t b) {  //C++ overloaded operator
	return a_cmplx_t(a - b.re, -b.im); 
} 

inline a_cmplx_t operator-(a_cmplx_t a) {  //C++ overloaded operator
	return a_cmplx_t(-a.re, -a.im); 
}

inline a_cmplx_t operator*(a_cmplx_t a, a_cmplx_t b) {  //C++ overloaded operator
	return a_cmplx_t((a.re * b.re) - (a.im * b.im), (a.re * b.im) + (a.im * b.re)); 
} 

inline a_cmplx_t operator*(a_cmplx_t a, a_float_t s) { //C++ overloaded operator
	return a_cmplx_t(a.re * s, a.im * s); 
} 

inline a_cmplx_t operator*(a_float_t s, a_cmplx_t a) { //C++ overloaded operator
	return a_cmplx_t(a.re * s, a.im * s); 
} 

inline a_cmplx_t operator/(a_cmplx_t a, a_cmplx_t b) {  //C++ overloaded operator
  a_float_t t=(1./(b.re*b.re+b.im*b.im)); 
  return a_cmplx_t( ( (a.re * b.re) + (a.im * b.im))*t, (-(a.re * b.im) + (a.im * b.re))*t ); 
}

inline a_cmplx_t operator/(a_cmplx_t a, a_float_t s) { //C++ overloaded operator
	return a * (1. / s); 
} 

inline a_cmplx_t operator/(a_float_t s, a_cmplx_t a) {  //C++ overloaded operator
  a_float_t inv = s*(1./(a.re*a.re+a.im*a.im));
  return a_cmplx_t(inv*a.re,-inv*a.im);
}

#endif // cmplx_h
