#include "cmath"
typedef int a_int_t;
typedef float a_float_t;
#define cmplx a_cmplx_t

#ifdef _CL_CUDA_READY_ ///////////////////////////////////////////////////////// DEVICE
#ifdef _OPENCL_
#define _POW_ pow
#define _SQRT_ sqrt
#define _EXP_ exp
#define _LOG_ log
#define _ATAN_ atan
#define _TAN_ tan
#define _ACOS_ acos
#define _COS_ cos
#define _ASIN_ asin
#define _SIN_ sin
#define _CL_CUDA_HOST_
#define _CL_CUDA_DEVICE_
#define _CL_CUDA_GLOBAL_ __global 
#define _CL_CUDA_CONSTANT_ __constant 
#define _CL_CUDA_KERNEL_ __kernel 
#endif
#ifdef _CUDA_
#define _POW_ powf
#define _SQRT_ sqrtf
#define _EXP_ expf
#define _LOG_ logf
#define _ATAN_ atanf
#define _TAN_ tanf
#define _ACOS_ acosf
#define _COS_ cosf
#define _ASIN_ asinf
#define _SIN_ sinf
#define _CL_CUDA_HOST_ __host__
#define _CL_CUDA_DEVICE_ __device__
#define _CL_CUDA_BOTH_ __host__ __device__
#define _CL_CUDA_GLOBAL_
#define _CL_CUDA_CONSTANT_
#define _CL_CUDA_KERNEL_ __global__
#endif
#else //////////////////////////////////////////////////////////////////////////// HOST #include <cmath>
#define _POW_ std::pow
#define _SQRT_ std::sqrt
#define _EXP_ std::exp
#define _LOG_ std::log
#define _TAN_ std::tan
#define _ATAN_ std::atan
#define _COS_ std::cos
#define _ACOS_ std::acos
#define _SIN_ std::sin
#define _ASIN_ std::asin
#define _CL_CUDA_HOST_
#define _CL_CUDA_DEVICE_
#define _CL_CUDA_GLOBAL_
#define _CL_CUDA_CONSTANT_
#define _CL_CUDA_KERNEL_
#define _CL_CUDA_IDX_ 0
#endif ////////////////////////////////////////////////////////////////////////////////
