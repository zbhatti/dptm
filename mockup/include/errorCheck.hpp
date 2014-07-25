//http://tom.scogland.com/blog/2013/03/29/opencl-errors/
//http://stackoverflow.com/a/14038590/2573580


#include <stdio.h>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif

const char* get_error_string(cl_int);
void clAssert(cl_int, const char*, int);
void programCheck(cl_int, cl_context, cl_program);

//#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }




