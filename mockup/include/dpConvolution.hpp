#include "dpKernel.hpp"
#ifndef __dpConvolution_H_INCLUDED__
#define __dpConvolution_H_INCLUDED__

class dpConvolution: public dpKernel{
	
	cl_int       width;              /**< Width of the Input array */
	cl_int       height;             /**< Height of the Input array */
	cl_uint      *input;             /**< Input array */
	cl_uint      *output;            /**< Output array */
	cl_float     *mask;              /**< mask array */
	cl_uint      maskWidth;          /**< mask dimensions */
	cl_uint      maskHeight;         /**< mask dimensions */
	cl_uint			 *verificationOutput;/**< Output array for reference implementation */
	cl_mem       inputBuffer;        /**< CL memory input buffer */
	cl_mem       outputBuffer;       /**< CL memory output buffer */
	cl_mem       maskBuffer; 				 /**< CL memory mask buffer */
	
	public:
		dpConvolution(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		template<typename T> void fillRandom(T*, const int, const int, const T, const T);
};

#endif