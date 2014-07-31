#include "dpKernel.hpp"
#ifndef __dpMersenneTwister_H_INCLUDED__
#define __dpMersenneTwister_H_INCLUDED__

class dpMersenneTwister: public dpKernel{
	cl_uint numRands;               /**< Number of random number to be generated*/
	cl_uint mulFactor;              /**< Number of generated random numbers for each seed */
	cl_uint *seeds;                 /**< Array of seeds */
	cl_float *deviceResult;         /**< Array of Generated random numbersby specified device */
	cl_mem seedsBuf;                /**< CL memory buffer for seeds */
	cl_mem resultBuf;               /**< CL memory buffer for random numbers */
	cl_int width;                   /**< width of the execution domain */
	cl_int height;                  /**< height of the execution domain */
	
	public:
		dpMersenneTwister(cl_context, cl_command_queue);
		void init(int,int,int);
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
};
#endif