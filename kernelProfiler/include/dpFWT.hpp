#include "dpKernel.hpp"
#ifndef __dpFWT_H_INCLUDED__
#define __dpFWT_H_INCLUDED__

class dpFWT: public dpKernel{

	cl_int                 length;       /**< Length of the input array */
	cl_float               *input;       /**< Input array */
	cl_float              *output;       /**< Ouput array */
	cl_mem            inputBuffer;       /**< CL memory buffer */
				
				
	public:
		dpFWT(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float*, int);
};

#endif