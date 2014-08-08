#include "dpKernel.hpp"
#ifndef __dpLUDecomposition_H_INCLUDED__
#define __dpLUDecomposition_H_INCLUDED__

class dpLUDecomposition: public dpKernel{

	void* inMapPtr;
  void* outMapPtr;
	cl_kernel					kernelLUD;					/**< CL kernel LU Decomposition*/
	cl_kernel					kernelCombine;			/**< CL Kerenl Combine */
	cl_mem						inplaceBuffer;			/**< CL memory buffer */
  cl_mem						inputBuffer2;				/**< CL memory output Buffer */
	cl_double					*input;							/**< Input array */
	cl_double					*matrixGPU;      		/**< Inplace Array for GPU */
	cl_int						effectiveDimension;	/**< effectiveDimension(square matrix) of the input matrix */
	
	public:
		dpLUDecomposition(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateMatrix(double*, int,int);
};

#endif