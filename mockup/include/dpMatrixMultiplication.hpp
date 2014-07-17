#include "dpKernel.hpp"
#ifndef __dpMatrixMultiplication_H_INCLUDED__
#define __dpMatrixMultiplication_H_INCLUDED__


class dpMatrixMultiplication: public dpKernel{
	float *A, *B, *C;
	int szA, szB, szC;
	int N, P, M;
	cl_mem a_in, b_in, c_out; 
	cl_int err;
	
	//source: http://www.cs.bris.ac.uk/home/simonm/workshops/OpenCL_lecture3.pdf
	const char* kernelString;
	
	public:
		dpMatrixMultiplication(cl_context, cl_command_queue);
		void init(int,int,int);
		void memoryCopyOut();
		void plan();
		void execute();
		void memoryCopyIn();
		void cleanUp();
		void generateMatrix(float*, int, int);
};

#endif