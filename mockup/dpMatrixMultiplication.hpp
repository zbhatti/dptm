#include "dpKernel.hpp"

class dpMatrixMultiplication: public dpKernel{
	float *A, *B, *C;
	int szA, szB, szC;
	int N, P, M;
	cl_mem a_in, b_in, c_out; 
	cl_int err;
	
	
	public:
		dpMatrixMultiplication(float*, float*, int, int, int, cl_context, cl_command_queue, int, int);
		void memoryCopyOut();
		void plan();
		void execute();
		void memoryCopyIn();
		void cleanUp();
};