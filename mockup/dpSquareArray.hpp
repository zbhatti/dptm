#include "dpKernel.hpp"

class dpSquareArray: public dpKernel{
	cl_mem Ain_d, Aout_d;
	float *Ain, *Aout;
	int Asize;
	int err;
	
	public:
		dpSquareArray(float*, int, cl_context, cl_command_queue, int);
		void memoryCopyOut();
		void plan();
		void execute();
		void memoryCopyIn();
		void cleanUp();
};