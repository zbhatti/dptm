#include "dpKernel.hpp"

class dpSquareArray: public dpKernel{
	cl_mem Ain_d, Aout_d;
	float *Ain, *Aout;
	int Asize;
	int err;

	const char* kernelString;
	
	public:
		dpSquareArray(cl_context, cl_command_queue);
		void init(int,int,int); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		void execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float*, int);
};