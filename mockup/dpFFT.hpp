#include <clFFT.h>
#include "dpKernel.hpp"

class dpFFT: public dpKernel{

	float *Ain, *Aout;
	int Asize;
	cl_mem buffer;
	cl_int err;
	clfftPlanHandle planHandle;
	clfftSetupData fftSetup;
	size_t clLengths[1];
	
	public:
		dpFFT(float*, int, cl_context, cl_command_queue);
		void memoryCopyOut();
		void plan();
		void execute();
		void memoryCopyIn();
		void cleanUp();
};