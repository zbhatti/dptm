#include "dpKernel.hpp"
#include "helperFunctions/bmpfuncs.h"

class dpRotateImage: public dpKernel{
	
	cl_mem d_input, d_output;
	cl_int err;
	float theta, cos_theta, sin_theta;
	int imageHeight, imageWidth, dataSize;
	float *inputImage, *outputImage;
	char inputFile[48];
	char outputFile[48];
	
	public:
		dpRotateImage(cl_context, cl_command_queue, int, int);
		void memoryCopyOut();
		void plan();
		void execute();
		void memoryCopyIn();
		void cleanUp();
};