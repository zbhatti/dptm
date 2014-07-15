#include "dpKernel.hpp"
#include "bmpfuncs.h"

class dpRotateImage: public dpKernel{
	
	cl_mem d_input, d_output;
	cl_int err;
	float theta, cos_theta, sin_theta;
	int imageHeight, imageWidth, dataSize;
	float *inputImage, *outputImage;
	char inputFile[48];
	char outputFile[48];
	
	//source: http://www.heterogeneouscompute.org/?page_id=7
	const char* kernelString;
	
	public:
		dpRotateImage(cl_context, cl_command_queue);
		void init(int,int,int);
		void memoryCopyOut();
		void plan();
		void execute();
		void memoryCopyIn();
		void cleanUp();
};