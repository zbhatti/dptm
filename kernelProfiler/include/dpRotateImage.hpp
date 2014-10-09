#include "dpKernel.hpp"
#include "bmpfuncs.h"

#ifndef __dpRotateImage_H_INCLUDED__
#define __dpRotateImage_H_INCLUDED__


class dpRotateImage: public dpKernel{
	
	cl_mem d_input, d_output;
	float theta, cos_theta, sin_theta;
	int imageHeight, imageWidth, dataSize;
	float *inputImage, *outputImage;
	char inputFile[48];
	char outputFile[48];
	
	//source: http://www.heterogeneouscompute.org/?page_id=7
	
	public:
		dpRotateImage(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
};

#endif