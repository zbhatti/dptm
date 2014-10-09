#include "dpRotateImage.hpp"
#include "errorCheck.hpp"
#include <string.h>
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }



dpRotateImage::dpRotateImage(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;
	name = "RotateImage";
	
	kernelString = "\n"
		"__kernel \n"
		"void img_rotate(__global float* dest_data, \n"
		"                __global float* src_data, \n"
		"                           int  W, \n"
		"                           int  H, \n"
		"                         float  sinTheta, \n" 
		"                         float  cosTheta) { \n"
		"   //Work-item gets its index within index space \n"
		"   const int ix = get_global_id(0); \n"
		"   const int iy = get_global_id(1); \n"
		"   //Calculate location of data to move into (ix,iy) \n"
		"   //Output decomposition as mentioned \n"
		"   float x0 = W/2.0f; \n"
		"   float y0 = H/2.0f; \n"
		"   float xOff = ix - x0; \n"
		"   float yOff = iy - y0; \n"
		"   int xpos = (int)(xOff*cosTheta + yOff*sinTheta + x0 ); \n"
		"   int ypos = (int)(yOff*cosTheta - xOff*sinTheta + y0 ); \n"
		"   // Bounds Checking \n"
		"   if((xpos>=0) && (xpos< W) && (ypos>=0) && (ypos< H)) { \n"
		"			//try taking away the global memory access done here. use local or private memory \n"
		"      dest_data[iy*W+ix] = src_data[ypos*W+xpos]; \n"
		"   } \n"
		"} ";
		
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "img_rotate", &err); clErrChk(err);
}

void dpRotateImage::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = xLocal; 
	localSize[1] = yLocal;
	localSize[2] = 1;
	
	MB=dataMB;

}

void dpRotateImage::init(){
	
	theta = 3.14159/6;
	cos_theta = cosf(theta);
	sin_theta = sinf(theta);
	strcpy(inputFile,"./src/data/input.bmp");
	strcpy(outputFile,"./src/data/output.bmp");
	float * tmp = readImage(inputFile, &imageWidth, &imageHeight);
	inputImage = new float[imageHeight*imageWidth];
	inputImage = tmp;
	dataSize = imageHeight*imageWidth*sizeof(float);
	outputImage = new float[imageHeight*imageWidth];
	
	dataParameters.push_back(imageHeight);
	dataParameters.push_back(imageWidth);
	dataNames.push_back("ImageHeight");
	dataNames.push_back("ImageWidth");
}

void dpRotateImage::memoryCopyOut(){
	d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &err);  clErrChk(err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL,	&err);  clErrChk(err);
	
	clErrChk(clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, dataSize, inputImage, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpRotateImage::plan(){
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_output));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_input));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &imageWidth));
	clErrChk(clSetKernelArg(kernel, 3, sizeof(int), &imageHeight));
	clErrChk(clSetKernelArg(kernel, 4, sizeof(float), &sin_theta));
	clErrChk(clSetKernelArg(kernel, 5, sizeof(float), &cos_theta));
	globalSize[0] = imageWidth;
	globalSize[1] = imageHeight;
}

int dpRotateImage::execute(){
	err=clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
	clErrChk(err);
	if(err<0) 
		return -1;
	clErrChk(clFinish(queue));
	return 0;
}

void dpRotateImage::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, dataSize, outputImage, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpRotateImage::cleanUp(){
	// Write the output image to file
	storeImage(outputImage, outputFile, imageHeight, imageWidth, inputFile);
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseMemObject(d_input));
	clErrChk(clReleaseMemObject(d_output));
	delete[] outputImage;
	delete[] inputImage;
}





	
