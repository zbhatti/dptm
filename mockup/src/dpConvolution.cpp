/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/
#include "dpConvolution.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }
/**
 * SimpleConvolution is where each pixel of the output image
 * is the weighted sum of the neighborhood pixels of the input image
 * The neighborhood is defined by the dimensions of the mask and 
 * weight of each neighbor is defined by the mask itself.
 * @param output Output matrix after performing convolution
 * @param input  Input  matrix on which convolution is to be performed
 * @param mask   mask matrix using which convolution was to be performed
 * @param inputDimensions dimensions of the input matrix
 * @param maskDimensions  dimensions of the mask matrix
 */
   
dpConvolution::dpConvolution(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;	
	
	name = "Convolution";
	kernelString = "\n"	
		"	__kernel void simpleConvolution(\n"
		"__global  uint  * output,\n"
		"__global  uint  * input,\n"
		"__global  float  * mask,\n"
		"const uint2  inputDimensions,\n"
		"const uint2  maskDimensions)\n"
		"{\n"
		"    uint tid   = get_global_id(0);\n"
		"    uint width  = inputDimensions.x;\n"
		"    uint height = inputDimensions.y;\n"
		"    uint x      = tid%width;\n"
		"    uint y      = tid/width;\n"
		"    uint maskWidth  = maskDimensions.x;\n"
		"    uint maskHeight = maskDimensions.y;\n"
		"    uint vstep = (maskWidth  -1)/2;\n"
		"    uint hstep = (maskHeight -1)/2;\n"
		"    uint left    = (x           <  vstep) ? 0         : (x - vstep);\n"
		"    uint right   = ((x + vstep) >= width) ? width - 1 : (x + vstep);\n"
		"    uint top     = (y           <  hstep) ? 0         : (y - hstep);\n"
		"    uint bottom  = ((y + hstep) >= height)? height - 1: (y + hstep);\n" 
		"    float sumFX = 0;\n"
		"    for(uint i = left; i <= right; ++i)\n"
		"        for(uint j = top ; j <= bottom; ++j)\n"
		"        {\n"
		"            uint maskIndex = (j - (y - hstep)) * maskWidth  + (i - (x - vstep));\n"
		"            uint index     = j                 * width      + i;\n"
		"            sumFX += ((float)input[index] * mask[maskIndex]);\n"
		"        }\n"
		"    sumFX += 0.5f;\n"
		"    output[tid] = (uint)sumFX;\n"
		"}\n";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "simpleConvolution", &err); clErrChk(err);
	
}

void dpConvolution::setup(int dataMB, int xLocal, int yLocal, int zLocal){
		
	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
	
	//for (int i =0; pow(2,i)*pow(2,i)*sizeof(cl_uint)/(float) 1048576 <= dataMB;i++){
		//width = pow(2,i);
		//height = pow(2,i);
	//}
	
	width = (int) sqrt(1048576*dataMB/(sizeof(cl_uint)) );
	height = width;
	
	//for mask to not run out of bounds:
	if(width * height < 256){
			width = 64;
			height = 64;
	}
	
	MB = width*height*sizeof(cl_uint)/(float) 1048576;
}

void dpConvolution::init(){
	cl_uint inputSizeBytes;
	
	maskWidth = 64;
	maskHeight = 64;
	
	dataParameters.push_back(width);
	dataParameters.push_back(height);
	dataParameters.push_back(maskWidth);
	dataParameters.push_back(maskHeight);
	dataNames.push_back("width");
	dataNames.push_back("height");
	dataNames.push_back("maskwidth");
	dataNames.push_back("maskheight");
	
	// allocate and init memory used by host
	inputSizeBytes = width * height * sizeof(cl_uint);
	input  = (cl_uint *) malloc(inputSizeBytes);
	output = (cl_uint  *) malloc(inputSizeBytes);
	cl_uint maskSizeBytes = maskWidth * maskHeight * sizeof(cl_float);
	mask = (cl_float  *) malloc(maskSizeBytes);
	// random initialisation of input
	fillRandom<cl_uint >(input, width, height, 0, 255);

	// Fill a blurr filter or some other filter of your choice
	for(cl_uint i = 0; i < maskWidth*maskHeight; i++)
		mask[i] = 0;

	cl_float val = 1.0f / (maskWidth * 2.0f - 1.0f);

	for(cl_uint i = 0; i < maskWidth; i++){
			cl_uint y = maskHeight / 2;
			mask[y * maskWidth + i] = val;
	}

	for(cl_uint i = 0; i < maskHeight; i++){
			cl_uint x = maskWidth / 2;
			mask[i * maskWidth + x] = val;
	}
	  
	
}

void dpConvolution::memoryCopyOut(){
	
	inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_uint ) * width * height, NULL, &err); clErrChk(err);
	outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint ) * width * height, NULL, &err); clErrChk(err);
	maskBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float ) * maskWidth * maskHeight, NULL, &err); clErrChk(err);
	
	//this is a combined allocation and copy function
	clErrChk(clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, sizeof(cl_uint ) * width * height, input, 0, NULL, NULL)); 
	clErrChk(clEnqueueWriteBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(cl_uint ) * width * height, output, 0, NULL, NULL));
	clErrChk(clEnqueueWriteBuffer(queue, maskBuffer, CL_TRUE, 0, sizeof(cl_float ) * maskWidth * maskHeight, mask, 0, NULL, NULL));
	clFinish(queue);
}

void dpConvolution::plan(){
	// Set appropriate arguments to the kernel
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&outputBuffer));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inputBuffer));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&maskBuffer));
	cl_uint2 inputDimensions = {{width, height}};
	cl_uint2 maskDimensions  = {{maskWidth, maskHeight}};
	clErrChk(clSetKernelArg( kernel, 3, sizeof(cl_uint2), (void *)&inputDimensions));
	clErrChk(clSetKernelArg( kernel, 4, sizeof(cl_uint2), (void *)&maskDimensions));
	globalSize[0]=width*height;
}

int dpConvolution::execute(){
	err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);
	clErrChk(err); 
	if (err < 0)
		return -1;
	clFinish(queue);
	return 0;
}

void dpConvolution::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, width * height * sizeof(cl_uint),  output, 0, NULL, NULL));
	clFinish(queue);
}

void dpConvolution::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(inputBuffer));
	clErrChk(clReleaseMemObject(outputBuffer));
	clErrChk(clReleaseMemObject(maskBuffer));
	free(input);
	free(output);
	free(mask);
}

template<typename T> void dpConvolution::fillRandom(T * arrayPtr, const int width, const int height, const T rangeMin, const T rangeMax){
	srand(time(NULL));
	double range = double(rangeMax - rangeMin) + 1.0;
	/* random initialisation of input */
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			int index = i*width + j;
			arrayPtr[index] = rangeMin + T(range*rand()/(RAND_MAX + 1.0));
		}
	}
}

