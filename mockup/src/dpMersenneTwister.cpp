/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

/*
 * Implemented Gaussian Random Number Generator. SIMD-oriented Fast Mersenne
 * Twister(SFMT) used to generate random numbers and Box mullar transformation used
 * to convert them to Gaussian random numbers.
 * 
 * The SFMT algorithm details could be found at 
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/index.html
 *
 * Box-Muller Transformation 
 * http://mathworld.wolfram.com/Box-MullerTransformation.html
 *
 * One invocation of this kernel(gaussianRand), i.e one work thread writes
 * mulFactor output values.
 */

#include "dpMersenneTwister.hpp"
#include "errorCheck.hpp"
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpMersenneTwister::dpMersenneTwister(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;
	name = "MersenneTwister";
	
	kernelString = "\n"
"class gaussianRandom                                                                           \n"
"{                                                                                              \n"
"protected:                                                                                     \n"
"	const __global uint4 *seedArray;                                                              \n"
"    uint width;                                                                                \n"
"    uint mulFactor;                                                                            \n"
"    __global float4 *gaussianRand;                                                             \n"
"	                                                                                              \n"
"    uint4 temp[8];                                                                             \n"
"                                                                                               \n"
"    size_t xPid;                                                                               \n"
"    size_t yPid;                                                                               \n"
"                                                                                               \n"
"    uint4 state1;                                                                              \n"
"    uint4 state2;                                                                              \n"
"    uint4 state3;                                                                              \n"
"    uint4 state4;                                                                              \n"
"    uint4 state5;                                                                              \n"
"                                                                                               \n"
"    uint4 mask4;                                                                               \n"
"    uint4 thirty4;                                                                             \n"
"    uint4 one4;                                                                                \n"
"    uint4 two4;                                                                                \n"
"    uint4 three4;                                                                              \n"
"    uint4 four4;                                                                               \n"
"                                                                                               \n"
"    uint4 r1;                                                                                  \n"
"    uint4 r2;                                                                                  \n"
"                                                                                               \n"
"    uint4 a;                                                                                   \n"
"    uint4 b;                                                                                   \n"
"                                                                                               \n"
"    uint4 e;                                                                                   \n"
"    uint4 f;                                                                                   \n"
"                                                                                               \n"
"    unsigned int thirteen;                                                                     \n"
"    unsigned int fifteen;                                                                      \n"
"    unsigned int shift;                                                                        \n"
"                                                                                               \n"
"    unsigned int mask11;                                                                       \n"
"    unsigned int mask12;                                                                       \n"
"    unsigned int mask13;                                                                       \n"
"    unsigned int mask14;                                                                       \n"
"                                                                                               \n"
"    size_t actualPos;                                                                          \n"
"                                                                                               \n"
"    float one;                                                                                 \n"
"    float intMax;                                                                              \n"
"    float PI;                                                                                  \n"
"    float two;                                                                                 \n"
"                                                                                               \n"
"    float4 r;                                                                                  \n"
"    float4 phi;                                                                                \n"
"                                                                                               \n"
"    float4 temp1;                                                                              \n"
"    float4 temp2;                                                                              \n"
"	                                                                                              \n"
"	unsigned int invshift;                                                                        \n"
"	                                                                                              \n"
"	uint4 lshift128(uint4);                                                                       \n"
"	uint4 rshift128(uint4);                                                                       \n"
"	void evaluate(uint4 num_1,                                                                    \n"
"                  uint4 num_2,                                                                 \n"
"                  uint4 num_3,                                                                 \n"
"                  uint4 num_4);                                                                \n"
"                                                                                               \n"
"public:                                                                                        \n"
"	void initial(const __global uint4*, uint, uint, __global float4*);                            \n"
"	void run();                                                                                   \n"
"	                                                                                              \n"
"                                                                                               \n"
"};                                                                                             \n"
"                                                                                               \n"
"void                                                                                           \n"
"gaussianRandom::initial(const __global uint4 *seedArray,                                       \n"
"            uint width,                                                                        \n"
"            uint mulFactor,                                                                    \n"
"            __global float4 *gaussianRand)                                                     \n"
"{                                                                                              \n"
"	this->seedArray = seedArray;                                                                  \n"
"    this->width = width;                                                                       \n"
"    this->mulFactor = mulFactor ;                                                              \n"
"    this->gaussianRand = gaussianRand ;                                                        \n"
"                                                                                               \n"
"	xPid = get_global_id(0);                                                                      \n"
"    yPid = get_global_id(1);                                                                   \n"
"	                                                                                              \n"
"	state1 = seedArray[yPid *width + xPid];                                                       \n"
"                                                                                               \n"
"    mask4 = (uint4)(1812433253u);                                                              \n"
"    thirty4 = (uint4)(30u);                                                                    \n"
"    one4 = (uint4)(1u);                                                                        \n"
"    two4 = (uint4)(2u);                                                                        \n"
"    three4 = (uint4)(3u);                                                                      \n"
"    four4 = (uint4)(4u);                                                                       \n"
"                                                                                               \n"
"    thirteen  = 13u;                                                                           \n"
"    fifteen = 15u;                                                                             \n"
"    shift = 8u * 3u;                                                                           \n"
"                                                                                               \n"
"    mask11 = 0xfdff37ffu;                                                                      \n"
"    mask12 = 0xef7f3f7du;                                                                      \n"
"    mask13 = 0xff777b7du;                                                                      \n"
"    mask14 = 0x7ff7fb2fu;                                                                      \n"
"                                                                                               \n"
"    one = 1.0f;                                                                                \n"
"    intMax = 4294967296.0f;                                                                    \n"
"    PI = 3.14159265358979f;                                                                    \n"
"    two = 2.0f;                                                                                \n"
"	                                                                                              \n"
"	invshift =  32u - shift;                                                                      \n"
"	                                                                                              \n"
"	state2 = mask4 * (state1 ^ (state1 >> thirty4)) + one4;                                       \n"
"    state3 = mask4 * (state2 ^ (state2 >> thirty4)) + two4;                                    \n"
"    state4 = mask4 * (state3 ^ (state3 >> thirty4)) + three4;                                  \n"
"    state5 = mask4 * (state4 ^ (state4 >> thirty4)) + four4;                                   \n"
"}                                                                                              \n"
"                                                                                               \n"
"void                                                                                           \n"
"gaussianRandom::run()                                                                          \n"
"{                                                                                              \n"
"	uint i = 0;                                                                                   \n"
"    for(i = 0; i < mulFactor; ++i)                                                             \n"
"    {                                                                                          \n"
"        switch(i)                                                                              \n"
"        {                                                                                      \n"
"            case 0:                                                                            \n"
"				evaluate(state4, state5, state1, state3);                                               \n"
"                break;                                                                         \n"
"            case 1:                                                                            \n"
"				evaluate(r2, temp[0], state2, state4);                                                  \n"
"                break;                                                                         \n"
"            case 2:                                                                            \n"
"				evaluate(r2, temp[1], state3, state5);                                                  \n"
"                break;                                                                         \n"
"            case 3:                                                                            \n"
"				evaluate(r2, temp[2], state4, state1);                                                  \n"
"                break;                                                                         \n"
"            case 4:                                                                            \n"
"				evaluate(r2, temp[3], state5, state2);                                                  \n"
"                break;                                                                         \n"
"            case 5:                                                                            \n"
"				evaluate(r2, temp[4], temp[0], temp[2]);                                                \n"
"                break;                                                                         \n"
"            case 6:                                                                            \n"
"				evaluate(r2, temp[5], temp[1], temp[3]);                                                \n"
"                break;                                                                         \n"
"            case 7:                                                                            \n"
"				evaluate(r2, temp[6], temp[2], temp[4]);                                                \n"
"                break;                                                                         \n"
"            default:                                                                           \n"
"                break;                                                                         \n"
"                                                                                               \n"
"        }                                                                                      \n"
"                                                                                               \n"
"        e = lshift128(a);                                                                      \n"
"        f = rshift128(r1);                                                                     \n"
"                                                                                               \n"
"        temp[i].x = a.x ^ e.x ^ ((b.x >> thirteen) & mask11) ^ f.x ^ (r2.x << fifteen);        \n"
"        temp[i].y = a.y ^ e.y ^ ((b.y >> thirteen) & mask12) ^ f.y ^ (r2.y << fifteen);        \n"
"        temp[i].z = a.z ^ e.z ^ ((b.z >> thirteen) & mask13) ^ f.z ^ (r2.z << fifteen);        \n"
"        temp[i].w = a.w ^ e.w ^ ((b.w >> thirteen) & mask14) ^ f.w ^ (r2.w << fifteen);        \n"
"    }                                                                                          \n"
"                                                                                               \n"
"    actualPos = (yPid * width + xPid) * mulFactor;                                             \n"
"                                                                                               \n"
"    for(i = 0; i < mulFactor / 2; ++i)                                                         \n"
"    {                                                                                          \n"
"        temp1 = convert_float4(temp[i]) * one / intMax;                                        \n"
"        temp2 = convert_float4(temp[i + 1]) * one / intMax;                                    \n"
"                                                                                               \n"
"        // Applying Box Mullar Transformations.                                                \n"
"        r = sqrt((-two) * log(temp1));                                                         \n"
"        phi  = two * PI * temp2;                                                               \n"
"        gaussianRand[actualPos + i * 2 + 0] = r * cos(phi);                                    \n"
"        gaussianRand[actualPos + i * 2 + 1] = r * sin(phi);                                    \n"
"    }                                                                                          \n"
"                                                                                               \n"
"}                                                                                              \n"
"                                                                                               \n"
"uint4                                                                                          \n"
"gaussianRandom::lshift128(uint4 input)                                                         \n"
"{                                                                                              \n"
"    uint4 temp;                                                                                \n"
"	                                                                                              \n"
"    temp.x = input.x << shift;                                                                 \n"
"    temp.y = (input.y << shift) | (input.x >> invshift);                                       \n"
"    temp.z = (input.z << shift) | (input.y >> invshift);                                       \n"
"    temp.w = (input.w << shift) | (input.z >> invshift);                                       \n"
"                                                                                               \n"
"    return temp;                                                                               \n"
"}                                                                                              \n"
"                                                                                               \n"
"uint4                                                                                          \n"
"gaussianRandom::rshift128(uint4 input)                                                         \n"
"{                                                                                              \n"
"    uint4 temp;                                                                                \n"
"                                                                                               \n"
"    temp.w = input.w >> shift;                                                                 \n"
"    temp.z = (input.z >> shift) | (input.w << invshift);                                       \n"
"    temp.y = (input.y >> shift) | (input.z << invshift);                                       \n"
"    temp.x = (input.x >> shift) | (input.y << invshift);                                       \n"
"                                                                                               \n"
"    return temp;                                                                               \n"
"}                                                                                              \n"
"                                                                                               \n"
"void                                                                                           \n"
"gaussianRandom::evaluate(uint4 num_1,                                                          \n"
"                         uint4 num_2,                                                          \n"
"                         uint4 num_3,                                                          \n"
"                         uint4 num_4)                                                          \n"
"{                                                                                              \n"
"	this->r1 = num_1;                                                                             \n"
"	this->r2 = num_2;                                                                             \n"
"	this->a = num_3;                                                                              \n"
"	this->b = num_4;                                                                              \n"
"}                                                                                              \n"
"                                                                                               \n"
"__kernel                                                                                       \n"
"void gaussianRand(const __global uint4 *seedArray,                                             \n"
"            uint width,                                                                        \n"
"            uint mulFactor,                                                                    \n"
"            __global float4 *gaussianRand)                                                     \n"
"{                                                                                              \n"
"                                                                                               \n"
"    gaussianRandom GaussianRand;                                                               \n"
"	GaussianRand.initial(seedArray, width, mulFactor, gaussianRand);                              \n"
"	GaussianRand.run();                                                                           \n"
"	                                                                                              \n"
"}                                                                                              \n"
"                                                                                               \n";
}

void dpMersenneTwister::init(int xLocal,int yLocal,int zLocal){

	localSize[0] = xLocal;
	localSize[1] = yLocal;
	localSize[2] = zLocal;
	
	width = 4096;
	height = 4096;
	mulFactor = 1;
	numRands = width * height; //this is total number of numbers to generate, it should be square of width and height;
	
	dataParameters.push_back(numRands);
	dataNames.push_back("nRandomNumbers");
	
	// Allocate and init memory used by host
	seeds = (cl_uint*)memalign(16, width * height * sizeof(cl_uint4));
	deviceResult = (cl_float *) malloc(width * height * mulFactor * sizeof(cl_float4));
	for(int i = 0; i < width * height * 4; ++i)
		seeds[i] = (unsigned int)rand();
	memset((void*)deviceResult, 0, width * height * mulFactor * sizeof(cl_float4));
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	//this build program with these arguments will not work on NVIDIA devices!
	//src: http://stackoverflow.com/questions/14991591/how-to-use-templates-with-opencl
	clErrChk(clBuildProgram(program, 0, NULL, "-x clc++", NULL, NULL));
	kernel = clCreateKernel(program, "gaussianRand", &err); clErrChk(err);

	seedsBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * sizeof(cl_float4), 0, &err); clErrChk(err);
	resultBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(cl_float4) * mulFactor, NULL, &err); clErrChk(err);
	
}
void dpMersenneTwister::memoryCopyOut(){
	clErrChk(clEnqueueWriteBuffer(queue, seedsBuf, CL_FALSE, 0, width * height * sizeof(cl_float4), seeds, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}
void dpMersenneTwister::plan(){
		globalSize[0] = width;
		globalSize[1] = height;
    clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&seedsBuf));
    clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_uint), (void*)&width));
    clErrChk(clSetKernelArg(kernel, 2, sizeof(cl_uint), (void*)&mulFactor));
    clErrChk(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&resultBuf));
}

void dpMersenneTwister::execute(){
	clErrChk(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpMersenneTwister::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, resultBuf, CL_FALSE, 0, width * height * mulFactor * sizeof(cl_float4), deviceResult, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpMersenneTwister::cleanUp(){
	clErrChk(clReleaseMemObject(seedsBuf));
	clErrChk(clReleaseMemObject(resultBuf));
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	free(deviceResult);
	free(seeds);
}


