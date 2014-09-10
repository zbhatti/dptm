/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/
#include "dpNBody.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }
#include <cmath>
#include <malloc.h>
#include <string.h>

dpNBody::dpNBody(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;
	
	name = "NBody";
	kernelString = "\n"
	"#define UNROLL_FACTOR  8                                                                          \n"
	"__kernel                                                                                          \n"
	"void nbody_sim(__global float4* pos, __global float4* vel                                         \n"
	"		,int numBodies ,float deltaTime, float epsSqr                                                  \n"
	"		,__global float4* newPosition, __global float4* newVelocity) {                                 \n"
	"                                                                                                  \n"
	"    unsigned int gid = get_global_id(0);                                                          \n"
	"    float4 myPos = pos[gid];                                                                      \n"
	"    float4 acc = (float4)0.0f;                                                                    \n"
	"                                                                                                  \n"
	"                                                                                                  \n"
	"    int i = 0;                                                                                    \n"
	"    for (; (i+UNROLL_FACTOR) < numBodies; ) {                                                     \n"
	"#pragma unroll UNROLL_FACTOR                                                                      \n"
	"        for(int j = 0; j < UNROLL_FACTOR; j++,i++) {                                              \n"
	"            float4 p = pos[i];                                                                    \n"
	"            float4 r;                                                                             \n"
	"            r.xyz = p.xyz - myPos.xyz;                                                            \n"
	"            float distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;                                \n"
	"                                                                                                  \n"
	"            float invDist = 1.0f / sqrt(distSqr + epsSqr);                                        \n"
	"            float invDistCube = invDist * invDist * invDist;                                      \n"
	"            float s = p.w * invDistCube;                                                          \n"
	"                                                                                                  \n"
	"            // accumulate effect of all particles                                                 \n"
	"            acc.xyz += s * r.xyz;                                                                 \n"
	"        }                                                                                         \n"
	"    }                                                                                             \n"
	"    for (; i < numBodies; i++) {                                                                  \n"
	"        float4 p = pos[i];                                                                        \n"
	"                                                                                                  \n"
	"        float4 r;                                                                                 \n"
	"        r.xyz = p.xyz - myPos.xyz;                                                                \n"
	"        float distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;                                    \n"
	"                                                                                                  \n"
	"        float invDist = 1.0f / sqrt(distSqr + epsSqr);                                            \n"
	"        float invDistCube = invDist * invDist * invDist;                                          \n"
	"        float s = p.w * invDistCube;                                                              \n"
	"                                                                                                  \n"
	"        // accumulate effect of all particles                                                     \n"
	"        acc.xyz += s * r.xyz;                                                                     \n"
	"    }                                                                                             \n"
	"                                                                                                  \n"
	"    float4 oldVel = vel[gid];                                                                     \n"
	"                                                                                                  \n"
	"    // updated position and velocity                                                              \n"
	"    float4 newPos;                                                                                \n"
	"    newPos.xyz = myPos.xyz + oldVel.xyz * deltaTime + acc.xyz * 0.5f * deltaTime * deltaTime;     \n"
	"    newPos.w = myPos.w;                                                                           \n"
	"                                                                                                  \n"
	"    float4 newVel;                                                                                \n"
	"    newVel.xyz = oldVel.xyz + acc.xyz * deltaTime;                                                \n"
	"    newVel.w = oldVel.w;                                                                          \n"
	"                                                                                                  \n"
	"    // write to global memory                                                                     \n"
	"    newPosition[gid] = newPos;                                                                    \n"
	"    newVelocity[gid] = newVel;                                                                    \n"
	"}                                                                                                 \n";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "nbody_sim", &err); clErrChk(err);
	
}

void dpNBody::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
	
	//for (int i=0; pow(2,i)*sizeof(cl_float4)/(float) 1048576 <dataMB; i++){
	//	numBodies=pow(2,i);
	//}
	
	numBodies = 1048576*dataMB/sizeof(cl_float4);
	
	MB = numBodies * sizeof(cl_float4)/1048576;
	
}

void dpNBody::init(){
	delT=0.005f;
	espSqr=500.0f;
	nSteps = 1;
	
	dataParameters.push_back(numBodies);
	dataParameters.push_back(delT);
	dataParameters.push_back(nSteps);
	dataNames.push_back("nBodies");
	dataNames.push_back("timeStep");
	dataNames.push_back("nSteps");
	
	initPos = (cl_float*)malloc(numBodies * sizeof(cl_float4));
	if(!initPos)
		fprintf(stderr, "initpos not allocated in dpNBody\n");
	// initialization of inputs
	for(int i = 0; i < numBodies; ++i){
		int index = 4 * i;

		// First 3 values are position in x,y and z direction
		for(int j = 0; j < 3; ++j){
			initPos[index + j] = random(3, 50);
		}
		
		// Mass value
		initPos[index + 3] = random(1, 1000);
	}
	
}

void dpNBody::memoryCopyOut(){
	size_t bufferSize = numBodies * sizeof(cl_float4);
	for (int i = 0; i < 2; i++){
		particlePos[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, 0, &err); clErrChk(err);
		particleVel[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, 0, &err); clErrChk(err);
	}

	clErrChk(clEnqueueWriteBuffer(queue,particlePos[0],CL_TRUE,0,bufferSize,initPos,0,0,NULL));
	
	// Initialize the velocity buffer to zero
	float* p = (float*) clEnqueueMapBuffer(	queue, particleVel[0], CL_TRUE, CL_MAP_WRITE,
																					0, bufferSize, 0, NULL, NULL, &err); clErrChk(err);
	memset(p, 0, bufferSize);
	clErrChk(clEnqueueUnmapMemObject(queue, particleVel[0], p, 0, NULL,NULL));
	clFinish(queue);
}

void dpNBody::plan(){
	clErrChk(clSetKernelArg(kernel,2,sizeof(cl_int),(void *)&numBodies));
	clErrChk(clSetKernelArg(kernel,3,sizeof(cl_float),(void *)&delT));
	clErrChk(clSetKernelArg(kernel,4,sizeof(cl_float),(void *)&espSqr));
	globalSize[0] = numBodies;
}

int dpNBody::execute(){
	int currentPosBufferIndex = 0;

	for (int i = 0; i < nSteps; i++){
		int currentBuffer = currentPosBufferIndex;
		int nextBuffer = (currentPosBufferIndex+1)%2;

		// Particle positions
		clErrChk(clSetKernelArg(kernel,0,sizeof(cl_mem),(void*) (particlePos+currentBuffer)));
		clErrChk(clSetKernelArg(kernel,1,sizeof(cl_mem),(void *) (particleVel+currentBuffer)));
		clErrChk(clSetKernelArg(kernel,5,sizeof(cl_mem),(void*) (particlePos+nextBuffer)));
		clErrChk(clSetKernelArg(kernel,6,sizeof(cl_mem),(void*) (particleVel+nextBuffer)));

		err=clEnqueueNDRangeKernel(queue,kernel,1,NULL,globalSize,localSize,0,NULL,NULL);
		clErrChk(err);
		if(err<0)
			return -1;
		
		currentPosBufferIndex = nextBuffer;
	}
	clFinish(queue);
	return 0;
}

void dpNBody::memoryCopyIn(){
	
}

void dpNBody::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	for (int i = 0; i < 2; i++){
		clErrChk(clReleaseMemObject(particlePos[i]));
		clErrChk(clReleaseMemObject(particleVel[i]));
	}

	free(initPos);
}

float dpNBody::random(float randMax, float randMin){
    float result;
    result =(float)rand() / (float)RAND_MAX;

    return ((1.0f - result) * randMin + result *randMax);
}


