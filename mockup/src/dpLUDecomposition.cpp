/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

#include "dpLUDecomposition.hpp"
#include "errorCheck.hpp"
#include <malloc.h>
#include <string.h>
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }
#define VECTOR_SIZE 4
#define SIZE effectiveDimension * effectiveDimension * sizeof(double)

dpLUDecomposition::dpLUDecomposition(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;
	name = "LUDecomposition";
	kernelString = "\n"
	"#ifdef KHR_DP_EXTENSION                                                                                                                                         \n"
	"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                                                                                   \n"
	"#else                                                                                                                                                           \n"
	"#pragma OPENCL EXTENSION cl_amd_fp64 : enable                                                                                                                   \n"
	"#endif                                                                                                                                                          \n"
	"                                                                                                                                                                \n"
	"#define VECTOR_SIZE 4                                                                                                                                           \n"
	"                                                                                                                                                                \n"
	"                                                                                                                                                                \n"
	"/* Kernel to decompose the matrix in LU parts                                                                                                                   \n"
	"    input taken from inlacematrix                                                                                                                               \n"
	"    param d tells which iteration of the kernel is executing                                                                                                    \n"
	"    output matrix U is generated in the inplacematrix itself                                                                                                    \n"
	"    output matrix L is generated in the LMatrix*/                                                                                                               \n"
	"                                                                                                                                                                \n"
	"__kernel void kernelLUDecompose(__global double4* LMatrix,                                                                                                      \n"
	"                           __global double4* inplaceMatrix,                                                                                                     \n"
	"                            int d,                                                                                                                              \n"
	"                            __local double* ratio)                                                                                                              \n"
	"{	                                                                                                                                                             \n"
	"    //get the global id of the work item                                                                                                                        \n"
	"    int y = get_global_id(1);                                                                                                                                   \n"
	"    int x = get_global_id(0);                                                                                                                                   \n"
	"    int lidx = get_local_id(0);                                                                                                                                 \n"
	"    int lidy = get_local_id(1);                                                                                                                                 \n"
	"    //the range in x axis is dimension / 4                                                                                                                      \n"
	"    int xdimension = get_global_size(0) + d / VECTOR_SIZE;                                                                                                      \n"
	"    int D = d % VECTOR_SIZE;                                                                                                                                    \n"
	"    if(get_local_id(0) == 0)                                                                                                                                    \n"
	"    {                                                                                                                                                           \n"
	"                                                                                                                                                                \n"
	"        //ratio needs to be calculated only once per workitem                                                                                                   \n"
	"        (D == 0) ? (ratio[lidy] = inplaceMatrix[ y * xdimension + d / VECTOR_SIZE].s0 / inplaceMatrix[ d * xdimension + d / VECTOR_SIZE].s0):1;                 \n"
	"        (D == 1) ? (ratio[lidy] = inplaceMatrix[ y * xdimension + d / VECTOR_SIZE].s1 / inplaceMatrix[ d * xdimension + d / VECTOR_SIZE].s1):1;                 \n"
	"        (D == 2) ? (ratio[lidy] = inplaceMatrix[ y * xdimension + d / VECTOR_SIZE].s2 / inplaceMatrix[ d * xdimension + d / VECTOR_SIZE].s2):1;                 \n"
	"        (D == 3) ? (ratio[lidy] = inplaceMatrix[ y * xdimension + d / VECTOR_SIZE].s3 / inplaceMatrix[ d * xdimension + d / VECTOR_SIZE].s3):1;                 \n"
	"    }                                                                                                                                                           \n"
	"                                                                                                                                                                \n"
	"    barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                               \n"
	"                                                                                                                                                                \n"
	"    //check which workitems need to be included for computation                                                                                                 \n"
	"    if(y >= d + 1 && ((x + 1) * VECTOR_SIZE) > d)                                                                                                               \n"
	"    {                                                                                                                                                           \n"
	"        double4 result;                                                                                                                                         \n"
	"                                                                                                                                                                \n"
	"        //the vectorized part begins here                                                                                                                       \n"
	"        {                                                                                                                                                       \n"
	"            result.s0 = inplaceMatrix[y * xdimension + x].s0 - ratio[lidy] * inplaceMatrix[ d * xdimension + x].s0;                                             \n"
	"            result.s1 = inplaceMatrix[y * xdimension + x].s1 - ratio[lidy] * inplaceMatrix[ d * xdimension + x].s1;                                             \n"
	"            result.s2 = inplaceMatrix[y * xdimension + x].s2 - ratio[lidy] * inplaceMatrix[ d * xdimension + x].s2;                                             \n"
	"            result.s3 = inplaceMatrix[y * xdimension + x].s3 - ratio[lidy] * inplaceMatrix[ d * xdimension + x].s3;                                             \n"
	"        }                                                                                                                                                       \n"
	"                                                                                                                                                                \n"
	"                                                                                                                                                                \n"
	"        if(x == d / VECTOR_SIZE)                                                                                                                                \n"
	"        {                                                                                                                                                       \n"
	"            (D == 0) ? (LMatrix[y * xdimension + x].s0 = ratio[lidy]) : (inplaceMatrix[y * xdimension + x].s0 = result.s0);                                     \n"
	"            (D == 1) ? (LMatrix[y * xdimension + x].s1 = ratio[lidy]) : (inplaceMatrix[y * xdimension + x].s1 = result.s1);                                     \n"
	"            (D == 2) ? (LMatrix[y * xdimension + x].s2 = ratio[lidy]) : (inplaceMatrix[y * xdimension + x].s2 = result.s2);                                     \n"
	"            (D == 3) ? (LMatrix[y * xdimension + x].s3 = ratio[lidy]) : (inplaceMatrix[y * xdimension + x].s3 = result.s3);                                     \n"
	"        }                                                                                                                                                       \n"
	"        else                                                                                                                                                    \n"
	"        {                                                                                                                                                       \n"
	"            inplaceMatrix[y * xdimension + x].s0 = result.s0;                                                                                                   \n"
	"            inplaceMatrix[y * xdimension + x].s1 = result.s1;                                                                                                   \n"
	"            inplaceMatrix[y * xdimension + x].s2 = result.s2;                                                                                                   \n"
	"            inplaceMatrix[y * xdimension + x].s3 = result.s3;                                                                                                   \n"
	"        }                                                                                                                                                       \n"
	"    }                                                                                                                                                           \n"
	"}                                                                                                                                                               \n"
	"                                                                                                                                                                \n"
	"                                                                                                                                                                \n"
	"/*	This function will combine L & U into 1 matrix                                                                                                               \n"
	"    param: inplace matrix contains the U matrix generated by above kernel                                                                                       \n"
	"    param: LMatrix contains L MAtrix                                                                                                                            \n"
	"    The kernel will combine them together as one matrix                                                                                                         \n"
	"    We ignore the diagonal elements of LMatrix during this as                                                                                                   \n"
	"    they will all be zero                                                                                                                                       \n"
	"    */                                                                                                                                                          \n"
	"                                                                                                                                                                \n"
	"                                                                                                                                                                \n"
	"__kernel void kernelLUCombine(__global double* LMatrix,                                                                                                         \n"
	"                         __global double* inplaceMatrix)                                                                                                        \n"
	"{                                                                                                                                                               \n"
	"    int i = get_global_id(1);                                                                                                                                   \n"
	"    int j = get_global_id(0);                                                                                                                                   \n"
	"    int gidx = get_group_id(0);                                                                                                                                 \n"
	"    int gidy = get_group_id(1);                                                                                                                                 \n"
	"    int dimension = get_global_size(0);                                                                                                                         \n"
	"    if(i>j )                                                                                                                                                    \n"
	"    {                                                                                                                                                           \n"
	"        int dimension = get_global_size(0);                                                                                                                     \n"
	"        inplaceMatrix[i * dimension + j] = LMatrix[i * dimension + j];                                                                                          \n"
	"    }                                                                                                                                                           \n"
	"}																																																																															 \n";
	
}
void dpLUDecomposition::init(int xLocal,int yLocal,int zLocal){
	effectiveDimension = 2048;
	
	localSize[0] = xLocal;
	localSize[1] = yLocal;
	localSize[2] = zLocal;
	
	dataParameters.push_back(effectiveDimension);
	dataParameters.push_back(effectiveDimension);
	dataNames.push_back("width");
	dataNames.push_back("height");
	
	if(effectiveDimension % VECTOR_SIZE != 0)
		effectiveDimension = effectiveDimension -(effectiveDimension % VECTOR_SIZE)+ VECTOR_SIZE;

	input = static_cast<double*>(memalign(4096, SIZE));
	matrixGPU = static_cast<double*>(memalign(4096, SIZE));

	//initialize with random double type elements
	generateMatrix(input, effectiveDimension,effectiveDimension);

	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, "-D KHR_DP_EXTENSION", NULL, NULL));
	// get a kernel object handle for a kernel with the given name
	kernelLUD = clCreateKernel(program, "kernelLUDecompose", &err); clErrChk(err);
	kernelCombine = clCreateKernel(program, "kernelLUCombine", &err); clErrChk(err);
}

void dpLUDecomposition::memoryCopyOut(){
	
	//Creating Buffers
	inplaceBuffer = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(double) * effectiveDimension * effectiveDimension,NULL,&err); clErrChk(err);
	inputBuffer2 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(double) * effectiveDimension * effectiveDimension,NULL,&err); clErrChk(err);
	
	//Copy data to buffers
	inMapPtr = clEnqueueMapBuffer(queue,inplaceBuffer,CL_FALSE,CL_MAP_WRITE,0,SIZE,0,NULL,NULL,&err); clErrChk(err);
	memcpy(inMapPtr, input, SIZE);
	clErrChk(clEnqueueUnmapMemObject(queue,inplaceBuffer,inMapPtr,0,NULL,NULL));
	clFinish(queue);
}

void dpLUDecomposition::plan(){
	globalSize[0]= effectiveDimension / VECTOR_SIZE;
	globalSize[1]= effectiveDimension;
	clSetKernelArg(kernelLUD,0,sizeof(cl_mem),(void *)&inputBuffer2);
	clSetKernelArg(kernelLUD,1,sizeof(cl_mem),(void *)&inplaceBuffer);
	
	clSetKernelArg(kernelCombine,0,sizeof(cl_mem),(void *)&inputBuffer2);
	clSetKernelArg(kernelCombine,1,sizeof(cl_mem),(void *)&inplaceBuffer);
}

int dpLUDecomposition::execute(){
	int workGroupArea = localSize[0]*localSize[1];
	size_t offset[2] = {0, 0};
	size_t localThreads[2] = {localSize[0] , localSize[1]};
	
	for(int index = 0; index < effectiveDimension - 1; ++index){
		/*
		*  code to find :
		*  1. new offset
		*  2. Local Work group size
		*  3. Global ND range
		*/
		if(index % VECTOR_SIZE == 0)
		{
			offset[0] = (index / VECTOR_SIZE);
			offset[1] = VECTOR_SIZE * (index / VECTOR_SIZE);

			if(!index)
			{
				globalSize[0] += 1;
				globalSize[1] += VECTOR_SIZE;
			}
			globalSize[0] -= 1;
			globalSize[1] -= VECTOR_SIZE;

			if(globalSize[0] <= (unsigned int)workGroupArea)
			{
				localThreads[0] = globalSize[0];
			}
			else
			{
				size_t temp = (int)workGroupArea;
				for(; temp > 1; temp--)
				{
					if(globalSize[0] % temp == 0)
					{
						break;
					}
				}
				localThreads[0] = temp;
			}

			if( globalSize[1] < workGroupArea / localThreads[0])
			{
				localThreads[1] = globalSize[1];
			}
			else
			{
				size_t temp = workGroupArea / localThreads[0];
				for(; temp > 1; temp--)
				{
					if(globalSize[1] % temp == 0)
					{
						break;
					}
				}
				localThreads[1] = temp;
			}
		}

		clErrChk(clSetKernelArg(kernelLUD,2,sizeof(cl_uint),&index));
		clErrChk(clSetKernelArg(kernelLUD,3,sizeof(cl_double) * localThreads[1],NULL));
		err=clEnqueueNDRangeKernel(queue,kernelLUD,2,offset,globalSize,localThreads,0,NULL,NULL);
		clErrChk(err);
		if(err<0)
			return -1;
		clFinish(queue);

	}

	/*
	* This will combine the L & U matrices at the GPU side
	* so that they can be bought in CPU space as a single matrix
	*/

	globalSize[0] = effectiveDimension;
	globalSize[1] = effectiveDimension;
	err= clEnqueueNDRangeKernel(queue,kernelCombine,2,NULL,globalSize,NULL,0,NULL,NULL);
	clErrChk(err);
	if(err<0)
		return -1;
	clFinish(queue);
	return 0;
}

void dpLUDecomposition::memoryCopyIn(){
	outMapPtr = clEnqueueMapBuffer(queue,inplaceBuffer,CL_FALSE,CL_MAP_READ,0,SIZE,0,NULL,NULL,&err); clErrChk(err);
	memcpy(matrixGPU, outMapPtr, SIZE);
	clErrChk(clEnqueueUnmapMemObject(queue,inplaceBuffer,outMapPtr,0,NULL,NULL));
	clFinish(queue);
}

void dpLUDecomposition::cleanUp(){
	// Releases OpenCL resources (Context, Memory etc.)
	clErrChk(clReleaseKernel(kernelLUD));
	clErrChk(clReleaseKernel(kernelCombine));
	clErrChk(clReleaseMemObject(inplaceBuffer));
	clErrChk(clReleaseMemObject(inputBuffer2));
	
	// release program resources (input memory etc.)
	free(input);
	free(matrixGPU);
	clReleaseProgram(program);
}

void dpLUDecomposition::generateMatrix(double *A, int height, int width){
	int i, j;
	srand(time(NULL));
	for (j=0; j<height; j++){//rows in A
		for (i=0; i<width; i++){//cols in A
			A[i + width*j] = rand() / (RAND_MAX/99999.9 + 1);
		}
	}
}






