/*******************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/
#include "dpFluidSimulation.hpp"
#include "errorCheck.hpp"
#include <malloc.h>
#include <string.h>
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }
#define ERR fprintf(stderr,"%d\n", __LINE__);

dpFluidSimulation::dpFluidSimulation(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;
	
	name = "FluidSimulation";
	kernelString = "\n"
	"#ifdef KHR_DP_EXTENSION                                                                            \n"
	"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                      \n"
	"#else                                                                                              \n"
	"#pragma OPENCL EXTENSION cl_amd_fp64 : enable                                                      \n"
	"#endif                                                                                             \n"
	"                                                                                                   \n"
	"double computefEq(double rho, double weight, double2 dir, double2 u)                        \n"
	"{                                                                                                  \n"
	"    double u2 = (u.x * u.x) + (u.y * u.y);		//x^2 + y^2                                           \n"
	"    double eu = (dir.x * u.x) + (dir.y * u.y);	                                                    \n"
	"    return rho * weight * (1.0f + (3.0f * eu) + (4.5f * eu * eu) - (1.5f * u2));                   \n"
	"}                                                                                                  \n"
	"                                                                                                   \n"
	"__kernel void lbm(__global double *if0, __global double *of0,                                      \n"
	"                  __global double4 *if1234, __global double4 *of1234,                              \n"
	"                  __global double4 *if5678, __global double4 *of5678,                              \n"
	"                  __global bool *type,	// This will only work for sizes <= 512 x 512 as constant buffer is only 64KB \n"
	"                  double8 dirX,  double8 dirY,	//Directions is (0, 0) for 0                        \n"
	"                  __constant double weight[9],	//Directions : 0, 1, 2, 3, 4, 5, 6, 7, 8            \n"
	"                  double omega,                                                                    \n"
	"                  __global double2 *velocityBuffer)                                                \n"
	"{                                                                                                  \n"
	"    uint2 id = (uint2)(get_global_id(0), get_global_id(1));                                        \n"
	"    uint width = get_global_size(0);                                                               \n"
	"    uint pos = id.x + width * id.y;                                                                \n"
	"                                                                                                   \n"
	"    // Read input distributions                                                                    \n"
	"    double f0 = if0[pos];                                                                          \n"
	"    double4 f1234 = if1234[pos];                                                                   \n"
	"    double4 f5678 = if5678[pos];                                                                   \n"
	"                                                                                                   \n"
	"                                                                                                   \n"
	"    double rho;	//Density                                                                         \n"
	"    double2 u;	//Velocity                                                                          \n"
	"                                                                                                   \n"
	"    // Collide                                                                                     \n"
	"    //boundary                                                                                     \n"
	"    if(type[pos])                                                                                  \n"
	"    {                                                                                              \n"
	"        // Swap directions by swizzling                                                            \n"
	"        f1234.xyzw = f1234.zwxy;                                                                   \n"
	"        f5678.xyzw = f5678.zwxy;                                                                   \n"
	"                                                                                                   \n"
	"        rho = 0;                                                                                   \n"
	"        u = (double2)(0, 0);                                                                       \n"
	"    }                                                                                              \n"
	"    //fluid                                                                                        \n"
	"    else                                                                                           \n"
	"    {                                                                                              \n"
	"        // Compute rho and u                                                                       \n"
	"        // Rho is computed by doing a reduction on f                                               \n"
	"        double4 temp = f1234 + f5678;                                                              \n"
	"        temp.lo += temp.hi;                                                                        \n"
	"        rho = temp.x + temp.y;                                                                     \n"
	"        rho += f0;                                                                                 \n"
	"                                                                                                   \n"
	"        // Compute velocity                                                                        \n"
	"        u.x = (dot(f1234, dirX.lo) + dot(f5678, dirX.hi)) / rho;                                   \n"
	"        u.y = (dot(f1234, dirY.lo) + dot(f5678, dirY.hi)) / rho;                                   \n"
	"                                                                                                   \n"
	"        double4 fEq1234;	// Stores feq                                                             \n"
	"        double4 fEq5678;                                                                           \n"
	"        double fEq0;                                                                               \n"
	"                                                                                                   \n"
	"        // Compute fEq                                                                             \n"
	"        fEq0 = computefEq(rho, weight[0], (double2)(0, 0), u);                                     \n"
	"        fEq1234.x = computefEq(rho, weight[1], (double2)(dirX.s0, dirY.s0), u);                    \n"
	"        fEq1234.y = computefEq(rho, weight[2], (double2)(dirX.s1, dirY.s1), u);                    \n"
	"        fEq1234.z = computefEq(rho, weight[3], (double2)(dirX.s2, dirY.s2), u);                    \n"
	"        fEq1234.w = computefEq(rho, weight[4], (double2)(dirX.s3, dirY.s3), u);                    \n"
	"        fEq5678.x = computefEq(rho, weight[5], (double2)(dirX.s4, dirY.s4), u);                    \n"
	"        fEq5678.y = computefEq(rho, weight[6], (double2)(dirX.s5, dirY.s5), u);                    \n"
	"        fEq5678.z = computefEq(rho, weight[7], (double2)(dirX.s6, dirY.s6), u);                    \n"
	"        fEq5678.w = computefEq(rho, weight[8], (double2)(dirX.s7, dirY.s7), u);                    \n"
	"                                                                                                   \n"
	"        f0 = (1 - omega) * f0 + omega * fEq0;                                                      \n"
	"        f1234 = (1 - omega) * f1234 + omega * fEq1234;                                             \n"
	"        f5678 = (1 - omega) * f5678 + omega * fEq5678;                                             \n"
	"    }                                                                                              \n"
	"                                                                                                   \n"
	"    velocityBuffer[pos] = u;                                                                       \n"
	"                                                                                                   \n"
	"    // Propagate                                                                                   \n"
	"    // New positions to write (Each thread will write 8 values)                                    \n"
	"    int8 nX = (int8) ( id.x + dirX.s0, id.x + dirX.s1, id.x + dirX.s2, id.x + dirX.s3,             \n"
	"											  id.x + dirX.s4, id.x + dirX.s5, id.x + dirX.s6, id.x + dirX.s7 );           \n"
	"	   int8 nY = (int8) ( id.y + dirY.s0, id.y + dirY.s1, id.y + dirY.s2, id.y + dirY.s3,             \n"
	"												id.y + dirY.s4, id.y + dirY.s5, id.y + dirY.s6, id.y + dirY.s7 );           \n"
	"    int8 nPos = 	(int8)(nX.s0 + width * nY.s0, nX.s1 + width * nY.s1,                              \n"
	"									 			 nX.s2 + width * nY.s2, nX.s3 + width * nY.s3,                              \n"
	"									 			 nX.s4 + width * nY.s4, nX.s5 + width * nY.s5,                              \n"
	"									 			 nX.s6 + width * nY.s6, nX.s7 + width * nY.s7);                             \n"
	"                                                                                                   \n"
	"    // Write center distrivution to thread's location                                              \n"
	"    of0[pos] = f0;                                                                                 \n"
	"                                                                                                   \n"
	"    int t1 = id.x < get_global_size(0) - 1; // Not on Right boundary                               \n"
	"    int t4 = id.y > 0;                      // Not on Upper boundary                               \n"
	"    int t3 = id.x > 0;                      // Not on Left boundary                                \n"
	"    int t2 = id.y < get_global_size(1) - 1; // Not on lower boundary                               \n"
	"                                                                                                   \n"
	"    // Propagate to right cell                                                                     \n"
	"    if(t1)                                                                                         \n"
	"        of1234[nPos.s0].x = f1234.x;                                                               \n"
	"                                                                                                   \n"
	"    // Propagate to Lower cell                                                                     \n"
	"    if(t2)                                                                                         \n"
	"        of1234[nPos.s1].y = f1234.y;                                                               \n"
	"                                                                                                   \n"
	"    // Propagate to left cell                                                                      \n"
	"    if(t3)                                                                                         \n"
	"        of1234[nPos.s2].z = f1234.z;                                                               \n"
	"                                                                                                   \n"
	"    // Propagate to Upper cell                                                                     \n"
	"    if(t4)                                                                                         \n"
	"        of1234[nPos.s3].w = f1234.w;                                                               \n"
	"                                                                                                   \n"
	"    // Propagate to Lower-Right cell                                                               \n"
	"    if(t1 && t2)                                                                                   \n"
	"        of5678[nPos.s4].x = f5678.x;                                                               \n"
	"                                                                                                   \n"
	"    // Propogate to Lower-Left cell                                                                \n"
	"    if(t2 && t3)                                                                                   \n"
	"        of5678[nPos.s5].y = f5678.y;                                                               \n"
	"                                                                                                   \n"
	"    // Propagate to Upper-Left cell                                                                \n"
	"    if(t3 && t4)                                                                                   \n"
	"        of5678[nPos.s6].z = f5678.z;                                                               \n"
	"                                                                                                   \n"
	"    // Propagate to Upper-Right cell                                                               \n"
	"    if(t4 && t1)                                                                                   \n"
	"        of5678[nPos.s7].w = f5678.w;                                                               \n"
	"}                                                                                                  \n";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	err = clBuildProgram(program, 0, NULL, "-D KHR_DP_EXTENSION", NULL, NULL); clErrChk(err);	
	programCheck(err, context, program);
	kernel = clCreateKernel(program, "lbm", &err); clErrChk(err);
	
}


// Directions (global)
double e[9][2] = {{0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}, {1,1}, {-1,1}, {-1,-1}, {1,-1}};

// Weights (global)
cl_double w[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

// Omega (global)
const double omega = 1.2f;

void dpFluidSimulation::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0]= xLocal;
	localSize[1]= yLocal;
	localSize[2]= 1;
	
	//for (int i =0; pow(2,i)*pow(2,i)*sizeof(cl_double)/(float) 1048576 <= dataMB;i++){
	//	dims[0] = pow(2,i);
	//	dims[1] = pow(2,i);
	//}
	
	dims[0] = (int)sqrt(1048576*dataMB/(sizeof(cl_double)));
	dims[1] = dims[0];
	
	MB=dims[0]*dims[1]*sizeof(cl_double)/(float) 1048576;
}


void dpFluidSimulation::init(){
	iterations = 5;
	
	dataParameters.push_back(dims[0]);
	dataParameters.push_back(dims[1]);
	dataParameters.push_back(iterations);
	dataNames.push_back("width");
	dataNames.push_back("height");
	dataNames.push_back("nSteps");
	
	//INIT:
	size_t temp = dims[0] * dims[1];
	
	// Allocate memory for host buffers
	h_if0 = (cl_double*)malloc(sizeof(cl_double) * temp);
	h_if1234 = (cl_double*)malloc(sizeof(cl_double4) * temp);
	h_if5678 = (cl_double*)malloc(sizeof(cl_double4) * temp);
	h_of0 = (cl_double*)malloc(sizeof(cl_double) * temp);
	h_of1234 = (cl_double*)malloc(sizeof(cl_double4) * temp);
	h_of5678 = (cl_double*)malloc(sizeof(cl_double4) * temp);
	h_type = (cl_bool*)malloc(sizeof(cl_bool) * temp);
	rho = (cl_double*)malloc(sizeof(cl_double) * temp);
	u = (cl_double2*)malloc(sizeof(cl_double2) * temp);
	if (!h_if0|| !h_if1234 || !h_if5678 || !h_of0 || !h_of1234 || !h_of5678 || !h_type || !rho || !u)
		fprintf(stderr,"memory allocation error on host");
	reset();
	
}

void dpFluidSimulation::memoryCopyOut(){
	//MEMORY COPY OUT:
	size_t temp = dims[0] * dims[1];
	d_if0 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_double) * temp,0,&err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(queue,d_if0,1,0,sizeof(cl_double) * temp,h_if0,0, 0, 0));
	d_if1234 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_double4) * temp,0,&err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(queue,d_if1234,1,0,sizeof(cl_double4) * temp,h_if1234,0, 0, 0));
	d_if5678 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_double4) * temp,0,&err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(queue,d_if5678,1,0,sizeof(cl_double4) * temp,h_if5678,0, 0, 0));
	d_of0 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_double) * temp,0,&err); clErrChk(err);
	d_of1234 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_double4) * temp,0,&err); clErrChk(err);
	d_of5678 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(cl_double4) * temp,0,&err); clErrChk(err);
	clErrChk(clEnqueueCopyBuffer(queue,d_if0,d_of0,0, 0, sizeof(cl_double) * temp,0, 0, 0));
	clErrChk(clEnqueueCopyBuffer(queue,d_if1234,d_of1234,0, 0, sizeof(cl_double4) * temp,0, 0, 0));
	clErrChk(clEnqueueCopyBuffer(queue,d_if5678,d_of5678,0, 0, sizeof(cl_double4) * temp,0, 0, 0));

	//Constant arrays
	type = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_bool) * temp,0,&err); clErrChk(err);
	weight = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(cl_double) * 9,0,&err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(queue,weight,1, 0, sizeof(cl_double) * 9,w,0, 0, 0));
	velocity = clCreateBuffer(context,CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,sizeof(cl_double2) * temp,0, &err); clErrChk(err);
	clFinish(queue);
}
void dpFluidSimulation::plan(){
	// initialize direction buffer
	for(int i = 0; i < 8; i++){
		dirX.s[i] = e[i + 1][0];
		dirY.s[i] = e[i + 1][1];
	}
	
	//PLAN:
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_if0));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_of0));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_if1234));
	clErrChk(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_of1234));
	clErrChk(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_if5678));
	clErrChk(clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_of5678));
	clErrChk(clSetKernelArg(kernel, 6, sizeof(cl_mem), &type));
	clErrChk(clSetKernelArg(kernel, 7, sizeof(cl_double8), &dirX));
	clErrChk(clSetKernelArg(kernel, 8, sizeof(cl_double8), &dirY));
	clErrChk(clSetKernelArg(kernel, 9, sizeof(cl_mem), &weight));
	clErrChk(clSetKernelArg(kernel, 10, sizeof(cl_double), &omega));
	
	globalSize[0] = dims[0];
	globalSize[1] = dims[1];
}

int dpFluidSimulation::execute(){
	int safeExit =0;
	for (int i = 1; i <= iterations; i++){

		size_t temp = dims[0] * dims[1];

		//Enqueue write data to device

		// Write the cell type data each frame
		clErrChk(clEnqueueWriteBuffer(queue,type,CL_FALSE,0,sizeof(cl_bool) * temp,h_type,0, 0, NULL));
		
		// If odd frame (starts from odd frame)
		// Then inputs : d_if0, d_if1234, d_if5678
		// Outputs : d_of0, f_of1234, d_of5678
		// Else they are swapped
		if(i % 2){
			clErrChk(clEnqueueWriteBuffer(queue,d_if0,CL_FALSE,0,sizeof(cl_double) * temp,h_if0,0, 0, NULL));
			clErrChk(clEnqueueWriteBuffer(queue,d_if1234,CL_FALSE,0,sizeof(cl_double4) * temp,h_if1234,0, 0, NULL));
			clErrChk(clEnqueueWriteBuffer(queue,d_if5678,CL_FALSE,0,sizeof(cl_double4) * temp,h_if5678,0, 0, NULL));
		}
		else{
			clErrChk(clEnqueueWriteBuffer(queue,d_of0,CL_FALSE,0,sizeof(cl_double) * temp,h_if0,0, 0, NULL));
			clErrChk(clEnqueueWriteBuffer(queue,d_of1234,CL_FALSE,0,sizeof(cl_double4) * temp,h_if1234,0, 0, NULL));
			clErrChk(clEnqueueWriteBuffer(queue,d_of5678,CL_FALSE,0,sizeof(cl_double4) * temp,h_if5678,0, 0, NULL));
		}

		// Set kernel arguments
		clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_if0));
		clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_of0));
		clErrChk(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_if1234));
		clErrChk(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_of1234));
		clErrChk(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_if5678));
		clErrChk(clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_of5678));
		clErrChk(clSetKernelArg(kernel, 6, sizeof(cl_mem), &type));
		clErrChk(clSetKernelArg(kernel, 7, sizeof(cl_double8), &dirX));
		clErrChk(clSetKernelArg(kernel, 8, sizeof(cl_double8), &dirY));
		clErrChk(clSetKernelArg(kernel, 9, sizeof(cl_mem), &weight));
		clErrChk(clSetKernelArg(kernel, 10, sizeof(cl_double), &omega));
		clErrChk(clSetKernelArg(kernel, 11, sizeof(cl_mem), &velocity));
		
		err=clEnqueueNDRangeKernel(queue,kernel,2,0,globalSize,localSize,0, 0, 0);
		clErrChk(err);
		if(err<0)
			safeExit = -1;
		
		clErrChk(clEnqueueReadBuffer(queue,velocity,CL_FALSE,0,sizeof(cl_double2) * temp,u,0, 0, NULL));
		clFinish(queue);
		
		//Read back the data into host buffer
		if(i % 2){
			clErrChk(clEnqueueReadBuffer(queue,d_of0,CL_FALSE,0,sizeof(cl_double) * temp,h_of0,0, 0, NULL));
			clErrChk(clEnqueueReadBuffer(queue,d_of1234,CL_FALSE,0,sizeof(cl_double4) * temp,h_of1234,0, 0, NULL));
			clErrChk(clEnqueueReadBuffer(queue,d_of5678,CL_FALSE,0,sizeof(cl_double4) * temp,h_of5678,0, 0, NULL));
		}
		else{
			clErrChk(clEnqueueReadBuffer(queue,d_if0,CL_FALSE,0,sizeof(cl_double) * temp,h_of0,0, 0, NULL));
			clErrChk(clEnqueueReadBuffer(queue,d_if1234,CL_FALSE,0,sizeof(cl_double4) * temp,h_of1234,0, 0, NULL));
			clErrChk(clEnqueueReadBuffer(queue,d_if5678,CL_FALSE,0,sizeof(cl_double4) * temp,h_of5678,0, 0, NULL));
		}

		// Copy from host output to the next input
		memcpy(h_if0, h_of0, sizeof(cl_double) * temp);
		memcpy(h_if1234, h_of1234, sizeof(cl_double4) * temp);
		memcpy(h_if5678, h_of5678, sizeof(cl_double4) * temp);

		cl_mem temp0, temp1234, temp5678;

		//swap input and output buffers
		temp0 = d_of0;
		temp1234 = d_of1234;
		temp5678 = d_of5678;

		d_of0 = d_if0;
		d_of1234 = d_if1234;
		d_of5678 = d_if5678;

		d_if0 = temp0;
		d_if1234 = temp1234;
		d_if5678 = temp5678;
		clFinish(queue);
		
		if (safeExit <0)
			return -1;
	}
	
	clFinish(queue);
	return 0;
}

void dpFluidSimulation::memoryCopyIn(){
	
}

void dpFluidSimulation::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(d_if0));
	clErrChk(clReleaseMemObject(d_of0));
	clErrChk(clReleaseMemObject(d_if1234));
	clErrChk(clReleaseMemObject(d_of1234));
	clErrChk(clReleaseMemObject(d_if5678));
	clErrChk(clReleaseMemObject(d_of5678));
	clErrChk(clReleaseMemObject(type));
	clErrChk(clReleaseMemObject(weight));
	clErrChk(clReleaseMemObject(velocity));

	/* release program resources */
	free(h_if0);
	free(h_if1234);
	free(h_if5678);
	free(h_of0);
	free(h_of1234);
	free(h_of5678);
	free(h_type);
	free(rho);
	free(u);
}

void dpFluidSimulation::reset(){

	// Initial velocity is 0
	cl_double2 u0;
	u0.s[0] = u0.s[1] = 0.0f;

	for (int y = 0; y < dims[1]; y++){
		for (int x = 0; x < dims[0]; x++){
			int pos = x + y * dims[0];

			double den = 10.0f;

			// Initialize the velocity buffer
			u[pos] = u0;

			//rho[pos] = 10.0f;
			h_if0[pos] = computefEq(w[0], e[0], den, u0);
			h_if1234[pos * 4 + 0] = computefEq(w[1], e[1], den, u0);
			h_if1234[pos * 4 + 1] = computefEq(w[2], e[2], den, u0);
			h_if1234[pos * 4 + 2] = computefEq(w[3], e[3], den, u0);
			h_if1234[pos * 4 + 3] = computefEq(w[4], e[4], den, u0);

			h_if5678[pos * 4 + 0] = computefEq(w[5], e[5], den, u0);
			h_if5678[pos * 4 + 1] = computefEq(w[6], e[6], den, u0);
			h_if5678[pos * 4 + 2] = computefEq(w[7], e[7], den, u0);
			h_if5678[pos * 4 + 3] = computefEq(w[8], e[8], den, u0);

			// Initialize boundary cells
			if (x == 0 || x == (dims[0] - 1) || y == 0 || y == (dims[1] - 1)){
					h_type[pos] = 1;
			}

			// Initialize fluid cells
			else{
				h_type[pos] = 0;
			}
		}
	}
}

// Calculates equivalent distribution
double dpFluidSimulation::computefEq(cl_double weight, double dir[2], double rho, cl_double2 velocity){
    double u2 = velocity.s[0] * velocity.s[0] + velocity.s[1] * velocity.s[1];
    double eu = dir[0] * velocity.s[0] + dir[1] * velocity.s[1];

    return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}
