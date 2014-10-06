#include "dpCudaVectorAdd.hpp"
#include "errorCheck.hpp"
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);

// Kernel that executes on the CUDA device
__global__ void vectorAdd(float *A_d, float *B_d, float *C_d, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) 
		C_d[idx] = A_d[idx] + B_d[idx];
}

//notice unused parameters for CUDA kernel:
dpCudaVectorAdd::dpCudaVectorAdd(cl_context ctx, cl_command_queue q){
	workDimension = ONE_D;
	//name is same as cl alternative allowing the analysis script to later figure 
	//out this measurement was from a cuda kernel by inspecting the platform id from dpClient
	name = "VectorAdd";

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
}

void dpCudaVectorAdd::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = localSize[1] = localSize[2] = 1;
	
	Asize = 1048576*dataMB/sizeof(float);
	MB = Asize * sizeof(float) / 1048576;
}

void dpCudaVectorAdd::init(){
	dataParameters.push_back(Asize);
	dataNames.push_back("nElements");
	
	// Allocate and initialize host arrays 
	A = (float *)malloc(sizeof(cl_float) * Asize);
	B = (float *)malloc(sizeof(cl_float) * Asize);
	C = (float *)malloc(sizeof(cl_float) * Asize);
	generateArray(A, Asize);
	generateArray(B, Asize);
}

void dpCudaVectorAdd::memoryCopyOut(){
	BEGIN
	cudaErrChk(cudaMalloc((void **) &A_d, Asize*sizeof(float)));
	cudaErrChk(cudaMalloc((void **) &B_d, Asize*sizeof(float)));
	cudaErrChk(cudaMalloc((void **) &C_d, Asize*sizeof(float)));
	cudaErrChk(cudaMemcpy(A_d, A, Asize*sizeof(float), cudaMemcpyHostToDevice));
	cudaErrChk(cudaMemcpy(B_d, B, Asize*sizeof(float), cudaMemcpyHostToDevice));
	END
	//printf("%0.3f,",delTime);
}

void dpCudaVectorAdd::plan(){
	BEGIN
	blockSize = props.maxThreadsPerBlock;
	nBlocks = Asize/blockSize; //nblocks = ceil(Asize/blockSize)
	if (Asize%blockSize != 0)
		nBlocks++;
	END
}

int dpCudaVectorAdd::execute(){
	cudaError_t err;
	BEGIN
	vectorAdd <<< nBlocks, blockSize >>> (A_d, B_d, C_d, Asize);
	err = cudaPeekAtLastError() ;
	cudaErrChk(err);
	cudaErrChk(cudaDeviceSynchronize());
	END
	//printf("%0.3f,",delTime);
	if(err!=cudaSuccess)
		return -1;
	return 0;
}

void dpCudaVectorAdd::memoryCopyIn(){
	BEGIN
	cudaErrChk(cudaMemcpy(C, C_d, Asize*sizeof(float), cudaMemcpyDeviceToHost));
	END
	//printf("%0.3f,\n",delTime);
}

void dpCudaVectorAdd::cleanUp(){
	// Cleanup and leave
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	free(A);
	free(B);
	free(C);
}

void dpCudaVectorAdd::generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}