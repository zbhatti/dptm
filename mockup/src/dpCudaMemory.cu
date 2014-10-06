#include "dpCudaMemory.hpp"
#include "errorCheck.hpp"
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);

// Kernel that executes on the CUDA device
__global__ void Memory(float *A_d, float *B_d, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) 
		B_d[idx] = A_d[idx];
}

//notice unused parameters for CUDA kernel:
dpCudaMemory::dpCudaMemory(cl_context ctx, cl_command_queue q){
	workDimension = ONE_D;
	//name is same as cl alternative allowing the analysis script to later figure 
	//out this measurement was from a cuda kernel by inspecting the platform id from dpClient
	name = "Memory";

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
}

void dpCudaMemory::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = localSize[1] = localSize[2] = 1;
	
	Asize = 1048576*dataMB/sizeof(float);
	MB = Asize * sizeof(float) / 1048576;
}

void dpCudaMemory::init(){
	dataParameters.push_back(Asize);
	dataNames.push_back("nElements");
	
	// Allocate and initialize host arrays 
	A = (float *)malloc(sizeof(cl_float) * Asize);
	B = (float *)malloc(sizeof(cl_float) * Asize);
	generateArray(A, Asize);
}

void dpCudaMemory::memoryCopyOut(){
	BEGIN
	cudaErrChk(cudaMalloc((void **) &A_d, Asize*sizeof(float)));
	cudaErrChk(cudaMalloc((void **) &B_d, Asize*sizeof(float)));
	cudaErrChk(cudaMemcpy(A_d, A, Asize*sizeof(float), cudaMemcpyHostToDevice));
	END 
}

void dpCudaMemory::plan(){
	BEGIN
	blockSize = props.maxThreadsPerBlock;
	lastBlock = 0;
	nBlocks = Asize/blockSize; //nblocks = ceil(Asize/blockSize)
	if (Asize%blockSize != 0)
		nBlocks++;
	if (nBlocks > 65535)
		nBlocks = 65535;
	nKernels = nBlocks / 65535;
	if (nKernels == 0){
		lastBlock = nBlocks; //run normally
	}
	else 
		lastBlock = nBlocks % 65535; //run repeated
	END
}

int dpCudaMemory::execute(){
	cudaError_t err;
	BEGIN
	for (int i = 0; i < nKernels; i++)
		Memory <<< nBlocks, blockSize >>> (A_d + (i*blockSize*nBlocks*sizeof(float)), B_d + (i*blockSize*nBlocks*sizeof(float)), Asize);
	if (lastBlock != 0)
		Memory <<<lastBlock, blockSize >>> (A_d + (nKernels*blockSize*nBlocks*sizeof(float)), B_d + (nKernels*blockSize*nBlocks*sizeof(float)), Asize);
	err = cudaPeekAtLastError() ;
	cudaErrChk(err);
	cudaErrChk(cudaDeviceSynchronize());
	END
	if(err!=cudaSuccess)
		return -1;
	return 0;
}

void dpCudaMemory::memoryCopyIn(){
	BEGIN
	cudaErrChk(cudaMemcpy(B, B_d, Asize*sizeof(float), cudaMemcpyDeviceToHost));
	END
}

void dpCudaMemory::cleanUp(){
	// Cleanup and leave
	cudaFree(A_d);
	cudaFree(B_d);
	free(A);
	free(B);
}

void dpCudaMemory::generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}
