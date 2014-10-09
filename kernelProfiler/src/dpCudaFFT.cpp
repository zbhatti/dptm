/*
#include "dpCudaFFT.hpp"
#include "errorCheck.hpp"

#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);
#define cudaErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//code from stackexchange to print cuda return messages
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=false){
   if (code != cudaSuccess){
      fprintf(stderr,"%s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) 
				exit(code);
   }
}

dpCudaFFT::dpCudaFFT(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;
	name = "FFT";
	
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

}

void dpCudaFFT::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] =localSize[1]=localSize[2]=1;
	
	Asize = (dataMB*1048576)/(sizeof(cufftComplex));
	if (Asize%2 != 0)
		Asize++;
	
	MB = Asize * sizeof(cufftComplex)/1048576;
	
}

void dpCudaFFT::init(){
	dataParameters.push_back(Asize);
	dataNames.push_back("nVectors");
	
	Ain = (cufftComplex*) malloc(Asize*sizeof(cufftComplex));
	Aout = (cufftComplex*) malloc(Asize * sizeof(cufftComplex));
	if (!Aout || !Ain)
		fprintf(stderr,"error in malloc");
	
	generate(Ain, Asize);
}

void dpCudaFFT::memoryCopyOut(){

	BEGIN
	cudaErrChk(cudaMalloc((void**)&A_d, Asize * sizeof(cufftComplex)));
	cudaErrChk(cudaMemcpy(A_d, Ain, Asize * sizeof(cufftComplex), cudaMemcpyHostToDevice));
	END
	//printf("%0.3f,", delTime);
}

void dpCudaFFT::plan(){
	int ret = 0;
	BEGIN
	ret = cufftPlan1d(&plancufft, Asize, CUFFT_C2C, 1);
	if (ret != 0)
		fprintf(stderr, "%s %d", "cufftplan1d fail err: ", ret);
	
	END
	//printf("%0.3f,", delTime);
}

int dpCudaFFT::execute(){
	int ret = 0;
	BEGIN
	ret = cufftExecC2C(plancufft, A_d, A_d, CUFFT_FORWARD);
	if (ret != 0){
		fprintf(stderr, "%s %d", "cufftexecc2c fail err: ", ret);
		return -1;
	}
	END
	//printf("%0.3f,", delTime);
	return 0;
}

void dpCudaFFT::memoryCopyIn(){
	BEGIN
	cudaErrChk(cudaMemcpy(Aout, A_d, Asize * sizeof(cufftComplex), cudaMemcpyDeviceToHost));	
	END
	//printf("%0.3f,\n", delTime);
}

void dpCudaFFT::cleanUp(){
	cudaErrChk(cudaFree(A_d));
	cufftDestroy(plancufft); 
	free(Aout);
	free(Ain);
}

void dpCudaFFT::generate(cufftComplex *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i].x = rand() / (RAND_MAX/99999.9 + 1);
		A[i].y = rand() / (RAND_MAX/99999.9 + 1);
	}
}


*/

