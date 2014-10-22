#include "dpCudaOxxxxx.hpp"
#include "errorCheck.hpp"
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);


__device__ void oxxxxx(double* p, double fmass, int nhel, int nsf, cmplx* fo){
	
	fo[4] = cmplx(p[0]*nsf, p[3]*nsf);
	fo[5] = cmplx(p[1]*nsf, p[2]*nsf);
	int nh = nhel*nsf;
	cmplx chi[2];

	if (fmass!=0.) {
		double pp = fmin(p[0],sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3] )  );

		if (pp==0.) {
			double sqm[2];

			sqm[0] = sqrt(fabs(fmass) );

			sqm[1] = copysign(sqm[0], fmass);

			int ip = -(1+nh)/2;

			int im = (1-nh)/2;

			fo[0] = cmplx((double)(im) *sqm[im]);
			fo[1] = cmplx((double)(ip*nsf)*sqm[im]);
			fo[2] = cmplx((double)(im*nsf)*sqm[-ip]);
			fo[3] = cmplx((double)(ip) *sqm[-ip]);

		} 
	
		else {
			double sf[2],omega[2];

			sf[0] = (double)(1 + nsf + (1-nsf)*nh)*0.5;
			sf[1] = (double)(1 + nsf - (1-nsf)*nh)*0.5;
			omega[0] = sqrt(p[0]+pp);

			omega[1] = fmass*(1./omega[0]);

			double pp3 = fmax(pp+p[3],0.);

			chi[0] = cmplx(sqrt(pp3*0.5*(1./pp)));
			
			if (pp3==0.) {
				chi[1] = cmplx((double)(nh));
			} 
			
			else {
				chi[1] =  cmplx((double)(nh)*p[1],-p[2])/sqrt(2.*pp*pp3)  ;
			}
			int ip = (3+nh)/2-1;

			int im = (3-nh)/2-1;

			fo[0] = sf[1]*omega[im]*chi[im];
			fo[1] = sf[1]*omega[im]*chi[ip];
			fo[2] = sf[0]*omega[ip]*chi[im];
			fo[3] = sf[0]*omega[ip]*chi[ip];
		}
	
	} 
	
	else {
		double sqp0p3;

		if (p[1]==0. && p[2]==0. && p[3]<0.) {
			sqp0p3 = 0.;
		} 
		else {
			sqp0p3 = sqrt(fmax(p[0]+p[3],0.))*(double)(nsf);
		}
		chi[0] = cmplx(sqp0p3);
		
		if (sqp0p3==0.) {
			chi[1] = cmplx( (double)(nhel)*sqrt(2.*p[0]) );
		} 
		
		else {
			chi[1] = (1./sqp0p3) * cmplx( (double)(nh)*p[1],-p[2] );
		}
		
		cmplx czero = cmplx(0.,0.);
		if (nh==1) {
			fo[0] = chi[0];
			fo[1] = chi[1];
			fo[2] = czero;
			fo[3] = czero;

		} 
		
		else {
			fo[0] = czero;
			fo[1] = czero;
			fo[2] = chi[1];
			fo[3] = chi[0];
		} 
	
	}
	
}


// Kernel that executes on the CUDA device
__global__ void Oxxxxx(double *P_d, cmplx *Fo_d, int Psize){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double fmass = 124.412;
	int nhel = 2;
	int nsf = 4;
	
	double P[4];
	cmplx Fo[6];
	
	P[0] = P_d[idx*4 + 0];
	P[1] = P_d[idx*4 + 1];
	P[2] = P_d[idx*4 + 2];
	P[3] = P_d[idx*4 + 3];
	
	if (idx*4 < Psize){
		oxxxxx(P, fmass, nhel, nsf, Fo);
	}
	
	else
		return;
	
	//synch?
	Fo_d[6*idx + 0] = Fo[0];
	Fo_d[6*idx + 1] = Fo[1];
	Fo_d[6*idx + 2] = Fo[2];
	Fo_d[6*idx + 3] = Fo[3];
	Fo_d[6*idx + 4] = Fo[4];
	Fo_d[6*idx + 5] = Fo[5];
}

//notice unused parameters for CUDA kernel:
dpCudaOxxxxx::dpCudaOxxxxx(cl_context ctx, cl_command_queue q){

	workDimension = ONE_D;
	//name is same as cl alternative allowing the analysis script to later figure 
	//out this measurement was from a cuda kernel by inspecting the platform id from dpClient
	name = "Oxxxxx";

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	cudaErrChk(cudaPeekAtLastError());
}

void dpCudaOxxxxx::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0] = localSize[1] = localSize[2] = 1;
	
	Psize = 1048576*dataMB/(sizeof(double)*4);
	MB = Psize * (sizeof(double)*4) / 1048576;
	
}

void dpCudaOxxxxx::init(){
	//allocate local memory for original array
	P = new double[4*Psize];
	Fo = new cmplx[6*Psize];
	
	if(!P || !Fo)
		fprintf(stderr, "error in malloc\n");
	
	generateArray(P, Psize);
	
	dataParameters.push_back(Psize);
	dataNames.push_back("nElements");

}

void dpCudaOxxxxx::memoryCopyOut(){
	BEGIN
	cudaErrChk( cudaMalloc((void **) &P_d, Psize*sizeof(double)*4 ));
	cudaErrChk( cudaMalloc((void **) &Fo_d, Psize*sizeof(cmplx)*6 ));
	cudaErrChk( cudaMemcpy(P_d, P, Psize*sizeof(double)*4, cudaMemcpyHostToDevice) );
	END
}

void dpCudaOxxxxx::plan(){
	BEGIN
	blockSize = props.maxThreadsPerBlock;
	lastBlock = 0;
	nBlocks = Psize/blockSize; //nblocks = ceil(Psize/blockSize)
	if (Psize%blockSize != 0)
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

int dpCudaOxxxxx::execute(){
	cudaError_t err;
	int stride = blockSize*nBlocks;
	int lastStride = blockSize * lastBlock;
	
	BEGIN
	for (int i = 0; i < nKernels; i++){
		Oxxxxx <<< nBlocks, blockSize >>> (P_d + (i*stride), Fo_d + (i*stride), Psize - (i*stride));
	}
	if (lastBlock != 0){
		Oxxxxx <<<lastBlock, blockSize >>> (P_d + (nKernels*lastStride), Fo_d + (nKernels*lastStride), Psize - (nKernels*lastStride));
	}
	err = cudaPeekAtLastError();
	cudaErrChk(err);
	cudaErrChk(cudaDeviceSynchronize());
	END
	if(err!=cudaSuccess)
		return -1;
	return 0;
}

void dpCudaOxxxxx::memoryCopyIn(){
	BEGIN
	cudaErrChk(cudaMemcpy(Fo, Fo_d, Psize*sizeof(cmplx)*6, cudaMemcpyDeviceToHost));
	END
}

void dpCudaOxxxxx::cleanUp(){
	cudaFree(P_d);
	cudaFree(Fo_d);
	delete[] P;
	delete[] Fo;
}

void dpCudaOxxxxx::generateArray(double *P, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N - 4; i=i+4){
		P[i+0]=rand() / (RAND_MAX/99999.9 + 1);
		P[i+1]=rand() / (RAND_MAX/99999.9 + 1);
		P[i+2]=rand() / (RAND_MAX/99999.9 + 1);
		P[i+3]=rand() / (RAND_MAX/99999.9 + 1);
	}
}
