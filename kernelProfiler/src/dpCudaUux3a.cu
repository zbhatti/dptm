#include "dpCudaUux3a.hpp"
#include "errorCheck.hpp"
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);
#define rSQRT2 0.707106781186
#define ERR fprintf(stderr, "%d\n" ,__LINE__);

__device__
void ixxxx1(float* p, int nHEL, int nSF, cmplx* fi)
{
	float SQP0P3 = sqrtf(p[0]+p[3])*(float)(nSF);
	int NH = nHEL*nSF;
	fi[4] = mkcmplx(p[0]*(float)(nSF), p[3]*(float)(nSF));
	fi[5] = mkcmplx(p[1]*(float)(nSF), p[2]*(float)(nSF));
	cmplx CHI = mkcmplx(NH*p[1]*(1.0f/SQP0P3), p[2]*(1.0f/SQP0P3));
	cmplx CZERO = mkcmplx(0.0f, 0.0f);
	cmplx CSQP0P3 = mkcmplx(SQP0P3, 0.0f);
	fi[0]=(NH== 1)*CZERO + (NH==-1)*CHI;
	fi[1]=(NH== 1)*CZERO + (NH==-1)*CSQP0P3;
	fi[2]=(NH== 1)*CSQP0P3 + (NH==-1)*CZERO;
	fi[3]=(NH== 1)*CHI + (NH==-1)*CZERO;
	return;
}

__device__
void oxxxx2(float* p, int nHEL, int nSF, cmplx* fo)
	{
	int NH=nHEL*nSF;
	fo[4] = mkcmplx(p[0]*(float)(nSF), p[3]*(float)(nSF));
	fo[5] = mkcmplx(p[1]*(float)(nSF), p[2]*(float)(nSF));
	cmplx CHI = mkcmplx(-nHEL*sqrtf(2.0f*p[0]), 0.0f);
	cmplx CZERO = mkcmplx(0.0f,0.0f);
	fo[0]=CZERO;
	fo[1]=(NH== 1)*CHI + (NH==-1)*CZERO;
	fo[2]=(NH== 1)*CZERO + (NH==-1)*CHI;
	fo[3]=CZERO;
	return;
}

__device__
void vxxxx0(float* p, int nHEL, int nSV, cmplx* vc)
{
	vc[4] = mkcmplx(p[0], p[3])*nSV;
	vc[5] = mkcmplx(p[1], p[2])*nSV;
	float rpt = rsqrtf(p[1]*p[1] + p[2]*p[2]);
	vc[0] = mkcmplx(0.0f, 0.0f);
	vc[3] = mkcmplx( (float)(nHEL)*(1.0f/(rpt*p[0]))*rSQRT2, 0.0f);
	float pzpt = (p[3]*(1.0f/p[0])*rpt)*rSQRT2 *(float)(nHEL);
	vc[1] = mkcmplx(-p[1]*pzpt, -nSV*p[2] * rpt * rSQRT2);
	vc[2] = mkcmplx(-p[2]*pzpt,
	+nSV*p[1] * rpt * rSQRT2);
	return;
}

__device__
void fvoxx0(cmplx* fo, cmplx* vc, float* gal, cmplx* fvo)
{
	fvo[4] = fo[4]+vc[4];
	fvo[5] = fo[5]+vc[5];
	float pf[4];
	pf[0] = fvo[4].re;
	pf[1] = fvo[5].re;
	pf[2] = fvo[5].im;
	pf[3] = fvo[4].im;
	float pf2 = pf[0]*pf[0] - (pf[1]*pf[1] + pf[2]*pf[2] + pf[3]*pf[3]);
	cmplx cI = mkcmplx( 0.0f, 1.0f);
	float d = -1.0f/pf2;
	cmplx sl1 = (vc[0] + vc[3])*fo[2] + (vc[1] + cI*vc[2])*fo[3];
	cmplx sl2 = (vc[0] - vc[3])*fo[3] + (vc[1] - cI*vc[2])*fo[2];
	cmplx sr1 = (vc[0] - vc[3])*fo[0] - (vc[1] + cI*vc[2])*fo[1];
	cmplx sr2 = (vc[0] + vc[3])*fo[1] - (vc[1] - cI*vc[2])*fo[0];
	fvo[0] = ( gal[1]*((pf[0]+pf[3])*sr1 + fvo[5] *sr2 ))*d;
	fvo[1] = ( gal[1]*((pf[0]-pf[3])*sr2 + conj(fvo[5])*sr1 ))*d;
	fvo[2] = ( gal[0]*((pf[0]-pf[3])*sl1 - fvo[5] *sl2 ))*d;
	fvo[3] = ( gal[0]*((pf[0]+pf[3])*sl2 - conj(fvo[5])*sl1 ))*d;
	return;
}

//note: this was defined as iovxx0() in http://arxiv.org/pdf/0908.4403.pdf pg 12
__device__
void iovxxx(cmplx* fi, cmplx* fo, cmplx* vc, float* gal, cmplx& vertex)
{
vertex =
	gal[0]*((fo[2]*fi[0]+fo[3]*fi[1])*vc[0]
	+(fo[2]*fi[1]+fo[3]*fi[0])*vc[1]
	-((fo[2]*fi[1]-fo[3]*fi[0])*vc[2])
	*mkcmplx(0.0f, 1.0f)
	+(fo[2]*fi[0]-fo[3]*fi[1])*vc[3])
	+gal[1]*((fo[0]*fi[2]+fo[1]*fi[3])*vc[0]
	-(fo[0]*fi[3]+fo[1]*fi[2])*vc[1]
	+((fo[0]*fi[3]-fo[1]*fi[2])*vc[2])
	*mkcmplx(0.0f, 1.0f)
	-(fo[0]*fi[2]-fo[1]*fi[3])*vc[3]);
return;
}

__device__
void fvixx0(cmplx* fi, cmplx* vc, float* gal, cmplx* fvi)
	{
	fvi[4] = fi[4]-vc[4];
	fvi[5] = fi[5]-vc[5];
	float pf[4];
	pf[0] = fvi[4].re;
	pf[1] = fvi[5].re;
	pf[2] = fvi[5].im;
	pf[3] = fvi[4].im;
	float pf2 = pf[0]*pf[0] -  (pf[1]*pf[1] + pf[2]*pf[2] + pf[3]*pf[3]);
	cmplx cI = mkcmplx( 0.0f, 1.0f);
	float d = -1.0f/pf2;
	cmplx sl1 = (vc[0] + vc[3])*fi[0] + (vc[1] - cI*vc[2])*fi[1];
	cmplx sl2 = (vc[0] - vc[3])*fi[1] + (vc[1] + cI*vc[2])*fi[0];
	cmplx sr1 = (vc[0] - vc[3])*fi[2] - (vc[1] - cI*vc[2])*fi[3];
	cmplx sr2 = (vc[0] + vc[3])*fi[3] - (vc[1] + cI*vc[2])*fi[2];
	fvi[0] = ( gal[0]*((pf[0]-pf[3])*sl1 - conj(fvi[5])*sl2))*d;
	fvi[1] = ( gal[0]*((pf[0]+pf[3])*sl2 - fvi[5] *sl1))*d;
	fvi[2] = ( gal[1]*((pf[0]+pf[3])*sr1 + conj(fvi[5])*sr2))*d;
	fvi[3] = ( gal[1]*((pf[0]-pf[3])*sr2 + fvi[5] *sr1))*d;
	return;
}

// Each thread corresponds to an event
// Each thread has 5 particles that have a 4momentum description
// No ouput

__global__ void Uux3a(float *P_d, int nEvents){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > nEvents)
		return;
	
	//first term gets us to the correct event in P_d, the second one gets us the corresponding 4momentum for each particle
	float *p1 = &P_d[idx*5*4 + 4*0];
	float *p2 = &P_d[idx*5*4 + 4*1];
	float *p3 = &P_d[idx*5*4 + 4*2];
	float *p4 = &P_d[idx*5*4 + 4*3];
	float *p5 = &P_d[idx*5*4 + 4*4];
	
	// coupling constants of FFV vertex, using meaningless fillers
	float gau[2];
	gau[0] = 5123.51;
	gau[1] = 3109.64;
	
	//twice fermion helicity (-1 or 1), using meaningless fillers
	int nh1 = -1;
	int nh2 = 1;
	int nh3 = -1;
	int nh4 = -1;
	int nh5 = 1;
	
	cmplx w01[6], w02[6], w03[6], w04[6], w05[6];
	ixxxx1(p1, nh1, +1, w01);
	oxxxx2(p2, nh2, -1, w02);
	vxxxx0(p3, nh3, +1, w03);
	vxxxx0(p4, nh4, +1, w04);
	vxxxx0(p5, nh5, +1, w05);
	
	cmplx w06[6], w07[6], w08[6];
	cmplx ampsum = mkcmplx(0.0f, 0.0f);
	cmplx amp;
	
	fvoxx0(w02,w03,gau,w06);
	fvoxx0(w06,w04,gau,w07);
	iovxxx(w01,w07,w05,gau,amp); 
	ampsum = ampsum + amp;
	
	fvixx0(w01,w04,gau,w07);
	fvoxx0(w02,w05,gau,w08);
	iovxxx(w07,w08,w03,gau,amp);
	ampsum = ampsum + amp;
	
	fvoxx0(w02,w03,gau,w06);
	fvixx0(w01,w04,gau,w07);
	iovxxx(w07,w06,w05,gau,amp);
	ampsum = ampsum + amp;
	
	fvoxx0(w02,w04,gau,w06);
	fvixx0(w01,w05,gau,w07);
	iovxxx(w07,w06,w03,gau,amp);
	ampsum = ampsum + amp;
	
	fvixx0(w01,w03,gau,w07);
	fvixx0(w07,w04,gau,w08);
	iovxxx(w08,w02,w05,gau,amp);
	ampsum = ampsum + amp;
	
	fvixx0(w01,w03,gau,w07);
	fvoxx0(w02,w04,gau,w06);
	iovxxx(w07,w06,w05,gau,amp);
	ampsum = ampsum + amp;
	
}

//notice unused parameters for CUDA kernel:
dpCudaUux3a::dpCudaUux3a(cl_context ctx, cl_command_queue q){

	workDimension = ONE_D;
	//name is same as cl alternative allowing the analysis script to later figure 
	//out this measurement was from a cuda kernel by inspecting the platform id from dpClient
	name = "Uux3a";

	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	cudaErrChk(cudaPeekAtLastError());
}

void dpCudaUux3a::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0] = localSize[1] = localSize[2] = 1;
	
	nEvents = 1048576*dataMB/(sizeof(float)*5*4);
	MB = ( nEvents * sizeof(float)*5*4) / 1048576;

}

void dpCudaUux3a::init(){
	//allocate local memory for original array
	eventsP = new float[5*4*nEvents]; //4 momentum for each of the 5 particles in an event. nEevents
	Fo = new cmplx[8*6*nEvents]; //6 complex "w" for each of the 8 outputes of an event. nEvents
	inputBytes = 5*4*nEvents*sizeof(float);
	outputBytes = 8*6*nEvents*sizeof(cmplx);
	
	if(!eventsP || !Fo)
		fprintf(stderr, "error in malloc\n");
	
	generateArray(eventsP, nEvents);

	dataParameters.push_back(nEvents);
	dataNames.push_back("nEvents");

}

void dpCudaUux3a::memoryCopyOut(){
	BEGIN
	cudaErrChk( cudaMalloc((void **) &eventsP_d, inputBytes ));
	//cudaErrChk( cudaMalloc((void **) &Fo_d, outputBytes ));
	cudaErrChk( cudaMemcpy(eventsP_d, eventsP, inputBytes, cudaMemcpyHostToDevice) );
	END
}

void dpCudaUux3a::plan(){
	BEGIN
	blockSize = props.maxThreadsPerBlock;
	lastBlock = 0;
	nBlocks = nEvents/blockSize; //nblocks = ceil(nEvents/blockSize)
	if (nEvents%blockSize != 0)
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

int dpCudaUux3a::execute(){
	cudaError_t err;
	int stride = blockSize*nBlocks;
	int lastStride = blockSize * lastBlock;
	
	BEGIN
	for (int i = 0; i < nKernels; i++){
		Uux3a <<< nBlocks, blockSize >>> (eventsP_d + (i*stride), nEvents - (i*stride));
	}
	if (lastBlock != 0){
		Uux3a <<<lastBlock, blockSize >>> (eventsP_d + (nKernels*lastStride), nEvents - (nKernels*lastStride));
	}
	err = cudaPeekAtLastError();
	cudaErrChk(err);
	cudaErrChk(cudaDeviceSynchronize());
	END
	if(err!=cudaSuccess)
		return -1;
	return 0;
}

void dpCudaUux3a::memoryCopyIn(){
	BEGIN
	//cudaErrChk(cudaMemcpy(Fo, Fo_d, outputBytes, cudaMemcpyDeviceToHost));
	END
}

void dpCudaUux3a::cleanUp(){
	cudaFree(eventsP_d);
	//cudaFree(Fo_d);
	delete[] eventsP;
	delete[] Fo;
}

//5 particles each described by their four-momentum for each event,
void dpCudaUux3a::generateArray(float *eventsP, int nEvents){
	int n,j,k;
	srand(time(NULL));
	
	for (n=0; n < nEvents; n++){
		for (j=0; j<5; j++){
			for (k=0; k<4; k++){
				eventsP[n*5*4 + 4*j + k]=rand() / (RAND_MAX/99999.9 + 1);
			}
		}
	}
}

