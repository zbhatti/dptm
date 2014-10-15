#include "dpOxxxxx.hpp"
#include "errorCheck.hpp"
#include "cmplx.h"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }
#include <string.h>

void oxxxxx(double* p, double fmass, int nhel, int nsf, cmplx* fo)
{
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
      
    } else {
      double sf[2],omega[2];
      
      sf[0] = (double)(1 + nsf + (1-nsf)*nh)*0.5;
      sf[1] = (double)(1 + nsf - (1-nsf)*nh)*0.5;
      omega[0] = sqrt(p[0]+pp);
      
      omega[1] = fmass*(1./omega[0]);
      
      double pp3 = fmax(pp+p[3],0.);
      
      chi[0] = cmplx(sqrt(pp3*0.5*(1./pp)));
      if (pp3==0.) {
	chi[1] = cmplx((double)(nh));
      } else {
	chi[1] =  cmplx((double)(nh)*p[1],-p[2])/sqrt(2.*pp*pp3)  ;
	
      }
      int ip = (3+nh)/2-1;
      
      int im = (3-nh)/2-1;
      
      fo[0] = sf[1]*omega[im]*chi[im];
      fo[1] = sf[1]*omega[im]*chi[ip];
      fo[2] = sf[0]*omega[ip]*chi[im];
      fo[3] = sf[0]*omega[ip]*chi[ip];
      
    }
  } else {
    double sqp0p3;
    
    if (p[1]==0. && p[2]==0. && p[3]<0.) {
      sqp0p3 = 0.;
    } else {
      sqp0p3 = sqrt(fmax(p[0]+p[3],0.))*(double)(nsf);
    }
    chi[0] = cmplx(sqp0p3);
    if (sqp0p3==0.) {
      chi[1] = cmplx( (double)(nhel)*sqrt(2.*p[0]) );
    } else {
      chi[1] = (1./sqp0p3) * cmplx( (double)(nh)*p[1],-p[2] );
      
    }
    cmplx czero = cmplx(0.,0.);
    if (nh==1) {
      fo[0] = chi[0];
      fo[1] = chi[1];
      fo[2] = czero;
      fo[3] = czero;
      
    } else {
      fo[0] = czero;
      fo[1] = czero;
      fo[2] = chi[1];
      fo[3] = chi[0];
      
    } }
}


dpOxxxxx::dpOxxxxx(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;

	std::string fileName = "cl/oxxxxx.cl";
	name = "Oxxxxx";
	std::string programSource = getFile("./src/cl/oxxxxx.cl");
	
	const char * c = programSource.c_str();
	kernelString  = c;
	//strcpy(kernelString, c);
	
	//printf("\n%s\n", kernelString);
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	clErrChk(err);
	programCheck(err, context, program);
	
	kernel = clCreateKernel(program, "Oxxxxx", &err); clErrChk(err);
}

void dpOxxxxx::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
	
	//for(int i=0; pow(2,i)*sizeof(float)/(float) 1048576<dataMB;i++)
	//	Asize = pow(2,i);
	
	Asize = 1048576*dataMB/sizeof(float);
	MB = Asize * sizeof(float) / 1048576;
	
}

void dpOxxxxx::init(){
	Ain = new float[Asize];
	Aout = new float[Asize];
	if (!Aout || !Ain)
		fprintf(stderr, "error in dynamic allocation");
	
	generateArray(Ain, Asize);
	
	dataParameters.push_back(Asize);
	dataNames.push_back("nElements");
	
}

void dpOxxxxx::memoryCopyOut(){
	Ain_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err); clErrChk(err);
	Aout_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err); clErrChk(err);

	clErrChk(clEnqueueWriteBuffer(queue, Ain_d, CL_TRUE, 0, Asize*sizeof(float), Ain, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpOxxxxx::plan(){
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &Ain_d));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Aout_d));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &Asize));
	globalSize[0] = Asize;
}

int dpOxxxxx::execute(){
	//err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL); 
	clErrChk(err);
	if(err<0)
		return -1;
	clErrChk(clFinish(queue));
	return 0;
}

void dpOxxxxx::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, Aout_d, CL_TRUE, 0, Asize*sizeof(float), Aout, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpOxxxxx::cleanUp(){
	//printf("%f^2 = %f\n",Ain[Asize-1],Aout[Asize-1]);
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(Ain_d));
	clErrChk(clReleaseMemObject(Aout_d));
	delete[] Aout;
	delete[] Ain;
}

void dpOxxxxx::generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}