#include <stddef.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>//shared memory
#include <sys/ipc.h>//shared memory
#include <sys/shm.h>//shared memory
#include <sys/time.h> //for random seed and timing
#include "/Developer/NVIDIA/CUDA-5.5/include/cuda.h"//cuda functions
#include "/Developer/NVIDIA/CUDA-5.5/include/cuda_runtime.h"//cuda functions
#include "/Developer/NVIDIA/CUDA-5.5/include/cufft.h"//definition for cufftComplex
#define __float128 long double
#include <fftw3.h>//fftw3 functions
#include "sharedfunctions.h"
#define START gettimeofday(&start, NULL);
#define FINISH gettimeofday(&finish, NULL);

//used by wrapper and clients to make original socket connections
int client_socket_connect(char *argv[]){
	int sockfd;
	struct sockaddr_un serv_addr;
	
	bzero((char*)&serv_addr, sizeof(serv_addr));
	serv_addr.sun_family = AF_UNIX;
	serv_addr.sun_len = sizeof(serv_addr);
	strcpy(serv_addr.sun_path, argv[1]);
	if ((sockfd = socket(AF_UNIX, SOCK_STREAM,0)) < 0)
		return -1;
	if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(struct sockaddr_un)) < 0) 
		return -1;
	return sockfd;
}

void print_cufft(cufftComplex *A, int size){
	int i;
	for (i = 0; i < size; i++){
		printf("%4.0f + j%4.0f, ", A[i].x, A[i].y);
	}
	printf("\n");
}

void print_fftw(fftw_complex *A, int size){
	int i;
	for (i = 0; i < size; i++){
		printf("%4.0f + j%4.0f, ", A[i][0], A[i][1]);
	}
	printf("\n");
}

void convertcopy(fftw_complex *dst, cufftComplex *src, int size){
	int i;
	for (i = 0; i < size; i++){
		dst[i][0] = src[i].x;
		dst[i][1] = src[i].y;
	}
}

void avg_printer(double **p, int size){
	int i;
	double sum;
	
	sum = 0.0;
	for (i = 0; i < size; i++)
		sum += p[i][0]; 
	printf("%f,", sum / size);
	
	sum = 0.0;
	for (i = 0; i < size; i++)
		sum += p[i][1];
	printf("%f,", sum / size);
}

void rms_printer(double **p, int size){
	int i;
	double sum = 0.0;
	double val;
	for (i = 0; i < size; i++)
		sum += p[i][0] * p[i][0];
	val = sqrt(sum / size);
	printf("%f,", val);

	sum = 0.0;
	for (i = 0; i < size; i++)
		sum += p[i][1] * p[i][1];
	val = sqrt(sum / size);
	printf("%f,", val);
}

void component_statistics_printer(cufftComplex *A, fftw_complex *B, int size){
	int i;
	double **pdiff;
	pdiff = (double** ) malloc(size * sizeof(double*));
	for (i = 0; i < size; i++){
		pdiff[i] = (double*) malloc(2 * sizeof(double));
	}
	
	for (i = 0; i < size; i++){
		pdiff[i][0] = (A[i].x - B[i][0]) / ( (A[i].x + B[i][0])/2 ); //pct diff in Re
		pdiff[i][1] = (A[i].y - B[i][1]) / ( (A[i].y + B[i][1])/2 ); //pct diff in Im
	}
	
	avg_printer(pdiff, size);
	rms_printer(pdiff, size);
	
	for (i = 0; i < size; i++)
		free(pdiff[i]);
	free(pdiff);
}

//take list of complex numbers, return list of theta / magnitude
void mag_angle_statistics_printer(cufftComplex *A, fftw_complex *B, int size){
		int i;
	double **pdiff;
	double magA, magB, thetaA, thetaB;
	
	pdiff = (double** ) malloc(size * sizeof(double*));
	for (i = 0; i < size; i++){
		pdiff[i] = (double*) malloc(2 * sizeof(double));
	}
	
	for (i = 0; i < size; i++){
		//convert in place:
		magA = sqrt(A[i].x*A[i].x + A[i].y*A[i].y);
		magB = sqrt(B[i][0]*B[i][0] + B[i][1]*B[i][1]);
		thetaA = atan(A[i].y/A[i].x);
		thetaB = atan(B[i][1]/B[i][0]);
		pdiff[i][0] = (magA - magB) / ( (magA + magB)/2 ); //pct diff in Mag
		pdiff[i][1] = (thetaA - thetaB) / ( (thetaA + thetaB)/2 ); //pct diff in Theta
	}
	avg_printer(pdiff, size);
	rms_printer(pdiff, size);
	
	for (i = 0; i < size; i++)
		free(pdiff[i]);
	free(pdiff);
}

int make_req(request *req, cufftComplex *Ain, int size){
	req->shmid=-1;
	int count=0;
	while (req->shmid == -1){
	  req->shmid = shmget(IPC_PRIVATE, sizeof(cufftComplex)* size, IPC_CREAT | 666);
	  if(req->shmid == -1){
	    fprintf(stderr,"shmget failed\n");
	  }
	  if (count==1) return -1;
	  count+=1;

	}
	  req->Asize = size;
		
	  req->A = (cufftComplex*) shmat(req->shmid, NULL, 0); 
	  if( req->A == (void*) -1){
		printf("shmat failed");
		return -1;
		
	}
	
	memcpy(req->A, Ain, size * sizeof(cufftComplex) ); 
	
	/*
	for(i = 0; i < size; i++){
		req->A[i].x = rand() / (RAND_MAX/99.9 + 1);
		req->A[i].y = rand() / (RAND_MAX/99.9 + 1);
	}
	*/
	return 0;
}

void generate(cufftComplex *Ain, int size){
	for(int i = 0; i < size; i++){
		Ain[i].x = rand() / (RAND_MAX/99.9 + 1);
		Ain[i].y = rand() / (RAND_MAX/99.9 + 1);
	}
}

int main(int argc, char *argv[]){
  int sockfd, Arrsize, flag;
  long int delT;
  cufftComplex *Ain;
  fftw_complex *Afftw;
  fftw_plan planfftw;
  request out;
  struct timeval start, finish;
  srand(time(NULL));
  //connect to server
  sockfd = client_socket_connect(argv);	
  if (sockfd < 0)
    return -1;
  printf("NumberOfVectors,FullBinSize,SimultaneuousClients,CUFFT,FFTW3\n");

  //main loop 
  //      for (j = 0; j < 6*atoi(argv[2]); j++){ // NClients  loop j 0->6*NClients
  //	for (i = 0; i < 10; i++){  // loop i 0->10
  //	  Arrsize = i * pow(10,(j+2))/atoi(argv[2]); // NClients Arrsize= i * 10^(j+2) / NClients
  //	  if (i==0)
  //	    Arrsize = 10;
  //      for (k = 0; k < atoi(argv[4]); k++){ // Repeat
  // Logic copied from the single client

  int MaxWires=2000;
  int NWireStep=100;
  int Repeat=1;
  bool UseCPU=true;
  int MinWires=0;
  int NBins=1;
  int NClients=1;
  int NSamples = 3200;

  if (argc>2)
    NBins=atoi(argv[2]);
  
  if (argc>3)
    NClients=atoi(argv[3]);
  
  if (argc>4)
    MaxWires=atoi(argv[4]);
  
  if (argc>5)
    NWireStep=atoi(argv[5]);
  
  if (argc>6)
    Repeat=atoi(argv[6]);
  
  if (argc>7)
    UseCPU=atoi(argv[7]);
  
  if (argc>8)
    MinWires=atoi(argv[8]);
  
  fprintf(stderr,"Running %d times.\n",Repeat);

	Ain = (cufftComplex*) malloc(MaxWires * NSamples * sizeof(cufftComplex));
	if (!Ain){
		fprintf(stderr,"error in malloc Ain\n");
		return -1;
	}
	  
	//generate here with (&out, MAXSIZE)
	generate(Ain, MaxWires * NSamples);
	
  for (int r=0;r<Repeat;r++) {
    //main loop
    for (int i=MinWires; i <= MaxWires; i+=NWireStep){ 
   
		Arrsize = i*NSamples;
		if (i == 0)
			Arrsize = NSamples;

		if (make_req(&out, Ain, Arrsize) == -1){
			fprintf(stderr,"error in make req\n");
			return -1;
		}
    
		Afftw = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Arrsize);
		if (!Afftw){
			fprintf(stderr,"error in malloc Afftw\n");
			return -1;
		}
    
		memcpy(out.A, Ain, Arrsize * sizeof(cufftComplex)); //copy generated sequence into outgoing
		convertcopy(Afftw, Ain, Arrsize);  //copy generated sequence into fftw data type
		
		printf("%d,%d,%d,", Arrsize, NBins, NClients);
      
		//cufft:
		START
		send(sockfd, &out, sizeof(request), 0);
		shmdt((void*) out.A);
		recv(sockfd, &flag, sizeof(int), 0); //out.A has the transformed sequence
		FINISH
		delT = timediff(start, finish);
		printf("%ld,", delT);
    
		delT=-1;
		if (UseCPU) {
		//fftw3:
			START		
			planfftw = fftw_plan_dft_1d(Arrsize, Afftw, Afftw, FFTW_FORWARD, FFTW_ESTIMATE);
			fftw_execute(planfftw);
			FINISH
			delT = timediff(start, finish);
		}

		printf("%ld,", delT);
      
	//compare data:
	//print_cufft(Ain, Arrsize);
	//print_cufft(out.A, Arrsize);
	//print_fftw(Afftw, Arrsize);
	//component_statistics_printer(out.A, Afftw, Arrsize);
	//mag_angle_statistics_printer(out.A, Afftw, Arrsize);
	
      printf("\n");
      

      fftw_free(Afftw);
      fftw_destroy_plan(planfftw);
      //shmdt((void*) out.A);
      shmctl(out.shmid, IPC_RMID, NULL);
    }
    fflush(stdout);
  }
	free(Ain);
	close(sockfd);

  return 0;
}


