#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>//using for sleep function and others
#include <stdlib.h>
#include <sys/un.h>
#include <stdio.h>
#include <sys/types.h>//shared memory
#include <sys/ipc.h>//shared memory
#include <sys/shm.h>//shared memory
#include <cuda.h>//cuda functions
#include <cuda_runtime.h>//cuda functions
#include <cufft.h>
#include <sys/time.h>//time funcion for clocking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#include "sharedfunctions.h"
#define START gettimeofday(&sT, NULL);
#define FINISH gettimeofday(&fT, NULL);

//code from stackexchange to help print cuda return messages
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true){
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

//used by wrapper and clients to make original socket connections
int clientSocketConnect(char *argv[]){
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

int main(int argc, char *argv[]){

int sockfd, r, v, Asize;
cufftComplex *massA_d;
bin k1;
int n[1] = {0};
struct timeval start, finish, sT, fT;

int num_devices, device;

cudaGetDeviceCount(&num_devices);

fprintf(stderr,"Number of devices found: %d\n",num_devices);

if (num_devices > 1) {

	int max_multiprocessors = 0, max_device = 0;
	for (device = 0; device < num_devices; device++) {
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);

		fprintf(stderr,"Device %d has %d multiProcessors. \n",device, properties.multiProcessorCount);

		if (max_multiprocessors < properties.multiProcessorCount) {
			max_multiprocessors = properties.multiProcessorCount;
			max_device = device;
		}
	}

	//optionally select GPU
	if (argc>5)
		max_device=atoi(argv[5]);

	fprintf(stderr,"Selected Device %d.\n",max_device);
	cudaSetDevice(max_device);
}



//cufft library variables
cufftHandle plancufft;

//connect to DPTM
sockfd = clientSocketConnect(argv);
if (sockfd < 0)
	return -1;	
printf("BinCondition,NumberRequests,SizePerClient,ShmemAtt,Shared2GPU,CufftPlan,CufftExec,Gpu2Shared,ShmemDet,TotalTimeSpent\n");
//MAIN LOOP:	
while(1){

	recv(sockfd, &k1, sizeof(bin), 0); //sets sockfd write in select
	printf("%s,%d,%d,", k1.bincondition, k1.nreqs, k1.Asize);
	gettimeofday(&start, NULL);		

	Asize = k1.Asize;
	n[0] = Asize;


	//attach to shared memories
	START
	for(r = 0; r < k1.nreqs; r++){
		k1.shmid_ptrs[r] = (cufftComplex*) shmat(k1.shmid[r], NULL, 0);
		if( k1.shmid_ptrs[r] == (void*) -1){
			printf("shmat failed");
			return -1;
		}
	}
	FINISH
	printf("%ld,", timediff(sT, fT));


	//copy the sequences to device
	START
	gpuErrchk(cudaMalloc(&massA_d, Asize * k1.nreqs * sizeof(cufftComplex) ));
	for (r = 0, v = 0; r < k1.nreqs; r++){
		gpuErrchk(cudaMemcpy(&massA_d[v], k1.shmid_ptrs[r], sizeof(cufftComplex) * Asize, cudaMemcpyHostToDevice));
		v = v + Asize;
	}
	FINISH
	printf("%ld,", timediff(sT, fT));

	//create plan for cufft
	START
	if (cufftPlanMany(&plancufft, 1, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, k1.nreqs) != CUFFT_SUCCESS){
		//cufftplan failed (should probably fit this into a function)
		gpuErrchk(cudaFree(massA_d));
		for(r = 0; r < k1.nreqs; r++){
			shmdt((void*) k1.shmid_ptrs[r]);
		}
		cufftDestroy(plancufft);
		gettimeofday(&finish, NULL);
		send(sockfd, &k1, sizeof(bin), 0);
		continue;
	}
	FINISH
	printf("%ld,", timediff(sT, fT) );

	//execute cufft
	START		
	if (cufftExecC2C(plancufft, massA_d, massA_d, CUFFT_FORWARD)!= CUFFT_SUCCESS){
		//cufftexec failed (should probably fit this into a function)
		gpuErrchk(cudaFree(massA_d));
		for(r = 0; r < k1.nreqs; r++){
			shmdt((void*) k1.shmid_ptrs[r]);
		}
		cufftDestroy(plancufft);
		gettimeofday(&finish, NULL);
		send(sockfd, &k1, sizeof(bin), 0);
		continue;
	}
	FINISH
	printf("%ld,", timediff(sT, fT) );

	if (cudaThreadSynchronize() != cudaSuccess){
		printf("error in synch");
		return -1;	
	}

	//copy massA back in to the host pointers
	START
	for(r = 0, v = 0; r < k1.nreqs; r++){
		gpuErrchk(cudaMemcpy(k1.shmid_ptrs[r], &massA_d[v], sizeof(cufftComplex) * Asize, cudaMemcpyDeviceToHost));
		v = v + Asize;	
	}
	FINISH
	printf("%ld,", timediff(sT, fT) );

	gettimeofday(&finish, NULL);
	k1.timespent = timediff(start, finish);

	//free csr from device
	gpuErrchk(cudaFree(massA_d));


	//detach from shared mems
	START
	for(r = 0; r < k1.nreqs; r++){
		shmdt((void*) k1.shmid_ptrs[r]);
	}
	FINISH
	printf("%ld,", timediff(sT, fT));

	cufftDestroy(plancufft);

	send(sockfd, &k1, sizeof(bin), 0);

	printf("%ld\n", k1.timespent);
}



}
