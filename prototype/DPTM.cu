#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/un.h>
#include <stdio.h>
#include <sys/types.h>//shared memory
#include <sys/ipc.h>//shared memory
#include <sys/shm.h>//shared memory
#include <sys/select.h>
#include <sys/time.h>//time argument for clocking
#include <errno.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>//definition for cufftComplex
#define MAXDATAPTS 5000 //max data pts used in linear fit
#include "sharedfunctions.h"

//datapts to provide linear fitted line
typedef struct{
	int nvect[MAXDATAPTS]; //number of vectors associated with time spent at gpu
	int tgpu[MAXDATAPTS]; //time associated with number of vectors requested to gpu
	int npts;
	float m;
	float b;
}linearFit;

void copyBin(bin *in, bin *out, int nreq){
  
  if (nreq>in->nreqs || nreq==-1)
    nreq=in->nreqs;

  (*out).nreqs=nreq;
  (*out).Asize=(*in).Asize;

  for (int i=0; i<nreq;i++) {
    (*out).shmid[i]=(*in).shmid[i];
    (*out).shmid_ptrs[i]=(*in).shmid_ptrs[i];
    (*out).clifd[i]=(*in).clifd[i];
  }

  (*out).timespent=(*in).timespent;

  for (int i=0; i<15;i++) {
    (*out).bincondition[i]=(*in).bincondition[i];
  }
}

void trimBin(bin *in, bin *out, int nreq) {
  if (out!=0)
    copyBin(in,out,nreq);

  bin temp;

  if (nreq>in->nreqs)
    nreq=in->nreqs;

  copyBin(in,&temp,-1);

  (*in).nreqs=temp.nreqs-nreq;

  int  j=0;

  for (int i=nreq; i<in->nreqs;i++) {
    (*in).shmid[j]=temp.shmid[i];
    (*in).shmid_ptrs[j]=temp.shmid_ptrs[i];
    (*in).clifd[j]=temp.clifd[i];
    j++;
  }
  
  if (j!=in->nreqs)
    printf("Problem trimming: %d %d",j,in->nreqs);

}

//takes command line argument for socket directory and binds dptm as the server
int serverSocketConnect(char *argv[]){
  int serverSock;
  //servlen used for linux
  struct sockaddr_un serv_addr;
  if ((serverSock = socket(AF_UNIX,SOCK_STREAM,0)) < 0){
    printf("1st error in serverSocketConnect");
    return -1;
  }
  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sun_family = AF_UNIX;
  serv_addr.sun_len = sizeof(serv_addr);
  strcpy(serv_addr.sun_path, argv[1]);
  unlink(serv_addr.sun_path); 
  
  if ( bind( serverSock, (struct sockaddr*) &serv_addr, sizeof(struct sockaddr_un)) < 0){
    printf("2nd error in serverSocketConnect.\n");
    printf("ERROR: %s\n", strerror(errno));
    return -1;
  }
  listen(serverSock, 15);
  return serverSock;
}

//takes npts of x's and y's to find m,b in y=mx+b
void getLinearCoeff(linearFit *T){
  int i, n;
  n = T->npts;
  float sumx, sumx2, sumy, sumy2, sumxy;
  for (i = 0; i < n; i++){
    sumx += T->nvect[i];
    sumx2 += T->nvect[i] * T->nvect[i];
    sumxy += T->nvect[i] * T->tgpu[i];
    sumy += T->tgpu[i];
    sumy2 += T->tgpu[i] * T->tgpu[i];
  }
  T->m = (n*sumxy - sumx*sumy)/ (n*sumx2 - (sumx*sumx));
  T->b = (sumy*sumx2 -sumx*sumxy) / (n*sumx2 - (sumx*sumx));
}

//takes socket of gpu, array of (number of clients * arrsize) and corresponding measured times, number of datapoints,linear coefficients
int receiveBin(int gpuSock, linearFit *binTime){
  int flag, i;
  bin binIn;

  printf("Receving bin...\n");
  //this will block if gpu is not at write stage
	if (recv(gpuSock, &binIn, sizeof(bin), 0) <= 0){
	  printf("wrapper closed at %d\n", gpuSock);
	  close(gpuSock);
	  return -1;
	}
	for (i = 0; i <binIn.nreqs; i++)
	  send(binIn.clifd[i], &flag, sizeof(int), 0);
	
	//not calculating time estimation
	/*
	  binTime->nvect[binTime->npts] = binIn.nreqs * binIn.Asize;
	  binTime->tgpu[binTime->npts] = binIn.timespent;
	  binTime->npts = binTime->npts + 1;
	  if ( (binTime->npts) >= 3)
	  getLinearCoeff(binTime);
	*/
	return 0;
}

//takes a bin and sends it to gpu, then resets the bin
void sendBin(int gpuSock, bin *binOut, char *reason, int fullBin){

  bin temp;
  trimBin(binOut,&temp,fullBin);
  strcpy(temp.bincondition, reason);
  printf("Sending bin reason: %s...\n",reason);
  send(gpuSock, &temp, sizeof(bin), 0);
}

void addRequestToBin(bin *binOut, request *newReq, int clifd){
  printf("Adding new request... Total requests: %d\n", binOut->nreqs+1);
  binOut->Asize = newReq->Asize;
  binOut->shmid[binOut->nreqs] = newReq->shmid; 
  binOut->clifd[binOut->nreqs] = clifd;	
  binOut->nreqs = binOut->nreqs + 1;
}

int checkWrapper(int gpuSock, fd_set *rfds, linearFit *binTime, struct timeval *idleStart, int gpuBusy, int block){
  if(gpuBusy==1){
    printf("Checking Wrapper Status... "); 
    if (FD_ISSET(gpuSock, rfds) || block==1){  // GPU is Done... or we are going to wait
      printf("Wrapper has finished, fetching results... "); 
      if (receiveBin(gpuSock, binTime) == -1) // Get the results.
		return -1;
      gpuBusy = 0;
      gettimeofday(idleStart, NULL);
	
    } else { //GPU is still working
      printf("Wrapper still working... "); 
      gpuBusy=1;
    }
  } else {
    printf("Wrapper is free... "); 
  }
  printf("\n");
  return gpuBusy;
}  

int main(int argc, char *argv[]){
  int serverSock, newSock, gpuSock, fullBin, i, gpuBusy;
  long int timeIdle, timeEstimate;
  socklen_t cliLen;
  struct sockaddr_un cliAddr;
  struct timeval selectTimeout, idleStart, idleFinish, timeStart, timeFinish;
  fd_set rfds, mfds;
  request newReq;
  bin binQueue,binOut;
  linearFit binTime;
  bool processedFullBin=false;
  
  cliLen = sizeof(cliAddr);
  fullBin = atoi(argv[2]); 
  
  //setting high initial values until we have enough data pts to use these values
  binTime.m = 9999999999; 
  binTime.b = 9999999999;
  binTime.npts = 0;
  binOut.nreqs = 0;
  binOut.Asize = 0;
  copyBin(&binOut,&binQueue,256);

  gpuBusy = 0;
  
  selectTimeout.tv_sec  = 0;
  selectTimeout.tv_usec = 500000; //0.5 second
  
  serverSock = serverSocketConnect(argv);
  if (serverSock <= -1)
    return -1;
  
  //connect the GPU program (will block)
  gpuSock = accept(serverSock, (struct sockaddr *) &cliAddr, &cliLen);
  
  FD_ZERO(&rfds);
  FD_ZERO(&mfds);
  FD_SET(serverSock, &mfds);
  FD_SET(gpuSock, &mfds);
  
  //main loop
	while(1){
		gettimeofday(&timeStart, NULL);

		// select marks sockets as ready if they won't block
		rfds = mfds;
		//add timeout for select.... Does this have to be done in the loop?... yes, the 
		if (select(FD_SETSIZE, &rfds, NULL, NULL, &selectTimeout) == -1){
			printf("error in select");
			return -1;
		}

		// Step 1: Query Wrappper to see if busy. If so continue. If just finished, get results.
		printf("Step 1... check Wrapper. "); 
		gpuBusy=checkWrapper(gpuSock, &rfds, &binTime, &idleStart,  gpuBusy,0);
		gettimeofday(&timeFinish, NULL);
		printf("Step 1 time: %ld,  \n", timediff(timeStart, timeFinish) );

		// Step 2: Check for any new clients.
		printf("Step 2... Check for new clients. "); 
		if (FD_ISSET(serverSock, &rfds)) {//New client

			newSock = accept(serverSock, (struct sockaddr *) &cliAddr, &cliLen); 
			printf("New client at socket %d. ",newSock); 

			if (newSock == -1){
				printf("error in accept");
				return -1;
			}
			//client socket opened
			FD_SET(newSock, &mfds);
		}
		else {
			printf("No new clients. "); 
		}
		gettimeofday(&timeFinish, NULL);
		printf("Step 2 time: %ld, ", timediff(timeStart, timeFinish) );
		printf("\n"); 

		//Step 3: Check the clients and collect all results
		printf("Step 3... Looping over sockets looking for clients. "); // We should keep a list of client sockets
		for (i = serverSock+1; i < FD_SETSIZE; i++){ //FD_SETSIZE should be highest numbered socket

			if (i==gpuSock)
				continue;

			if (binQueue.nreqs >= fullBin) { // We have enough requests... lets process them. 
				printf("Bin full... ");
				gpuBusy=checkWrapper(gpuSock, &rfds, &binTime, &idleStart,  gpuBusy,0);   
				if (gpuBusy==0) {
					printf("GPU free so submitting. \n"); 
					sendBin(gpuSock, &binQueue, "fullBin",fullBin);
					gpuBusy = 1;
				}
			}

			if (FD_ISSET(i, &rfds)){ 	//a client is making a request
				printf("Client %d making request. ",i); 

				if (recv(i, &newReq, sizeof(request), 0) <= 0){
					printf("Client %d closed. ",i); 
					//client socket closed
					close(i);
					FD_CLR(i, &mfds);
					continue;
				}

				// Check if newest request mismatches previous request.
				if ( (newReq.Asize != binQueue.Asize) && (binQueue.Asize != 0) ){
					printf("Size mismatch. Waiting for wrapper to finish. Then sending remaining requests.");	

					while(binQueue.nreqs>0){
						gpuBusy=checkWrapper(gpuSock, &rfds, &binTime, &idleStart,  gpuBusy,1);   
						if (gpuBusy==0) 
							sendBin(gpuSock, &binQueue, "mismatch", fullBin);
						gpuBusy = 1;
					}
				}

				// Now add request
				addRequestToBin(&binQueue, &newReq, i);
			}
		}
		gettimeofday(&timeFinish, NULL);
		printf("Step 3 time: %ld, \n", timediff(timeStart, timeFinish) );

		// Step 4: Query Wrappper to see if busy. If so continue. If just finished, get results.
		printf("Step 4... check Wrapper again. ");  
		gpuBusy=checkWrapper(gpuSock, &rfds, &binTime, &idleStart,  gpuBusy,0);	
		gettimeofday(&timeFinish, NULL);
		printf("Step 4 time: %ld,  \n", timediff(timeStart, timeFinish) );

		// Step 5: If GPU is free, process what we have in hand.
		if ((gpuBusy == 0) && (binQueue.nreqs > 0)){
			printf("Step 5... telling Wrapper to process with %d requests.",binQueue.nreqs); 
			sendBin(gpuSock, &binQueue, "gpufree", fullBin);
			gpuBusy = 1;
		} 
		else {
			printf("Step 5... Waiting for Wrapper to finish.");
			gpuBusy=checkWrapper(gpuSock, &rfds, &binTime, &idleStart,  gpuBusy, 1);
		}
		gettimeofday(&timeFinish, NULL);
		printf("Step 5 time: %ld,  \n", timediff(timeStart, timeFinish) );

		gettimeofday(&idleFinish, NULL);
		timeIdle = timediff(idleStart, idleFinish);		
		//timeEstimate not used until a better line fitting function is found
		//timeEstimate = binTime.m * (binOut.Asize * binOut.nreqs) + binTime.b; y = mx + b

		fflush(stdout);
		fflush(stderr);

		if (gpuBusy==0 && binOut.nreqs==0){
			printf("GPU Idle and no requests... sleeping for 1 second.\n"); 
			sleep(1); // Ideally, the clients and the wrapper can wake this process up by sending signal.
		}
		
	}//end while(1) loop
	
}
