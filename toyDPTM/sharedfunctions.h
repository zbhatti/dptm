//place shared functions in here and add #include "sharedfunctions.h" to source files
#include <sys/socket.h>
#include <sys/time.h> //for random seed and timing

//used by DPTM and client
typedef struct{
	int shmid;
	int Asize; 
	int clifd; //set by server
	cufftComplex *A; //client's pointer to matrix
}request;

//used by DPTM and GPU
typedef struct{ 
	int nreqs;
	int Asize; //# of elements in each sequence A
	int shmid[256];
	cufftComplex *shmid_ptrs[256]; // only used in wrapper
	int clifd[256];
	long int timespent;
	char bincondition[15];
}bin;

//can use anywhere for timing
/*
	EXAMPLE USAGE:
	
	gettimeofday(&sT, NULL);
	THING TO TIME
	gettimeofday(&sF, NULL);
	delT = timediff(sT, fT);
	printf("Reason: %ld, \n" delT);
*/

long int timediff(struct timeval start, struct timeval finish){
	return (finish.tv_sec * 1000000 + finish.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
}

