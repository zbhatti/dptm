#include <stddef.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>//shared memory
#include <sys/ipc.h>//shared memory
#include <sys/shm.h>//shared memory
#include <time.h> //for random seed and timing
#include <sys/time.h> //for random seed and timing
#include <errno.h>
#include <string.h>
#include <stdbool.h>
#define __float128 long double
#include <fftw3.h>
#define BEGIN gettimeofday(&start, NULL);	
#define END gettimeofday(&finish, NULL); time=timeDiff(start, finish);

void printFftw(fftw_complex *A, int Asize){
	int i;
	for (i = 0; i < Asize; i++){
		printf("%4.0f + j%4.0f, ", A[i][0], A[i][1]);
	}
	printf("\n");
}

void generate(fftw_complex* A, int Asize){
	int i;
	srand(time(NULL));
	for (i = 0; i < Asize; i++){
		A[i][0] = rand() / (RAND_MAX/99.9 + 1);
		A[i][1] = rand() / (RAND_MAX/99.9 + 1);
	}
}

float timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000 + finish.tv_usec) - (start.tv_sec*1000000 + start.tv_usec))/(1000);
}

int main(int argc, char *argv[]){
	int Asize, i, shmid, ret;
	struct timeval start, finish;
	float time;

	fftw_complex *Ain, *A_h, *Afftw;
	fftw_plan planfftw;

	int Nthreads=1;
	int MaxVectors=2000;
	int MinVectors=0;
	int StepSize=100;
	int Repeat=1;
	
	if (argc>1)
		Nthreads=atoi(argv[1]);
	
	if (argc>2)
	  MaxVectors=atoi(argv[2]);

	if (argc>3)
	  MinVectors=atoi(argv[3]);

	if (argc>4)
	  StepSize=atoi(argv[4]);

	if (argc>5)
	  Repeat=atoi(argv[5]);

	
	shmid = shmget(IPC_PRIVATE, MaxVectors * sizeof(fftw_complex), IPC_CREAT | 0666);
	if (shmid == -1)
		printf("shmid: %s\n", strerror(errno));
	
	Ain = (fftw_complex*) shmat(shmid, NULL, 0);
	if (!Ain){
		printf("error in shmget\n");
		return -1;
	}
	
	ret = fftw_init_threads();
	if (ret == 0){
		printf("failed to initialize multithreads\n");
		return -1;
	}
	generate(Ain, MaxVectors);
	printf("nVectors,memcpyOut,plan,execu,memcpyIn\n");
	
	//main loop
	for (int r=0;r<Repeat;r++) {
		for (i=MinVectors; i <= MaxVectors; i+=StepSize){
			Asize = i;
			if (i == 0)
				Asize = 2;
			printf("%d,", Asize);
			
			A_h = (fftw_complex*) malloc(Asize * sizeof(fftw_complex));
			if (!A_h){
			  fprintf(stderr,"error in malloc 2");
			  return -1;
			}
			
			BEGIN
			Afftw = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Asize);
			if (!Afftw){
				fprintf(stderr,"error in malloc 3");
				return -1;
			}
			memcpy(Afftw, Ain, sizeof(fftw_complex)*Asize);
			END
			printf("%0.3f,", time);
			
			BEGIN
			fftw_plan_with_nthreads(Nthreads);
			planfftw = fftw_plan_dft_1d(Asize, Afftw, Afftw, FFTW_FORWARD, FFTW_ESTIMATE);
			END
			printf("%0.3f,",time);
			
			BEGIN
			fftw_execute(planfftw);
			END
			printf("%0.3f,",time);
			
			BEGIN
			memcpy(A_h, Afftw, sizeof(fftw_complex)*Asize);
			END
			printf("%0.3f,\n",time);
			
			fflush(stdout);
			fftw_free(Afftw); 
			fftw_destroy_plan(planfftw);
			free(A_h);
		}
	}
	void fftw_cleanup_threads(void);
	shmdt( (void*) Ain);
	shmctl(shmid, IPC_RMID, NULL);
	return 0;
}

