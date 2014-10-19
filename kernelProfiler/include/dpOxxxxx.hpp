#include "dpKernel.hpp"
#include "cmplx.h"
#ifndef __dpOxxxxx_H_INCLUDED__
#define __dpOxxxxx_H_INCLUDED__

class dpOxxxxx: public dpKernel{
	cl_mem P_d, Fo_d;
	double *P; // P is .. double; P is array of 4 ..double; P is array of 4 pointer to double
	cmplx *Fo;
	int Psize;
	
	public:
		dpOxxxxx(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init(); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(double *P, int);
};

#endif