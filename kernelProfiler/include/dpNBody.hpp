#include "dpKernel.hpp"
#ifndef __dpNBody_H_INCLUDED__
#define __dpNBody_H_INCLUDED__

class dpNBody: public dpKernel{
	
	int numBodies;                      /**< number of particles */
	int nSteps;		                      /**< number of steps to calculate*/				
	cl_float delT;                      /**< dT (timestep) */
	cl_float espSqr;                    /**< Softening Factor*/
	cl_float* initPos;                  /**< initial position */
	cl_float* initVel;                  /**< initial velocity */
	cl_float* vel;                      /**< Output velocity */
	cl_mem particlePos[2];              // positions of particles
	cl_mem particleVel[2];              // velocity of particles
	cl_float* pos;      								/**< Output position */
	
	public:
		dpNBody(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		float random(float , float );
};

#endif