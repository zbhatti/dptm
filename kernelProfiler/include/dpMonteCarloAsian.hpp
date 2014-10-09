#include "dpKernel.hpp"
#include <math.h>
#include <malloc.h>
#include <string.h>
#ifndef __dpMonteCarloAsian_H_INCLUDED__
#define __dpMonteCarloAsian_H_INCLUDED__

/*
 *  structure for attributes of Monte Carlo
 *  simulation
 */

template<typename T>
struct MonteCarloAttrib
{
    T strikePrice;
    T c1;
    T c2;
    T c3;
    T initPrice;
    T sigma;
    T timeStep;
};

class dpMonteCarloAsian: public dpKernel{
	
	// Declare attribute structure
	MonteCarloAttrib<cl_float4> attributes;
	cl_int steps;                       /**< Steps for Asian Monte Carlo simution */
	cl_float initPrice;                 /**< Initial price */
	cl_float strikePrice;               /**< Strike price */
	cl_float interest;                  /**< Interest rate */
	cl_float maturity;                  /**< maturity */

	cl_int noOfSum;                     /**< Number of excersize points */
	cl_int noOfTraj;                    /**< Number of samples */

	cl_float *sigma;                    /**< Array of sigma values */
	cl_float *price;                    /**< Array of price values */
	cl_float *vega;                     /**< Array of vega values */

	cl_uint *randNum;                   /**< Array of random numbers */

	cl_float *priceVals;                /**< Array of price values for given samples */
	cl_float *priceDeriv;               /**< Array of price derivative values for given samples */

	cl_mem priceBuf;                    /**< CL memory buffer for sigma */
	cl_mem priceDerivBuf;               /**< CL memory buffer for price */
	cl_mem randBuf;                     /**< CL memroy buffer for random number */

	cl_int width;												//dependet on 
	cl_int height;											//dependent on noOfTraj

	int iterations;                     /**< Number of iterations for kernel execution */
	int vectorWidth;
	
	
	public:
		dpMonteCarloAsian(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		/**
		 * @brief Accepts rand, price & priceDeriv cl_mem objects and sets it to kernel object
		 */
		void setKernelArgs(int step, cl_mem *rand, cl_mem *price, cl_mem *priceDeriv);
};

#endif

