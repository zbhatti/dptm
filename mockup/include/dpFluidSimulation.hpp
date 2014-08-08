#include "dpKernel.hpp"
#ifndef __dpFluidSimulation_H_INCLUDED__
#define __dpFluidSimulation_H_INCLUDED__

class dpFluidSimulation: public dpKernel{
	
	int dims[2];                                /**< Dimension of LBM simulation area */
	int iterations;
	// 2D Host buffers
	cl_double *rho;                              /**< Density */
	cl_double2 *u;                               /**< Velocity */
	cl_double *h_if0, *h_if1234, *h_if5678;      /**< Host input buffers */
	cl_double *h_of0, *h_of1234, *h_of5678;      /**< Host output buffers */
	
	cl_double *v_ef0, *v_ef1234, *v_ef5678;      /**< Host Eq distribution buffers for verification */
	cl_double *v_of0, *v_of1234, *v_of5678;      /**< Host output buffers for verification */

	cl_bool *h_type;                            /**< Cell Type - Boundary = 1 or Fluid = 0 */
	cl_double *h_weight;                         /**< Weights for each direction */
	cl_double8 dirX, dirY;                       /**< Directions */

	// Device buffers
	cl_mem d_if0, d_if1234, d_if5678;           /**< Input distributions */
	cl_mem d_of0, d_of1234, d_of5678;           /**< Output distributions */
	cl_mem type;                                /**< Constant bool array for position type = boundary or fluid */
	cl_mem weight;                              /**< Weights for each distribution */
	cl_mem velocity;                            /**< 2D Velocity vector buffer */

	void reset();
	double computefEq(cl_double, double* , double, cl_double2);
				
	public:
		dpFluidSimulation(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		
		
};

#endif