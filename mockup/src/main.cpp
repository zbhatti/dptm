#include <stdio.h>
#include "dpClient.hpp"
 
int main (int argc, const char* argv[]) {
	
	//take platform and device argument:
	
		dpClient cli1(0,0);
		dpClient cli2(1,0);
		dpClient cli3(1,1);
		dpClient cli4(2,0);
		dpClient cli5(2,1);
		dpClient* cliList[5] = {&cli1, &cli2, &cli3, &cli4, &cli5};
	
	//take task scan argument:
	for (int r=0;r<15;r++){
		for (int i=0;i<5;i++){
			fprintf(stderr, "Client:%d\n",i);
			//cliList[i]->runTaskScan("VectorAdd");
			//cliList[i]->runTaskScan("SquareArray");
			//cliList[i]->runTaskScan("FluidSimulation");
			//cliList[i]->runTaskScan("LUDecomposition");
			//cliList[i]->runTaskScan("FloydWarshall");
			//cliList[i]->runTaskScan("FWT");
			//cliList[i]->runTaskScan("NBody");
			//cliList[i]->runTaskScan("MatrixMultiplication");
			//cliList[i]->runTaskScan("MatrixTranspose");
			//cliList[i]->runTaskScan("Convolution");
			cliList[i]->runTaskScan("Array3dAverage");
			//cliList[i]->runTaskScan("RotateImage");
			//cliList[i]->runTaskScan("MersenneTwister");
			//cliList[i]->addTaskScan("FFT");
			//cliList[i]->printTimes();
			cliList[i]->printFile();
		}
	}
	
	return 0;
}

