#include <stdio.h>
#include "dpClient.hpp"
 
int main (int argc, const char* argv[]) {

	dpClient cli1(0,0);
	dpClient cli2(1,0);
	dpClient cli3(1,1);
	dpClient cli4(2,0);
	dpClient* cliList[4] = {&cli1, &cli2,&cli3, &cli4};
	
	for (int i=0;i<4;i++){
		cliList[i]->addTaskScan("FWT");
		//cliList[i]->addTaskScan("NBody");
		//cliList[i]->addTaskScan("FFT");
		//cliList[i]->addTaskScan("LUDecomposition"); cliList[i]->runTasks();
		//cliList[i]->addTaskScan("VectorAdd");
		//cliList[i]->addTaskScan("SquareArray"); cliList[i]->runTasks();
		//cliList[i]->addTaskScan("MatrixMultiplication"); cliList[i]->runTasks();
		//cliList[i]->addTaskScan("RotateImage"); cliList[i]->runTasks();
		//cliList[i]->addTaskScan("MatrixTranspose");
		//cliList[i]->addTaskScan("MersenneTwister"); cliList[i]->runTasks();
		//cliList[i]->addTaskScan("Convolution"); 
		cliList[i]->runTasks();
		printf("------------------------Client: %d-------------------------\n",i+1);
		cliList[i]->printTimes();
	}
	
	
	return 0;
}

