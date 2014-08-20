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
	for (int r=0;r<1;r++){
		for (int j=5;pow(2,j)<=512;j++){
			for (int i=0;i<5;i++){
				fprintf(stderr, "Client:%d\n",i);
				
				/*
				fprintf(stderr,"Array3dAverage\n");
				cliList[i]->addWGScan("Array3dAverage",pow(2,j));
				//cliList[i]->addMBScan("Array3dAverage",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				fprintf(stderr,"Convolution\n");
				cliList[i]->addWGScan("Convolution",pow(2,j));
				//cliList[i]->addMBScan("Convolution",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				fprintf(stderr,"FloydWarshall\n");  
				cliList[i]->addWGScan("FloydWarshall",pow(2,j));
				//cliList[i]->addMBScan("FloydWarshall",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				fprintf(stderr,"FluidSimulation\n");
				cliList[i]->addWGScan("FluidSimulation",pow(2,j));
				//cliList[i]->addMBScan("FluidSimulation",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				fprintf(stderr,"FWT\n");
				cliList[i]->addWGScan("FWT",pow(2,j));
				//cliList[i]->addMBScan("FWT",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				
				fprintf(stderr,"LUDecomposition\n");
				cliList[i]->addWGScan("LUDecomposition",pow(2,j));
				//cliList[i]->addMBScan("LUDecomposition",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				
				fprintf(stderr,"MatrixMultiplication\n");
				cliList[i]->addWGScan("MatrixMultiplication",pow(2,j));
				//cliList[i]->addMBScan("MatrixMultiplication",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				fprintf(stderr,"MatrixTranspose\n");
				cliList[i]->addWGScan("MatrixTranspose",pow(2,j));
				//cliList[i]->addMBScan("MatrixTranspose",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				*/
				
				fprintf(stderr,"MonteCarloAsian\n");
				cliList[i]->addWGScan("MonteCarloAsian",pow(2,j));
				//cliList[i]->addMBScan("MonteCarloAsian",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				/*
				fprintf(stderr,"NBody\n");
				cliList[i]->addWGScan("NBody",pow(2,j));
				//cliList[i]->addMBScan("NBody",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				
				fprintf(stderr,"Reduction\n");
				cliList[i]->addWGScan("Reduction",pow(2,j));
				//cliList[i]->addMBScan("Reduction",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				
				fprintf(stderr,"SquareArray\n");
				cliList[i]->addWGScan("SquareArray",pow(2,j));
				//cliList[i]->addMBScan("SquareArray",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				
				fprintf(stderr,"VectorAdd\n");
				cliList[i]->addWGScan("VectorAdd",pow(2,j));
				//cliList[i]->addMBScan("VectorAdd",8,8,8);
				cliList[i]->runTasks();
				cliList[i]->printFile();
				*/
				
				//cliList[i]->printTimes();
			}
		}
	}
	
	return 0;
}

