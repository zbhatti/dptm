#include <stdio.h>
#include "dpClient.hpp"
 
int main (int argc, const char* argv[]) {
	
	dpClient cli1(0,0);
	/*
	cli1.addTask("SquareArray",1);
	cli1.addTask("SquareArray",32);
	cli1.addTask("SquareArray",64);
	cli1.addTask("SquareArray",16);
	cli1.addTask("MatrixMultiplication",32,64);
	cli1.addTask("MatrixMultiplication",32,8);
	cli1.addTask("MatrixMultiplication",16,32);
	cli1.addTask("MatrixMultiplication",16,16);
	cli1.addTask("RotateImage",32,64);
	cli1.addTask("RotateImage",16,16);
	cli1.addTask("RotateImage",32,8);
	cli1.addTask("RotateImage",16,32);
	cli1.addTask("FFT",1);
	cli1.addTask("FFT",64);
	cli1.addTask("FFT",8192);
	cli1.addTask("FFT",32);
	*/
	//cli1.addTaskScan("SquareArray");
	//cli1.addTaskScan("RotateImage");
	//cli1.addTaskScan("MatrixMultiplication");
	cli1.addTask("Convolution",16);
	cli1.addTask("MatrixTranspose",16,16);
	cli1.addTaskScan("MatrixTranspose");
	//cli1.addTaskScan("Convolution");
	cli1.addTask("MersenneTwister",16,16);
	//cli1.addTaskScan("MersenneTwister");
	cli1.runTasks();
	cli1.printTimes();
	
	dpClient cli2(1,0);
	cli2.addTask("MersenneTwister",16,16);
	cli2.addTask("MatrixTranspose",16,16);
	cli2.addTaskScan("MatrixTranspose");
	cli2.addTaskScan("MersenneTwister");
	//cli2.addTaskScan("SquareArray");
	//cli2.addTaskScan("MatrixMultiplication");
	//cli2.addTaskScan("RotateImage");
	cli2.runTasks();
	cli2.printTimes();
	
	dpClient cli3(1,1);
	cli3.addTaskScan("RotateImage");
	cli3.addTaskScan("MatrixMultiplication");
	cli3.addTaskScan("SquareArray");
	cli3.runTasks();
	cli3.printTimes();
	
	dpClient cli4(2,0);
	cli4.addTaskScan("MatrixMultiplication");
	cli4.addTaskScan("RotateImage");
	cli4.addTaskScan("SquareArray");
	cli4.runTasks();
	cli4.printTimes();
	
	dpClient cli5(2,1);
	cli5.addTaskScan("MatrixMultiplication");
	cli5.addTaskScan("SquareArray");
	cli5.addTaskScan("RotateImage");
	cli5.runTasks();
	cli5.printTimes();
	
	return 0;
}

