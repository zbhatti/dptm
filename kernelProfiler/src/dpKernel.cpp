#include "dpKernel.hpp"
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

void dpKernel::FillerFunction(){
	printf("filler function\n");
}


//source:http://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
std::string dpKernel::getFile(std::string fileName){ //where fileName is 'src/cl/kernelName.cl'
	std::ifstream t(fileName.c_str());
	std::stringstream buffer;
	buffer << t.rdbuf();
	return buffer.str();
} 