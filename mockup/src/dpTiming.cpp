#include "dpTiming.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>

std::string dpTiming::getVariables(){
	std::stringstream ss;
	ss<<"device/C,";
	ss<<"kernel/C,";
	ss<<"workDimension/C,";
	for (unsigned int i = 0; i < dataNames.size();i++){
		ss<<dataNames.at(i)<<"/F,";
	}
	ss<<"xLocal/I,";
	ss<<"yLocal/I,";
	ss<<"zLocal/I,";
	ss<<"memoryCopyOut/F,";
	ss<<"plan/F,";
	ss<<"execute/F,";
	ss<<"memoryCopyIn/F,";
	ss<<"cleanUp/F";
	return ss.str();
}

std::string dpTiming::getTimes(){
	std::stringstream ss;
	ss<<device<<",";
	ss<<name<<",";
	ss<<workDimension<<",";
	for (unsigned int i = 0; i < data.size();i++){
		ss<<data.at(i)<<",";
	}
	ss<<localSize[0]<<",";
	ss<<localSize[1]<<","; 
	ss<<localSize[2]<<",";
	ss<<memoryCopyOut<<",";
	ss<<plan<<",";
	ss<<execute<<",";
	ss<<memoryCopyIn<<",";
	ss<<cleanUp;
	return ss.str();
}