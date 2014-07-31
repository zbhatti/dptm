#include "dpClient.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

//set up context and queue on a device and retrieve 
//device information for other methods
dpClient::dpClient(int platform, int device){
	unsigned int numDevices;
	int err;
	cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,0,0};
	clErrChk(clGetPlatformIDs(16, platform_ids, NULL));
	clErrChk(clGetDeviceIDs(platform_ids[platform], CL_DEVICE_TYPE_ALL, 16, device_ids, &numDevices));
	clErrChk(clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_NAME, sizeof(name), name, NULL));
	fprintf(stderr,"On Platform %s\n", name);
	clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_NAME, sizeof(name), name, NULL));
	clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(MaxWorkGroupSize), &MaxWorkGroupSize, NULL));
	clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(MaxComputeUnits), &MaxComputeUnits, NULL));
	clErrChk(clGetDeviceInfo(device_ids[device], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(MaxWorkDim), &MaxWorkDim, NULL));
	fprintf(stderr,"using device %s\n", name);
	props[1] = (cl_context_properties) platform_ids[platform];
	context = clCreateContext(props, 1, &device_ids[device], NULL, NULL, &err); clErrChk(err);
	queue = clCreateCommandQueue( context, device_ids[device], 0, &err); clErrChk(err);
}

void dpClient::runTasks(){
	dpTiming timeTmp;
	int err;
	
	for (unsigned int i =0; i <taskList.size(); i++){
		
		timeTmp.name = taskList.at(i)->name;
		timeTmp.localSize = taskList.at(i)->getLocalSize();
		timeTmp.data= taskList.at(i)->dataParameters;
		timeTmp.dataNames= taskList.at(i)->dataNames;
		timeTmp.device = name;
		//if block for enum to string:
		if (taskList.at(i)->workDimension == ONE_D)
			timeTmp.workDimension ="ONE_D";
		if (taskList.at(i)->workDimension == TWO_D)
			timeTmp.workDimension = "TWO_D";
		if (taskList.at(i)->workDimension == THREE_D) 
			timeTmp.workDimension = "THREE_D";
		
		gettimeofday(&start, NULL);
		taskList.at(i)->memoryCopyOut();
		gettimeofday(&finish, NULL);
		timeTmp.memoryCopyOut = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->plan();
		gettimeofday(&finish, NULL);
		timeTmp.plan = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		err = taskList.at(i)->execute();
		gettimeofday(&finish, NULL);
		if (err<0)
			timeTmp.execute = -1;
		else
			timeTmp.execute = timeDiff(start,finish);
			
		gettimeofday(&start, NULL);
		taskList.at(i)->memoryCopyIn();
		gettimeofday(&finish, NULL);
		timeTmp.memoryCopyIn = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->cleanUp();
		gettimeofday(&finish, NULL);
		timeTmp.cleanUp = timeDiff(start,finish);
		
		
		timeList.push_back(timeTmp);
	}
	
	taskList.clear();
}

//One Dimensional kernel task:
void dpClient::addTask(std::string name, int xLocal){
	if(xLocal > (int)MaxWorkDim[0])
		xLocal = MaxWorkDim[0]; 

	taskList.push_back(kernelFactory.makeTask(name, context, queue));
	//add error check here to make sure newly created kernel is of right dimension
	taskList.at(taskList.size()-1)->init(xLocal, 1, 1);
}

//Two Dimensional kernel task:
void dpClient::addTask(std::string name, int xLocal, int yLocal){
	if((xLocal>(int)MaxWorkDim[0])||(yLocal>(int)MaxWorkDim[1])||(xLocal*yLocal>(int)MaxWorkGroupSize))
		xLocal = yLocal = 16;

	taskList.push_back(kernelFactory.makeTask(name, context, queue));
	//add error check here to make sure newly created kernel is of right dimension
	taskList.at(taskList.size()-1)->init(xLocal, yLocal, 1);
}

//Three Dimensional kernel task:
void dpClient::addTask(std::string name, int xLocal, int yLocal, int zLocal){
	if((xLocal>(int)MaxWorkDim[0])||(yLocal>(int)MaxWorkDim[1])||(zLocal>(int)MaxWorkDim[2])||(xLocal*yLocal*zLocal>(int)MaxWorkGroupSize))
		xLocal=yLocal=zLocal = 8;
	
	taskList.push_back(kernelFactory.makeTask(name, context, queue));
	//add error check here to make sure newly created kernel is of right dimension
	taskList.at(taskList.size()-1)->init(xLocal, yLocal, zLocal);
}


void dpClient::runTaskScan(std::string name){
	workGroupSpace workDim;
	dpTiming timeTmp;
	int i,j,k;
	j =0;
	k =0;
	//make a temporary kernel to read information from:
	workDim = kernelFactory.makeTask(name, context, queue)->workDimension;
	
	if (workDim == ONE_D){
		for (i=0; pow(2,i)<=MaxWorkGroupSize; i++){
			addTask(name, pow(2,i));
			runTasks();
		}
	}
	
	if (workDim == TWO_D){
		for(i=0; pow(2,i)*pow(2,j)<=MaxWorkGroupSize; i++){
			for(j=0; pow(2,i)*pow(2,j)<=MaxWorkGroupSize; j++){
				addTask(name, pow(2,i), pow(2,j));
				runTasks();
			}
			j=0;
		}
	}
	
	if (workDim == THREE_D){
		for(i=0; pow(2,i)*pow(2,j)*pow(2,k)<=MaxWorkGroupSize; i++){
			for(j=0; pow(2,i)*pow(2,j)*pow(2,k)<=MaxWorkGroupSize; j++){
				for(k=0; pow(2,i)*pow(2,j)*pow(2,k)<=MaxWorkGroupSize; k++){
					addTask(name, pow(2,i), pow(2,j), pow(2,k));
					runTasks();
				}
				j=0;
			}
			k=0;
		}
	}
	
}

//helper function that returns time difference
float dpClient::timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000.0 + finish.tv_usec) - (start.tv_sec*1000000.0 + start.tv_usec))/(1000.0);
}

//print times, probably change to export the timeList instance
void dpClient::printTimes(){
	printf("%s\n",timeList.at(0).getVariables().c_str());
	for (unsigned int i = 0; i < timeList.size(); i++){
		if(i>0){
			if ( timeList.at(i).name.compare(timeList.at(i-1).name) )
				printf("%s\n",timeList.at(i).getVariables().c_str());
		}
		printf("%s\n", timeList.at(i).getTimes().c_str() );
	}
	timeList.clear();
}

void dpClient::printFile(){
	std::ofstream fileOut;
	std::ifstream fileIn;
	char tmp[100];
	strcpy(tmp, "analysis/");
	strcat(tmp, name);
	mkdir(tmp, 0777);
	
	for (unsigned int i=0; i<timeList.size(); i++){
		char tmpString[200];
		//get filename to save to at ./device/kernelname.log
		strcpy(tmpString,"analysis/");
		strcat(tmpString,name);
		strcat(tmpString,"/");
		strcat(tmpString, timeList.at(i).name.c_str());
		strcat(tmpString,".log");
		
		
		fileOut.open(tmpString, std::ofstream::app );
		fileIn.open(tmpString);
		
		//if the file is empty, add the variable names first
		if (isEmpty(fileIn))
			fileOut << timeList.at(i).getVariables().c_str() << "\n";

		fileOut << timeList.at(i).getTimes().c_str() << "\n";
		fileOut.close();
		fileIn.close();
	}
	timeList.clear();
}

//http://stackoverflow.com/questions/2390912/checking-for-an-empty-file-in-c
bool dpClient::isEmpty(std::ifstream& pFile){
    return pFile.peek() == std::ifstream::traits_type::eof();
}



