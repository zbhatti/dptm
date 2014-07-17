#include "dpClient.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

//set up context and queue on a device and retrieve 
//device information for other methods
dpClient::dpClient(int platform, int device){
	unsigned int numDevices;
	char name[256];
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
	
	for (unsigned int i =0; i <taskList.size(); i++){
		gettimeofday(&start, NULL);
		taskList.at(i)->memoryCopyOut();
		gettimeofday(&finish, NULL);
		timeTmp.memoryCopyOut = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->plan();
		gettimeofday(&finish, NULL);
		timeTmp.plan = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->execute();
		gettimeofday(&finish, NULL);
		timeTmp.execute = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->memoryCopyIn();
		gettimeofday(&finish, NULL);
		timeTmp.memoryCopyIn = timeDiff(start,finish);
		
		gettimeofday(&start, NULL);
		taskList.at(i)->cleanUp();
		gettimeofday(&finish, NULL);
		timeTmp.cleanUp = timeDiff(start,finish);
		
		timeTmp.name = taskList.at(i)->name;
		timeTmp.localSize = taskList.at(i)->getLocalSize();
		
		timeList.push_back(timeTmp);
	}
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

//Loop through dimensions of a task:
void dpClient::addTaskScan(std::string name){
	workGroupSpace workDim;
	int i,j,k;
	j =0;
	k =0;
	//make a temporary kernel to read information from:
	workDim = kernelFactory.makeTask(name, context, queue)->workDimension;
	
	if (workDim == ONE_D){
		for (i=0; pow(2,i)<=MaxWorkGroupSize; i++)
			addTask(name, pow(2,i));
	}
	
	if (workDim == TWO_D){
		for(i=0; pow(2,i)*pow(2,j)<=MaxWorkGroupSize; i++){
			for(j=0; pow(2,i)*pow(2,j)<=MaxWorkGroupSize; j++){
				addTask(name, pow(2,i), pow(2,j));
			}
			j=0;
		}
	}
	
	if (workDim == THREE_D){
		for(i=0; pow(2,i)*pow(2,j)*pow(2,k)<=MaxWorkGroupSize; i++){
			for(j=0; pow(2,i)*pow(2,j)*pow(2,k)<=MaxWorkGroupSize; j++){
				for(k=0; pow(2,i)*pow(2,j)*pow(2,k)<=MaxWorkGroupSize; k++){
					addTask(name, pow(2,i), pow(2,j), pow(2,k));
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
	for (unsigned int i = 0; i < timeList.size(); i++){
		printf("%s\t%d\t%d\t%d\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\n",
			timeList.at(i).name.c_str(),
			(int)timeList.at(i).localSize[0],
			(int)timeList.at(i).localSize[1],
			(int)timeList.at(i).localSize[2],
			timeList.at(i).memoryCopyOut,
			timeList.at(i).plan,
			timeList.at(i).execute, 
			timeList.at(i).memoryCopyIn,
			timeList.at(i).cleanUp);
	}
}





