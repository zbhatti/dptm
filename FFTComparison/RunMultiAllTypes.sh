sudo ls > /dev/null
echo Running Multi Client Single Wire Test For All Types Consecutively

GPU=$1
NClients=$2
Version=$3
NThreads=12
MaxVectors=200000
MinVectors=200000
StepSize=1
Repeat=5
UseCPU=0
NThreads=12

SleepTime=5
RunTime=100

echo GPU=$GPU
echo NClients=$NClients
echo Version=$Version

################################################################
echo CUDA Portion
dir=logs/CUDA-$GPU-$NClients-$Version

# Exit if directory exists
if [ -e $dir ] ; then
    echo Looks like $dir exists. Skipping.
    exit 0
fi

echo Making output directory: $dir
mkdir -pv $dir

echo Starting Clients
echo $MaxVectors $MinVectors $StepSize

x=1
while [ $x -le $NClients ]
do
  echo Launching Client $x...
  echo ./ClientStandaloneCUDA $GPU $MaxVectors $MinVectors $StepSize $Repeat
  ./ClientStandaloneCUDA $GPU $MaxVectors $MinVectors $StepSize $Repeat 1>$dir/Client-$x.csv 2>$dir/Client-$x.err.log & 
#  sleep 1
  x=$(( $x + 1 ))
done

echo Monitoring Processes:

z=0
NClientsLeft=`ps aux | grep Client | grep -v grep | grep -v sudo | wc -l`
echo $NClientsLeft Clients Found
NJobs=`jobs | grep Running | wc -l`
echo $NJobs Jobs Found
jobs

x=1
while [ $NJobs -gt $z ]
do 
    jobs
    sleep 5
    NClientsLeft=`ps aux | grep Client | grep -v grep | grep -v sudo | wc -l`
    echo $NClientsLeft Clients Found
    NJobs=`jobs | grep Running | wc -l`
    echo $NJobs Jobs Found
    x=$(( $x + 1 ))
    if [ $x -gt $RunTime ] ; then
	echo Running for more than 30 mins. Quitting.
	killall ClientStandaloneCUDA
    fi
done
./ipcs_clean.sh
killall ClientStandaloneCUDA

echo Done Running CUDA 
################################################################

echo OpenCL Portion
dir=logs/OpenCL-$GPU-$NClients-$Version

# Exit if directory exists
if [[ -e $dir ]] ; then
    echo Looks like $dir exists. Skipping.
    exit 0
fi

echo Making output directory: $dir
mkdir -pv $dir

echo Starting Clients
echo $MaxVectors $MinVectors $StepSize

x=1
while [ $x -le $NClients ]
do
  echo Launching Client $x...
  echo ./ClientStandaloneOpenCL $GPU $MaxVectors $MinVectors $StepSize $Repeat
  ./ClientStandaloneOpenCL $GPU $MaxVectors $MinVectors $StepSize $Repeat 1>$dir/Client-$x.csv 2>$dir/Client-$x.err.log & 
#  sleep 1
  x=$(( $x + 1 ))
done

echo Monitoring Processes:

z=0
NClientsLeft=`ps aux | grep Client | grep -v grep | grep -v sudo | wc -l`
echo $NClientsLeft Clients Found
NJobs=`jobs | grep Running | wc -l`
echo $NJobs Jobs Found
jobs

x=1
while [ $NJobs -gt $z ]
do 
    jobs
    sleep 5
    NClientsLeft=`ps aux | grep Client | grep -v grep | grep -v sudo | wc -l`
    echo $NClientsLeft Clients Found
    NJobs=`jobs | grep Running | wc -l`
    echo $NJobs Jobs Found
    x=$(( $x + 1 ))
    if [ $x -gt $RunTime ] ; then
	echo Running for more than 30 mins. Quitting.
	killall ClientStandaloneOpenCL
    fi
done
./ipcs_clean.sh
killall ClientStandaloneOpenCL

echo Done Running OpenCL

##########################################################################

echo FFTW Portion
dir=logs/FFTW-$NThreads-$NClients-$Version

# Exit if directory exists
if [[ -e $dir ]] ; then
    echo Looks like $dir exists. Skipping.
    exit 0
fi

echo Making output directory: $dir
mkdir -pv $dir

echo Starting Clients
echo  $MaxVectors $MinVectors $StepSize

x=1
while [ $x -le $NClients ]
do
  echo Launching Client $x...
  echo ./ClientStandaloneCPU $NThreads $MaxVectors $MinVectors $StepSize $Repeat
  ./ClientStandaloneCPU $NThreads $MaxVectors $MinVectors $StepSize $Repeat 1>$dir/Client-$x.csv 2>$dir/Client-$x.err.log & 
#  sleep 1
  x=$(( $x + 1 ))
done

echo Monitoring Processes:

z=0
NClientsLeft=`ps aux | grep Client | grep -v grep | grep -v sudo | wc -l`
echo $NClientsLeft Clients Found
NJobs=`jobs | grep Running | wc -l`
echo $NJobs Jobs Found
jobs

x=1
while [ $NJobs -gt $z ]
do 
    jobs
    sleep 5
    NClientsLeft=`ps aux | grep Client | grep -v grep | grep -v sudo | wc -l`
    echo $NClientsLeft Clients Found
    NJobs=`jobs | grep Running | wc -l`
    echo $NJobs Jobs Found
    x=$(( $x + 1 ))
    if [ $x -gt $RunTime ] ; then
	echo Running for more than 30 mins. Quitting.
	killall ClientStandaloneCPU
    fi
done
./ipcs_clean.sh
killall ClientStandaloneCPU

echo Done Running FFTW