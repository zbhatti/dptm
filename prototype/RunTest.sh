sudo ls > /dev/null
echo Running DPTM Test
echo NBin=$1 
echo NClient=$2 
echo Repeat=$3 and 
echo 

GPU=0


# # Exit if directory exists
# if [ -e $dir ] ; then
#     Looks like $dir exists. Skipping.
#     exit 0
# fi

# echo Making output directory: $dir
# mkdir -pv $dir

GPU=0

NBins=$1
NClients=$2
Version=$4
TotalWires=2048
MaxWires=2048
MinWires=2048
NWireStep=1
Repeat=$3
UseCPU=0

SleepTime=5
RunTime=100  # 500 sec

echo GPU=$GPU
echo NClients=$NClients
echo Version=$Version

dir=logs/$NBins-$NClients-$Repeat-$Version
#echo Making sure output directory $dir is clean
#rm -rf $dir

# Exit if directory exists
if [ -e $dir ] ; then
    echo Looks like $dir exists. Skipping.
    exit 0
fi

echo Making output directory: $dir
mkdir -pv $dir

echo Starting DPTM.
sudo ./DPTM /var/tmp/socket $NBins $NClients $Repeat>$dir/DPTM.log 2>&1 & 
sleep 2

echo Starting Wrapper
sudo ./wrapper_FFT /var/tmp/socket $NBins $NClients $Repeat $GPU>$dir/Wrapper.csv 2>$dir/Wrapper.err.log &
sleep 2

echo Starting Clients
x=1
while [ $x -le $NClients ]
do
  echo Launching Client $x...
  #$NBins $NClients $Repeat
  sudo ./client /var/tmp/socket $NBins $NClients $MaxWires $NWireStep $Repeat $UseCPU $MinWires >$dir/Client-$x.csv 2>$dir/Client-$x.err.log & 
  sleep 1
  x=$(( $x + 1 ))
done

echo Monitoring Processes:

z=3
NClients=`ps aux | grep client | grep -v grep | grep -v sudo | wc -l`
echo $NClients Clients Found
NJobs=`jobs | grep Running | wc -l`
echo $NJobs Jobs Found
jobs

x=1
while [ $NJobs -ge $z ]
do 
    jobs
    sleep $SleepTime
    NClients=`ps aux | grep client | grep -v grep | grep -v sudo | wc -l`
    echo $NClients Clients Found
    NJobs=`jobs | grep Running | wc -l`
    echo $NJobs Jobs Found
    x=$(( $x + 1 ))
    if [ $x -gt $RunTime ] ; then
	echo Running for more than 30 mins. Quitting.
	./Clean.sh
    fi
	
done

./Clean.sh

echo Done Running DPTM Test $1 ,$2, $3.