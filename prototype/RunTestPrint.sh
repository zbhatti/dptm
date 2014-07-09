sudo ls
echo "bin size is $1 and clients requested is $2" &
echo "when task is finished, 'sudo kill' DPTM process" &
echo "and 'ipcs' to check for rogue shared memory segments" &
sudo ./DPTM /var/tmp/socket $1 $2 & 
sleep 2
sudo ./wrapper_FFT /var/tmp/socket $1 $2 & 
sleep 2
x=1
while [ $x -le $2 ]
do
  echo "Launching Client $x.."
  sudo ./client /var/tmp/socket $1 $2 $3 & 
  sleep 1
  x=$(( $x + 1 ))
done
