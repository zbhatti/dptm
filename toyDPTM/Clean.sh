echo Cleaning...

echo Killing Clients
sudo killall client

echo Killing DPTM
sudo killall DPTM

echo Killing wrapper_FFT
sudo killall wrapper_FFT

echo Cleaning IPCS
./ipcs_clean.sh

echo Checking for processes...
ps aux | grep client | grep -v grep
ps aux | grep DPTM | grep -v grep
ps aux | grep wrapper_FFT | grep -v grep

echo Done Cleaning.
