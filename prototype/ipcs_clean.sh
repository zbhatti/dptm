#! /bin/sh
shmem=`ipcs -m|grep 'wheel'| awk '{print $2}'`
for sharedmem in $shmem;  do sudo ipcrm -m $sharedmem;  done
