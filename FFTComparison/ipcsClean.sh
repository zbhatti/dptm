#! /bin/sh
shmem=`ipcs -m|grep 'zxb0111'| awk '{print $2}'`
for sharedmem in $shmem;  do sudo ipcrm -m $sharedmem;  done

shmem=`ipcs -m|grep 'root'| awk '{print $2}'`
for sharedmem in $shmem;  do sudo ipcrm -m $sharedmem;  done
