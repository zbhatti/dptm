MaxClients=$1
Version=$2
sudo ls
x=1
while [ $x -le $MaxClients ]
do
  ./RunMultiAllTypes.sh 0 $x $Version 
  x=$(( $x + 1 ))
done
