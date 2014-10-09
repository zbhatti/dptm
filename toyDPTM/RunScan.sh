echo Running Tests $1

NClients=8
Repeat=100

x=1
while [ $x -le $1 ]
do
    ./RunTest.sh 1 $NClients $Repeat $x
    ./RunTest.sh 2 $NClients $Repeat $x
    ./RunTest.sh 3 $NClients $Repeat $x
    ./RunTest.sh 4 $NClients $Repeat $x
    ./RunTest.sh 5 $NClients $Repeat $x
    ./RunTest.sh 6 $NClients $Repeat $x
    ./RunTest.sh 7 $NClients $Repeat $x
    ./RunTest.sh 8 $NClients $Repeat $x
    # ./RunTest.sh 9 $NClients $Repeat $x
    # ./RunTest.sh 10 $NClients $Repeat $x
    # ./RunTest.sh 11 $NClients $Repeat $x
    # ./RunTest.sh 12 $NClients $Repeat $x
    x=$(( $x + 1 ))
done
