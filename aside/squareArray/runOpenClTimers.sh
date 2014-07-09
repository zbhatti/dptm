sudo ls
# sudo ./TimingKernel 0 0 100000000 0 10000000 2
# sudo ./TimingKernel 1 0 100000000 0 10000000 2
# sudo ./TimingKernel 1 1 100000000 0 10000000 2
# sudo ./TimingKernel 2 0 100000000 0 10000000 2
# sudo ./TimingKernel 2 1 100000000 0 10000000 2

#sudo ./TimingKernel 0 0 100000000 1 500000 25 > logs/0-0.csv
sudo ./TimingKernel 1 0 100000000 1 500000 25 > logs/1-0.csv
sudo ./TimingKernel 1 1 100000000 1 500000 25 > logs/1-1.csv
sudo ./TimingKernel 2 0 100000000 1 500000 25 > logs/2-0.csv
sudo ./TimingKernel 2 1 100000000 1 500000 25 > logs/2-1.csv

