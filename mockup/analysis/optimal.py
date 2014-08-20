import csv

Read text into dictionary:

kernel,device,x,y,z,mb,time

data = {}


#each unique key stores a 4-tuple
data[kernel][mb][device] = (x,y,z,time)