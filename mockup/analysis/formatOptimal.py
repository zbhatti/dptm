import csv

#File Structure: kernel,device,x,y,z,mb,time

data = {} #keys are kernel names


#source from docs.python.org/2/library/csv.html
with open ("optimal.csv", "rb") as csvfile:
	reader = csv.reader(csvfile)
	
	line = 0;
	#Read text into dictionary:
	for lines in reader:
		
		#each unique key stores a 4-tuple
		if (line==0):
			line=line+1
			continue
		
		kernel = lines[0] #string
		device = lines[1] #string
		x = int(lines[2]) #integer
		y = int(lines[3]) #integer
		z = int(lines[4]) #integer
		mb = float(lines[5]) #float
		time = float(lines[6]) #float
		
		if kernel not in data:
			data[kernel] = {}
		
		if mb not in data[kernel]:
			data[kernel][mb] = {}
		
		data[kernel][mb][device] = (time,x,y,z)

"""
#data dictionary is generated, now print to screen to specified format:		
for kernels in data:
	print kernels
	for mb in data[kernels]:
		print mb
		for devices in data[kernels][mb]:
			print devices,
			print data[kernels][mb][devices]
		print "\n"
	print "\n"
"""

#printing to a file:
outFile = open("optimalFormatted.txt","w")	
for kernels in data:
	outFile.write("########## "+kernels+" ##########\n")
	for mb in data[kernels]:
		outFile.write(str(mb)+"\n")
		for devices in data[kernels][mb]:
			outFile.write(devices+","),
			tmpTup = data[kernels][mb][devices]
			outFile.write("%.2f"%tmpTup[0]+",%d,%d,%d\n"%(tmpTup[1],tmpTup[2],tmpTup[3]))
		outFile.write("\n")
	outFile.write("\n")
outFile.close()