import csv
import operator
#source for sorting and subsorting:
#http://stackoverflow.com/questions/14416652/how-to-do-sub-sorting-in-python

#File Structure: kernel,device,x,y,z,mb,time
csvFile = open("optimal.csv", "rb")
optimalStrings = csv.DictReader(csvFile)

optimalList=[] #converted from strings to integers and floats

for row in optimalStrings:
	#fix formatting of original input to strings and intengers without ROOT specifiers
	tempDict = {"kernel": row['kernel/C'], 
							"mb": int(float(row['mb/F'])),
							"time": float(row['time/F']),
							"device": row['device/C'],
							"x": int(row['x/I']),
							"y": int(row['y/I']),
							"z": int(row['z/I'])
							}
	optimalList.append(tempDict)

optimalList = sorted(optimalList, key=operator.itemgetter('kernel', 'mb', 'time'))

"""
for row in optimalList:
	print(row["kernel"] +"\t"+ str(row["mb"]) +"\t"+ str(row['time']) +"\t"+ row['device']  )
"""

#printing to a file:
outFile = open("optimalFormatted.txt","w")

kernelMatch = optimalList[0]['kernel']
mbMatch = optimalList[0]['mb']
outFile.write("\n########## "+kernelMatch +" ##########\n")
outFile.write("\n----- "+str(mbMatch)+"MiB -----"+"\n")

for entry in optimalList:
	tempTuple = (entry['time'],entry['x'],entry['y'],entry['z'])
	
	if entry['kernel'] == kernelMatch:
		if entry['mb'] == mbMatch:
			outFile.write(entry['device']+",")
			#outFile.write("%.2f"%tempTuple[0]+",%d,%d,%d\n"%(tempTuple[1],tempTuple[2],tempTuple[3]))
			outFile.write(" %.2fms"%tempTuple[0]+"\n")
			
		else:
			mbMatch = entry['mb']
			outFile.write("\n----- "+str(mbMatch)+"MiB -----"+"\n")

	else:
		kernelMatch = entry['kernel']
		outFile.write("\n########## "+kernelMatch +" ##########\n")

outFile.close()
