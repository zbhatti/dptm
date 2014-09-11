from ROOT import *
import libPyROOT
import numpy as np

#read optimal for execution_device(kernelMB)
#graph each kernel's performance at optimal levels dependent on MB on a canvas grouped with other devices
tree=TTree()
tree.ReadFile("optimal.csv","",',')

data = {} #key is kernel, value is list of functions
kernels=[]
devices=[]

"""
s=std.string()
tree.Branch('kernel',s)
for word in 
"""

#get all names of kernels and devices
for i in xrange(0,tree.GetEntries()):
	tree.GetEntry(i)
	kerName=tree.kernel.split('\x00')[0]
	devName=tree.device.split('\x00')[0]
	
	if not (kerName in kernels):
		kernels.append(kerName)
	
	if not (devName in devices):
		devices.append(devName)

#populate the data structure:
for ker in kernels:
	functions={} #key is device, value is two lists
	#access like functions[device][0 for mb, 1 for time][index]
	
	for dev in devices:
		if not functions.has_key(dev):
			functions[dev] = [[],[]]
		
		
		
		for i in xrange(0, tree.GetEntries()):
			tree.GetEntry(i)
			if (tree.kernel.split('\x00')[0]==ker) and (tree.device.split('\x00')[0]==dev):
				functions[dev][0].append(tree.mb)
				functions[dev][1].append(tree.time)

	data[ker]=functions

	
#format floats into python arrays and sort them
for ker in kernels:
	for dev in devices:
	
		n = len(data[ker][dev][0])
		
		#copy lists out of data:
		mbList = data[ker][dev][0]
		timeList = data[ker][dev][1]
		
		#sort maintining ordering between lists using macro found on SE:
		#http://stackoverflow.com/questions/9764298/is-it-possible-to-sort-two-listswhich-reference-each-other-in-the-exact-same-w
		mbList, timeList = zip(*sorted(zip(mbList, timeList)))
		mbList, timeList = (list(t) for t in zip(*sorted(zip(mbList, timeList))))
		
		mbListConverted = np.zeros(n, dtype=float)
		timeListConverted = np.zeros(n, dtype=float)
		for i in xrange(0, n):
			mbListConverted[i] = mbList[i]
			timeListConverted[i] = timeList[i]

		data[ker][dev][0] = mbListConverted
		data[ker][dev][1] = timeListConverted
		
	
#graph the points from the data
#using tutorial for multigraph:
#http://root.cern.ch/root/html532/tutorials/graphs/exclusiongraph.C.html
for ker in kernels:
	c = TCanvas(ker)
	mg = TMultiGraph()
	mg.SetTitle(ker)
	
	color=1
	
	for dev in devices:
		n = len(data[ker][dev][0])
		devGraph = TGraph(n, data[ker][dev][0], data[ker][dev][1])
		devGraph.SetLineColor(color)
		mg.Add(devGraph)
		color = color + 1
	mg.Draw("AC")
	
	c.Print("./results/optimal/"+ker+".png")
	
#tree.Print()