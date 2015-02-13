from ROOT import *
import numpy as np
import os

#dictionary of canvases with filenames as key, must be persistant through entire execution 
Canvases={}

#data[filename]=[tree, canvas, TGRAPHS]
data={}

#get bounds for histogram that will calculate mean,
#get a list of unique threads that will be used as keys for the histogram, remove any failed kernels
def getThreadsAndBounds(tree):
	Bounds= {}
	threads=[]
	for i in xrange(0,tree.GetEntries()):
		tree.GetEntry(i)
		if(tree.execute < 0): # the current test at xLocal is a failed test
			continue
	
		wgDim = (tree.xLocal, tree.yLocal, tree.zLocal)
		
		
		if not (wgDim in threads): #add the current wgDim to our list of threads
			threads.append(wgDim)
			Bounds[wgDim]=[100000,0]
		
		if tree.execute>Bounds[wgDim][1]:
			Bounds[wgDim][1]=tree.execute
		if tree.execute<Bounds[wgDim][0]:
			Bounds[wgDim][0]=tree.execute
	return [Bounds, threads]


#gets the mean time seen at each (xLocal,yLocal,zLocal) and finds the optimal point to launch at
def getMeanAndMinimumTimes(threads, Bounds, tree, kerName, devName):
		min = 100000000000
		minP = (0,0,0)
		xThreads = np.zeros(len(threads), dtype=float)
		yThreads = np.zeros(len(threads), dtype=float)
		zThreads = np.zeros(len(threads), dtype=float)
		t = np.zeros(len(threads), dtype=float)
		
		i=0
		for thr in threads: #where thr is a 3-tuple
			histname = kerName + str(tree.MB) + devName + str(thr[0]) +","+ str(thr[1]) +","+ str(thr[2])
			h=TH1F(histname, histname, len(threads), Bounds[thr][0]*.9, Bounds[thr][1]*1.1)
			tree.Draw("execute>>"+histname, "xLocal==" + str(thr[0]) + "&&" + "yLocal==" + str(thr[1]) + "&&" + "zLocal==" + str(thr[2]))
			
			xThreads[i]= thr[0]
			yThreads[i]= thr[1]
			zThreads[i]= thr[2]
			t[i]= h.GetMean()
			if h.GetMean() < min:
				min = h.GetMean()
				minP = thr
			i=i+1 
		return [xThreads, yThreads, zThreads, t, minP, min]

def plot_ONE_D(fileName,tree, optimalFile, devName,kerName, keepData = false):
	Canvases[fileName]=TCanvas(fileName)
	Canvases[fileName].cd()
	
	#get maximum and minimum times found for (xLocal,1,1)
	ret = getThreadsAndBounds(tree)
	Bounds = ret[0] #dictionary with tuples (X,1,1) as keys and [min,max] as values
	threads = ret[1] #array of tuples (xLocal,1,1)
	if (len(threads) ==0):
		return
	
	MeanAndMin = getMeanAndMinimumTimes(threads, Bounds, tree, kerName, devName)
	xThreads = MeanAndMin[0]
	yThreads = MeanAndMin[1]
	zThreads = MeanAndMin[2]
	t = MeanAndMin[3]
	minP = MeanAndMin[4]
	min = MeanAndMin[5]
		
	minString= "min at: " + str(minP)+" in: " + str("%.2f"%min) + " ms"
	minLabel = TPaveLabel(.65,.83,.90,.9,minString,"NDC")
	tree.GetEntry(0)
	
	optimalFile.write(kerName+","+devName+","+str(minP)[1:-1]+","+str(tree.MB)+","+str(min)+"\n")
	plot = TGraph(len(threads), xThreads, t)
	
	title=""+tree.kernel[:-1]+devName+str(tree.MB)
	plot.SetTitle(title)
	plot.SetMarkerStyle(20)
	plot.Draw("AL")
	minLabel.Draw()
	plot.GetXaxis().SetTitle("xThreads")
	plot.GetYaxis().SetTitle("time (ms)")
	
	Canvases[fileName].SetLogx(1)
	Canvases[fileName].Update()
	
	picFile = "./kernelMB/" + devName + "/" + kerName + str(tree.MB)
	Canvases[fileName].Print(picFile+".png")
	Canvases[fileName].Close()
	
	#used to keep data for use in main
	if keepData:
		data[fileName]=(tree, Canvases[fileName], plot) 
	else:
		data[fileName]=(-1, Canvases[fileName], -1)
	
	
def plot_TWO_D(fileName,tree, optimalFile,devName,kerName, keepData = false):
	Canvases[fileName]=TCanvas(fileName)
	Canvases[fileName].cd()
	
	#get maximum and minimum times found for (xLocal,yLocal,1)
	ret = getThreadsAndBounds(tree)
	Bounds = ret[0] #dictionary with tuples (X,Y,1) as keys and [min,max] as values
	threads = ret[1] #array of tuples (xLocal,yLocal,1)
	if (len(threads) ==0):
		return
	
	MeanAndMin = getMeanAndMinimumTimes(threads, Bounds, tree, kerName, devName)
	xThreads = MeanAndMin[0]
	yThreads = MeanAndMin[1]
	zThreads = MeanAndMin[2]
	t = MeanAndMin[3]
	minP = MeanAndMin[4]
	min = MeanAndMin[5]

	minString= "min at: " + str(minP)+" in: " + str("%.2f"%min) + " ms"
	minLabel = TPaveLabel(.65,.83,.90,.9,minString,"NDC")
	
	tree.GetEntry(0)
	graphName = kerName+", "+devName+", "+str(tree.MB)
	optimalFile.write(kerName+","+devName+","+str(minP)[1:-1]+","+str(tree.MB)+","+str(min)+"\n")
	plot = TGraph2D(graphName,"empty", len(threads), xThreads, yThreads, t)
	title = ""+tree.kernel[:-1]+devName+str(tree.MB)
	plot.SetTitle(title)
	plot.SetMarkerStyle(20)
	plot.Draw("P0&&TRI2T&&colz")
	minLabel.Draw()
	
	Canvases[fileName].SetLogx(1)
	Canvases[fileName].SetLogy(1)
	Canvases[fileName].SetLogz(1)
	Canvases[fileName].SetPhi(0)
	Canvases[fileName].SetTheta(90)
	Canvases[fileName].Update()
	
	picFile = "./kernelMB/" + devName + "/" + kerName + str(tree.MB)
	
	#used to keep data for use in main
	if keepData:
		data[fileName]=(tree, Canvases[fileName], plot) 
	else:
		data[fileName]=(-1, Canvases[fileName], -1)
	
	plot.GetXaxis().SetTitle("xThreads")
	plot.GetYaxis().SetTitle("yThreads")
	plot.GetZaxis().SetTitle("time (ms)")
	
	Canvases[fileName].Update()
	Canvases[fileName].Print(picFile+".png")
	Canvases[fileName].Close()
	

def plot_THREE_D(fileName,tree, optimalFile,devName,kerName, keepData = false):

	Canvases[fileName]=TCanvas(fileName)
	Canvases[fileName].cd()
	
	#get maximum and minimum times found for (xLocal,yLocal,zLocal), and get list of wgSizes
	ret = getThreadsAndBounds(tree)
	Bounds = ret[0] #dictionary with tuple keys. (X,Y,Z) is key and [min,max] is value
	threads = ret[1] #array of tuples (xLocal,yLocal,zLocal)
	if (len(threads) ==0):
		return
	
	MeanAndMin = getMeanAndMinimumTimes(threads, Bounds, tree, kerName, devName)
	xThreads = MeanAndMin[0]
	yThreads = MeanAndMin[1]
	zThreads = MeanAndMin[2]
	t = MeanAndMin[3]
	minP = MeanAndMin[4]
	min = MeanAndMin[5]
		
	ntuple = TNtuple("ntuple",tree.kernel[:-1],"xThreads:yThreads:zThreads:t")
	
	i=0
	for thr in threads:
		ntuple.Fill(xThreads[i],yThreads[i],zThreads[i],t[i]) #t(x,y,z)
		i = i + 1

	
	minString= "min at: " + str(minP)+" in: " + str("%.2f"%min) + " ms"
	minLabel = TPaveLabel(.65,.83,.90,.9,minString,"NDC")
	
	tree.GetEntry(0)
	optimalFile.write(kerName+","+devName+","+str(minP)[1:-1]+","+str(tree.MB)+","+str(min)+"\n")
	
	ntuple.SetMarkerStyle(20)
	ntuple.Draw("xThreads:yThreads:zThreads:t","","L&&colz",len(threads),0)
	
	title = ""+tree.kernel[:-1]+devName+str(tree.MB)
	titleLabel = TPaveLabel(.20,.93,.80,1,title,"NDC") #need to draw these labels over the false ntuple title
	
	minLabel.Draw()
	
	Canvases[fileName].SetPhi(260)
	Canvases[fileName].SetTheta(20)
	Canvases[fileName].Update()
	
	picFile = "./kernelMB/" + devName + "/" + kerName + str(tree.MB)
	Canvases[fileName].Print(picFile+".png")
	Canvases[fileName].Close()
	
	#used to keep data for use in main
	if keepData:
		data[fileName]=(tree, Canvases[fileName], plot)
	else:
		data[fileName]=(-1, Canvases[fileName], -1)


################ MAIN ###################

FileNames=[]
#find files that end with .log and add them into FileNames array
for root, dirs, files in os.walk("../raw"):
	for file in files:
		if file.endswith(".log"):
			FileNames.append(os.path.join(root, file))	

#make ttree for each device/kernel.log
newFile=open("optimal.csv","w")
newFile.write("kernel/C,device/C,x/I,y/I,z/I,mb/F,time/F\n")
newFile.close() #used to clear the preexisting file

for f in FileNames:
	print "Working on " + f + "..."
	data[f]=()
	tree=TTree()
	tree.ReadFile(f,"",',')
	
	tree.GetEntry(0);
	optimalFile=open("optimal.csv","a")	
	
	slashIndex = f.rfind("/");
	devName = f[7:slashIndex]
	kerName = tree.kernel[:-1]
	
	if tree.workDimension[0:3]=="ONE":
		plot_ONE_D(f, tree, optimalFile, devName, str(kerName))
	if tree.workDimension[0:3]=="TWO":
		plot_TWO_D(f, tree, optimalFile, devName, str(kerName))
	if tree.workDimension[0:3]=="THR":
		plot_THREE_D(f, tree, optimalFile, devName, str(kerName))
	
	else:
		continue

