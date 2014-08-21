from ROOT import *
import numpy as np
import os



FileNames=[]
#find files that end with .log and add them into FileNames array
for root, dirs, files in os.walk("."):
	for file in files:
		if file.endswith(".log"):
			print os.path.join(root, file)
			FileNames.append(os.path.join(root, file))

#dictionary of canvases with filenames as key
Canvases={}

#dictionary of graphs with filenames as key //Graphs[filename] = TGRAPH2D(
Graphs={}

#dictionary of dictionaries of histograms with filenames and ordered pair as key //Hists[filename][(x,y,z,)] = 
#Hists={}

data={}#data[filename]=[tree, Bounds[threads[]], Hists[threads[]], MeanRMS[threads[]], canvas, TGRAPHS]


def plot_ONE_D(f,tree, optimalFile,devName):
	threads=[] #array of xLocal
	Bounds={} #dictionary with tuples (xLocal) as keys and [min,max] as values
	Hists={}
	MeanRMS={}
	min =1000000000
	minP = 1
	Canvases[f]=TCanvas(f)
	
	for i in xrange(0,tree.GetEntries()):
		tree.GetEntry(i)
		if(tree.execute < 0): # the current test at xLocal is a failed test
			continue
		X=tree.xLocal
		
		if not (X in threads):
			threads.append(X)
			Bounds[X]=[100000,0]
		
		if tree.execute>Bounds[X][1]:
			Bounds[X][1]=tree.execute
		if tree.execute<Bounds[X][0]:
			Bounds[X][0]=tree.execute
	
	Canvases[f].cd()
	x = np.zeros(len(threads), dtype=float) #xLocal
	t = np.zeros(len(threads), dtype=float) #execution time
	
	i=0
	for thr in threads: #where thr is an integer
		if not thr in MeanRMS:
			MeanRMS[thr]=[-1,-1]
		if not thr in Hists:
			Hists[thr]= -1
		
		histname="execute"+str(thr)
		h=TH1F(histname, histname, len(threads), Bounds[thr][0]*.9, Bounds[thr][1]*1.1)
		tree.Draw("execute>>"+histname, "xLocal==" + str(thr))
		MeanRMS[thr][0]=h.GetMean()
		MeanRMS[thr][1]=h.GetRMS()
		
		Hists[thr]= h
		
		#print str(thr[0])+str(thr[1])+str(h.GetMean())
		x[i]= thr
		t[i]= h.GetMean()
		if h.GetMean() < min:
			min = h.GetMean()
			minP = thr
		i=i+1
	
	minString= "min at: " + str(minP)+" in: " + str("%.2f"%min) + " ms"
	minLabel = TPaveLabel(.65,.83,.90,.9,minString,"NDC")
	tree.GetEntry(0)
	
	optimalFile.write(tree.kernel[:-1]+","+devName+","+str(minP)+",1,1,"+str(tree.MB)+","+str(tree.execute)+"\n")
	plot = TGraph(len(threads), x, t)
	title=""+tree.kernel[:-1]+devName+str(tree.MB)
	plot.SetTitle(title)
	
	plot.SetMarkerStyle(20)
	plot.Draw("AL")
	minLabel.Draw()
	plot.GetXaxis().SetTitle("xThreads")
	plot.GetYaxis().SetTitle("time (ms)")
	Canvases[f].SetLogx(1)
	
	Canvases[f].Update()
	pic = "./results/" + f[2:-4]
	Canvases[f].Print(pic+".png")
	Canvases[f].Close()
	data[f]=(tree, Bounds, Hists, MeanRMS, Canvases[f], plot)
	
	
def plot_TWO_D(f,tree, optimalFile,devName):
	threads=[] #array of tuples (xLocal,yLocal)
	Bounds={} #dictionary with tuples (X,Y) as keys and [min,max] as values
	Hists={}
	MeanRMS={}
	min =1000000000
	minP = (0,0)
	Canvases[f]=TCanvas(f)
	
	#get maximum and minimum times found for (xLocal,yLocal)
	for i in xrange(0,tree.GetEntries()):
		tree.GetEntry(i)
		if(tree.execute < 0 ): #the current x,y point is a failed test
			continue
		X=tree.xLocal
		Y=tree.yLocal
		
		if not ((X,Y) in threads):
			threads.append((X,Y))
			Bounds[(X,Y)]=[100000,0]
		
		if tree.execute>Bounds[(X,Y)][1]:
			Bounds[(X,Y)][1]=tree.execute
		if tree.execute<Bounds[(X,Y)][0]: 
			Bounds[(X,Y)][0]=tree.execute
	
	Canvases[f].cd()
	x = np.zeros(len(threads), dtype=float)
	y = np.zeros(len(threads), dtype=float)
	t = np.zeros(len(threads), dtype=float)
	#build histograms, one for each x,y
	
	i=0
	for thr in threads: #where thr is a tuple
		if not thr in MeanRMS:
			MeanRMS[thr]=[-1,-1]
		if not thr in Hists:
			Hists[thr]= -1
		
		histname="execute"+str(thr[0])+str(thr[1])
		h=TH1F(histname, histname, len(threads), Bounds[thr][0]*.9, Bounds[thr][1]*1.1)
		tree.Draw("execute>>"+histname, "xLocal==" + str(thr[0]) + "&&" + "yLocal==" + str(thr[1]) )
		MeanRMS[thr][0]=h.GetMean()
		MeanRMS[thr][1]=h.GetRMS()
		
		Hists[thr]= h
		
		x[i]= thr[0]
		y[i]= thr[1]
		t[i]= h.GetMean()
		if h.GetMean() < min:
			min = h.GetMean()
			minP = thr
		i=i+1

	
	minString= "min at: " + str(minP)+" in: " + str("%.2f"%min) + " ms"
	minLabel = TPaveLabel(.65,.83,.90,.9,minString,"NDC")
	#print xThreads,yThreads,zThreads,MB,\n
	tree.GetEntry(0)
	
	optimalFile.write(tree.kernel[:-1]+","+devName+","+str(minP[0])+","+str(minP[1])+","+"1"+","+str(tree.MB)+","+str(tree.execute)+"\n")
	plot = TGraph2D("empty","empty", len(threads), x, y, t)
	title = ""+tree.kernel[:-1]+devName+str(tree.MB)
	plot.SetTitle(title)
	#add color legend
	plot.SetMarkerStyle(20)
	
	plot.Draw("P0&&TRI2T&&colz")
	minLabel.Draw()
	
	Canvases[f].SetLogx(1)
	Canvases[f].SetLogy(1)
	Canvases[f].SetLogz(1)
	Canvases[f].SetPhi(0)
	Canvases[f].SetTheta(90)
	Canvases[f].Update()
	
	pic = "./results/" + f[2:-4]
	
	data[f]=(tree, Bounds, Hists, MeanRMS, Canvases[f], plot)
	data[f][5].GetXaxis().SetTitle("xThreads")
	data[f][5].GetYaxis().SetTitle("yThreads")
	data[f][5].GetZaxis().SetTitle("time (ms)")
	Canvases[f].Update()
	Canvases[f].Print(pic+".png")
	Canvases[f].Close()
	
	
	

def plot_THREE_D(f,tree, optimalFile,devName):
	threads=[] #array of tuples (xLocal,yLocal,zLocal)
	Bounds={} #dictionary with tuples (X,Y,Z) as keys and [min,max] as values
	Hists={}
	MeanRMS={}
	min =1000000000
	minP = (0,0,0)
	Canvases[f]=TCanvas(f)
	
	#get maximum and minimum times found for (xLocal,yLocal,zLocal)
	for i in xrange(0,tree.GetEntries()):
		tree.GetEntry(i)
		if(tree.execute < 0 ): #the current x,y,z point is a failed test
			continue
		X=tree.xLocal
		Y=tree.yLocal
		Z=tree.zLocal
		
		if not ((X,Y,Z) in threads):
			threads.append((X,Y,Z))
			Bounds[(X,Y,Z)]=[100000,0]
		
		if tree.execute>Bounds[(X,Y,Z)][1]:
			Bounds[(X,Y,Z)][1]=tree.execute
		if tree.execute<Bounds[(X,Y,Z)][0]: 
			Bounds[(X,Y,Z)][0]=tree.execute
	
	Canvases[f].cd()
	xThreads = np.zeros(len(threads), dtype=float)
	yThreads = np.zeros(len(threads), dtype=float)
	zThreads = np.zeros(len(threads), dtype=float)
	t = np.zeros(len(threads), dtype=float)
	#build histograms, one for each x,y,z
	
	ntuple = TNtuple("ntuple",tree.kernel[:-1],"xThreads:yThreads:zThreads:t")
	i=0
	for thr in threads: #where thr is a 3-tuple
		if not thr in MeanRMS:
			MeanRMS[thr]=[-1,-1]
		if not thr in Hists:
			Hists[thr]= -1
		
		histname="execute"+str(thr[0])+str(thr[1])+str(thr[2])
		h=TH1F(histname, histname, len(threads), Bounds[thr][0]*.9, Bounds[thr][1]*1.1)
		tree.Draw("execute>>"+histname, "xLocal==" + str(thr[0]) + "&&" + "yLocal==" + str(thr[1]) + "&&" + "zLocal==" + str(thr[2]))
		MeanRMS[thr][0]=h.GetMean()
		MeanRMS[thr][1]=h.GetRMS()
		
		Hists[thr]= h 
		
		xThreads[i]= thr[0]
		yThreads[i]= thr[1]
		zThreads[i]= thr[2]
		t[i]= h.GetMean()
		if h.GetMean() < min:
			min = h.GetMean()
			minP = thr
		ntuple.Fill(xThreads[i],yThreads[i],zThreads[i],t[i]) #t(x,y,z)
		
		i=i+1  
	
	minString= "min at: " + str(minP)+" in: " + str("%.2f"%min) + " ms"
	minLabel = TPaveLabel(.65,.83,.90,.9,minString,"NDC")
	
	tree.GetEntry(0)
	optimalFile.write(tree.kernel[:-1]+","+devName+","+str(minP[0])+","+str(minP[1])+","+str(minP[2])+","+str(tree.MB)+","+str(tree.execute)+"\n")
	
	ntuple.SetMarkerStyle(20)
	ntuple.Draw("xThreads:yThreads:zThreads:t","","L&&colz",len(threads),0)
	
	title = ""+tree.kernel[:-1]+devName+str(tree.MB)
	titleLabel = TPaveLabel(.20,.93,.80,1,title,"NDC") #need to draw these labels over the false ntuple title
	
	minLabel.Draw()
	
	Canvases[f].SetPhi(260)
	Canvases[f].SetTheta(20)
	Canvases[f].Update()
	
	pic = "./results/" + f[2:-4]
	Canvases[f].Print(pic+".png")
	Canvases[f].Close()
	
	data[f]=(tree, Bounds, Hists, MeanRMS, Canvases[f],ntuple)

#make ttree for each device/kernel.log

newFile=open("optimal.csv","w")
newFile.write("kernel,device,x,y,z,mb,time\n")
newFile.close() #used to clear the preexisting file


for f in FileNames:
	data[f]=()
	tree=TTree()
	tree.ReadFile(f,"",',')
	print "Working on " + f
	
	tree.GetEntry(0);
	optimalFile=open("optimal.csv","a")	
	
	slashIndex = f.rfind("/");
	devName = f[2:slashIndex]
	
	if tree.workDimension[0:3]=="ONE":
		plot_ONE_D(f,tree, optimalFile,devName)
	if tree.workDimension[0:3]=="TWO":
		plot_TWO_D(f,tree, optimalFile,devName)
	if tree.workDimension[0:3]=="THR":
		plot_THREE_D(f,tree, optimalFile,devName)
	else:
		continue
	
	#save minimum information
