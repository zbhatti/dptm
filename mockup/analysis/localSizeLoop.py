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
					

#make canvases to store files in:
Canvases={}

data={}#data[filename]=[tree, Bounds[threads[]], Hists[threads[]], MeanRMS[threads[]], canvas, TGRAPHS]
#make ttree for each device/kernel.log
for f in FileNames:
	data[f]=()
	tree=TTree()
	tree.ReadFile(f,"",',')
	print "Working on " + f
	tree.GetEntry(0);
	if tree.workDimension[0:3]=="ONE": #some issue with string encoding
		continue
	threads=[] #array if tuples (X,Y)
	Bounds={} #dictionary with tuples (X,Y) as keys and [min,max] as values
	Hists={}
	MeanRMS={}
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
			Bounds[(X,Y)]=[1000,0]
		
		if tree.execute>Bounds[(X,Y)][1]:
			Bounds[(X,Y)][1]=tree.execute
		if tree.execute<Bounds[(X,Y)][0]: 
			Bounds[(X,Y)][0]=tree.execute
	
	Canvases[f].cd()
	x = np.zeros(len(threads), dtype=float)
	y = np.zeros(len(threads), dtype=float)
	z = np.zeros(len(threads), dtype=float)
	#build histograms, one for each x,y
	i=0
	for t in threads: #where t is a tuple
		if not t in MeanRMS:
			MeanRMS[t]=[-1,-1]
		if not t in Hists:
			Hists[t]= -1
		
		histname="execute"+str(t[0])+str(t[1])
		h=TH1F(histname, histname, len(threads), Bounds[t][0]*.9, Bounds[t][1]*1.1)
		tree.Draw("execute>>"+histname, "xLocal==" + str(t[0]) + "&&" + "yLocal==" + str(t[1]) )
		MeanRMS[t][0]=h.GetMean()
		MeanRMS[t][1]=h.GetRMS()
		
		Hists[t]= h
		
		#print str(t[0])+str(t[1])+str(h.GetMean())
		x[i]= t[0]
		y[i]= t[1]
		z[i]= h.GetMean()
		i=i+1
	
	tree.GetEntry(0)
	plot = TGraph2D("empty","empty", len(threads), x, y, z)
	#plot.SetXTitle("xLocal")
	#plot.SetYTitle("yLocal")
	#plot.SetZTitle("execute (ms)")
	plot.SetTitle(tree.kernel+tree.device)
	#add color legend
	plot.SetMarkerStyle(20)
	gStyle.SetPalette(1)
	plot.Draw("P&&TRI1&&colz")
	Canvases[f].SetLogx(1)
	Canvases[f].SetLogy(1)
	
	data[f]=(tree, Bounds, Hists, MeanRMS, Canvases[f], plot)
	
	#Canvases[c].SetLogz(1)
	#hold = raw_input("press enter for next file")


#plot contour
#Mean values to plot for contour: MeanRMS[t][0]
#RMS values associated with each Mean: MeanRMS[t][1]
