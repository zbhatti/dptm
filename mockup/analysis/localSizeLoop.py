from ROOT import *
import numpy as np

"""
FileNames=[	"MatrixMult2048x2048.csv", 
						"MatrixMult512x512.csv", 
						"rotation8192x8192.csv" ]
						
KernelNames=[	"MatrixMult2048x2048", 
							"MatrixMult512x512", 
							"rotation8192x8192" ]


def LoadATree(filename):
	t = TTree()
	t.ReadFile(filename)
	return t

def GetThreads(Tree):
	Threads=[][]
	Bounds={}
	
	for i in xrange(0,Tree.GetEntries()):
		Tree.GetEntry(i)
		X=Tree.xThreads
		Y=Tree.yThreads
		
		if not (X in NVecs
	return [NVecs,Bounds]
	
#list of trees
kernels=[]

for i in FileNames:
	Kernels.append(LoadATree(i))
	print i


#build histograms
def MakeHistograms(Tree):

	return kernel[][][]

#get mean and rms
def 

#draw contour plots
	
#histograms[kernel][xThreads][yThreads]
#histograms{kernel}
histograms = [][][]

t=ROOT.TTree()

th1f has "number of loops" entries 
"""

#build ROOT tree
FileNames=[	"MatrixMult512x512.csv"
						#"MatrixMult2048x2048.csv"
						#"rotation8192x8192.csv" 
					]

Canvases=[]
for i in xrange(0,len(FileNames)):
	Canvases.append( TCanvas( FileNames[i]) )
	
c=0
for f in FileNames:
	
	tree=TTree()
	tree.ReadFile(f)
	print "Working on " + f
	threads=[] #array if tuples (X,Y)
	Bounds={} #dictionary with tuples (X,Y) as keys and [min,max] as values
	Hists={}
	MeanRMS={}
	
	Canvases[c].cd()
	
	x = np.zeros(100, dtype=float)
	y = np.zeros(100, dtype=float)
	z = np.zeros(100, dtype=float)

	#get maximum and minimum times found for (xThreads,yThreads)
	for i in xrange(0,tree.GetEntries()):
		tree.GetEntry(i)
		X=tree.xThreads
		Y=tree.yThreads
		
		if not ((X,Y) in threads):
			threads.append((X,Y))
			Bounds[(X,Y)]=[1000,0]
		
		if tree.time>Bounds[(X,Y)][1]:
			Bounds[(X,Y)][1]=tree.time
		if tree.time<Bounds[(X,Y)][0]: 
			Bounds[(X,Y)][0]=tree.time

	#build histograms, one for each x,y
	i=0
	for t in threads:
		if not t in MeanRMS:
			MeanRMS[(t)]=[-1,-1]
		if not t in Hists:
			Hists[(t)]= -1
		
		
		histname="time"+str(t[0])+str(t[1])
		h=TH1F(histname, histname, len(threads), Bounds[(t)][0]*.9, Bounds[(t)][1]*1.1)
		tree.Draw("time>>"+histname, "xThreads==" + str(t[0]) + "&&" + "yThreads==" + str(t[1]) )
		MeanRMS[(t)][0]=h.GetMean()
		MeanRMS[(t)][1]=h.GetRMS()
		
		Hists[(t)]= h
		
		#print str(t[0])+str(t[1])+str(h.GetMean())
		x[i]= t[0]
		y[i]= t[1]
		z[i]= h.GetMean()
		i=i+1
		
	l = TGraph2D("filling","Work Group Size Variations;xThreads;yThreads;time (ms)", len(threads), x, y, z)
	l.SetMarkerStyle(20)
	l.Draw("P&&TRI1")
	Canvases[c].SetLogx(1)
	Canvases[c].SetLogy(1)
	Canvases[c].SetLogz(1)
	c = c + 1
	
#plot contour
#Mean values to plot for contour: MeanRMS[(t)][0]
#RMS values associated with each Mean: MeanRMS[(t)][1]
