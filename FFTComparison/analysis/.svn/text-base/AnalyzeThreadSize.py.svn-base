from ROOT import *

def LoadATree(filename): 
		
		t=TTree()
		t.ReadFile(filename)
		
		return t
	

FileNames=["AFSuperCPUThreaded10000Vectors.csv",
					 "AFSuperCPUThreaded100000Vectors.csv",
					 "AFSuperCPUThreaded1000000Vectors.csv",
					]

DeviceNames=["10,000 Vectors",
						 "100,000 Vectors",
						 "1,000,000 Vectors",					 
						]

NVECTORS = [10000,100000,1000000]
Devices=[]


for f in FileNames:
		Devices.append(LoadATree(f))
		print f

		
#Bounds Dictionary Index
minmemcpyOut=0
minplan=2
minexecu=4
minmemcpyIn=6
maxmemcpyOut=1
maxplan=3
maxexecu=5
maxmemcpyIn=7

def GetNThreads(Tree):
	NThreads=[] 
	Bounds={}

	for i in xrange(0,Tree.GetEntries()):
		Tree.GetEntry(i)
		N=Tree.nThreads
		#N=Tree.NThr
		if not (N in NThreads):
			NThreads.append(N)
			NThreads.sort()
			Bounds[N] = [10000,0, 10000,0, 10000,0, 10000,0]
			#Bounds[N]=[ 10000000, 0, 10000000, 0]

		if Tree.memcpyOut>Bounds[N][maxmemcpyOut]:
			Bounds[N][maxmemcpyOut]=Tree.memcpyOut
		if Tree.plan>Bounds[N][maxplan]:
			Bounds[N][maxplan]=Tree.plan
		if Tree.execu>Bounds[N][maxexecu]:
			Bounds[N][maxexecu]=Tree.execu
		if Tree.memcpyIn>Bounds[N][maxmemcpyIn]:
			Bounds[N][maxmemcpyIn]=Tree.memcpyIn

		if Tree.memcpyOut<Bounds[N][minmemcpyOut]:
			Bounds[N][minmemcpyOut]=Tree.memcpyOut
		if Tree.plan<Bounds[N][minplan]:
			Bounds[N][minplan]=Tree.plan
		if Tree.execu<Bounds[N][minexecu]:
			Bounds[N][minexecu]=Tree.execu
		if Tree.memcpyIn<Bounds[N][minmemcpyIn]:
			Bounds[N][minmemcpyIn]=Tree.memcpyIn
				
	return [NThreads,Bounds]

def GetMeanAndRMS(Tree,BinFactor=1):
	[NThreads,Bounds]=GetNThreads(Tree)

	#NThreads is an array of vectorsizes
	#Bounds is a list of vectorsizes where each vectorsize entry has min/max of all seen vectors of that size's plan,execu,memcpy times		
	Results={}
	Hists={}
	Bins=len(NThreads)*BinFactor
	
	for NThr in NThreads:
		if not NThr in Results:
			Results[NThr]=[-1,-1,-1,-1,-1,-1,-1,-1]
			Hists[NThr] = [-1,-1,-1,-1]

		histname="memcpyOut"+str(NThr)
		hmemcpyOut=TH1F(histname,histname,Bins,Bounds[NThr][minmemcpyOut]*.9,Bounds[NThr][maxmemcpyOut]*1.1)
		Tree.Draw("memcpyOut>>"+histname,"nThreads=="+str(NThr))
		Results[NThr][0]=hmemcpyOut.GetMean()
		Results[NThr][1]=hmemcpyOut.GetRMS()

		histname="plan"+str(NThr)
		hplan=TH1F(histname,histname,Bins,Bounds[NThr][minplan]*.9,Bounds[NThr][maxplan]*1.1)
		Tree.Draw("plan>>"+histname,"nThreads=="+str(NThr))
		Results[NThr][2]=hplan.GetMean()
		Results[NThr][3]=hplan.GetRMS()

		histname="execu"+str(NThr)
		hexecu=TH1F(histname,histname,Bins,Bounds[NThr][minexecu]*.9,Bounds[NThr][maxexecu]*1.1)
		Tree.Draw("execu>>"+histname,"nThreads=="+str(NThr))
		Results[NThr][4]=hexecu.GetMean()
		Results[NThr][5]=hexecu.GetRMS()

		histname="memcpyIn"+str(NThr)
		hmemcpyIn=TH1F(histname,histname,Bins,Bounds[NThr][minmemcpyIn]*.9,Bounds[NThr][maxmemcpyIn]*1.1)
		Tree.Draw("memcpyIn>>"+histname,"nThreads=="+str(NThr))
		Results[NThr][6]=hmemcpyIn.GetMean()
		Results[NThr][7]=hmemcpyIn.GetRMS()

		Hists[NThr][0]=hmemcpyOut
		Hists[NThr][1]=hplan
		Hists[NThr][2]=hexecu
		Hists[NThr][3]=hmemcpyIn
				
	return [Results,Hists,Bounds]

def PlotTime(name,R,MeanI=0,RMSI=1,BinFactor=1):
	NThreads=R.keys()
	c1=TCanvas("plottime")
	c1.cd()

	Bins=len(NThreads)*BinFactor
	h=TH1F(name,name,Bins,min(NThreads),max(NThreads))

	for NThr in NThreads:
		bin=h.FindBin(NThr)
		h.SetBinContent(bin,R[NThr][MeanI])
		h.SetBinError(bin,R[NThr][RMSI])

	return h

def log2(x):
		return log(x)/log(2)

def PlotFlopsFFT(name,R,NVECTORS,MeanI=0,RMSI=1,BinFactor=1):
		NThreads=R.keys()
		NThreads.sort()
#		print NThreads
#		 NThreads=NThreads[1:]

		Bins=len(NThreads)*BinFactor
		h=TH1F(name,name,Bins, min(NThreads)-.5 ,max(NThreads) + 0.5)

		for NThr in NThreads:
				bin=h.FindBin(NThr)
				#Based on http://www.fftw.org/speed/method.html
				#if R[NThr][MeanI]<=0:
				#		 gflops=.000000001
				#else:
				x=R[NThr][MeanI]
				gflops=(2.5*float(NVECTORS)*log2(float(NVECTORS))/x)/1000000
				gflopsdt=(2.5*float(NVECTORS)*log2(float(NVECTORS))/(x*x))/1000000
				
				h.SetBinContent(bin,gflops)
				h.SetBinError(bin,gflopsdt*R[NThr][RMSI])
		return h
		
		#plot flops on the assumption the function scales linearly or is constant with data size
def PlotFlopsLin(name,R,NVECTORS,MeanI=0,RMSI=1,BinFactor=1):
		NThreads=R.keys()
		NThreads.sort()

		Bins=len(NThreads)*BinFactor
		h=TH1F(name,name,Bins,min(NThreads),max(NThreads))

		for NThr in NThreads:
				bin=h.FindBin(NThr)
				if R[NThr][MeanI]<=0:
						gflops=.000000001
				else:
						gflops=((8*float(NVECTORS)/1073741824)/R[NThr][MeanI])*1000
				h.SetBinContent(bin,gflops)

		return h

def PrettyHists(h,color):
		h.SetStats(False)
		h.SetLineColor(color)
		h.SetMarkerColor(color)
		h.SetMarkerStyle(2)
		h.SetMinimum(0)

objs=[]

#[hist1, jist] ["GPU1", "GPU2"]

def DrawPlots(Perf_GPU,Names,opts=None,Title=None,xTitle=None,yTitle=None):

		if opts:
				opts=","+opts

		leg=TLegend(0.8,0.7,0.9,0.9)

		hists=[]

		colors=[1,2,4,3,5,6,7,8,9]

		for i in range(len(Perf_GPU)):
				PrettyHists(Perf_GPU[i],colors[i])
				leg.AddEntry(Perf_GPU[i],Names[i])
				hists.append( ( Perf_GPU[i].GetBinContent(Perf_GPU[i].GetMaximumBin()),Perf_GPU[i]) )


		first=True
		hists.sort()
		hists.reverse()
		print hists
		for hh in hists:
				h=hh[1]
				if first:
						h.Draw(opts)
						first=False
						
						if xTitle:
								h.SetXTitle(xTitle)

						if yTitle:
								h.SetYTitle(yTitle)
						
						if Title:
								h.SetTitle(Title)
								
				else:
						h.Draw("same"+opts)

		leg.Draw()

		objs.append(leg)


# Start Analysis

# 2000 Wire Sample
# Format For output of GetMeanAndRMS [Results,Hists,Bounds]

Results=[] 
Time_MemcpyOut=[]
Time_Plan=[]
Time_Execu=[]
Time_MemcpyIn=[]
GFLOPS_MemcpyOut=[]
GFLOPS_Plan=[]
GFLOPS_Execu=[]
GFLOPS_MemcpyIn=[]

for i in range(len(Devices)):
		Results.append(GetMeanAndRMS(Devices[i]))
		
#has 4 data points for each device in Devices
		
		Time_MemcpyOut.append(PlotTime(DeviceNames[i],Results[i][0],0,1))
		Time_Plan.append(PlotTime(DeviceNames[i],Results[i][0],2,3))
		Time_Execu.append(PlotTime(DeviceNames[i],Results[i][0],4,5))
		Time_MemcpyIn.append(PlotTime(DeviceNames[i],Results[i][0],6,7))

		GFLOPS_MemcpyOut.append(PlotFlopsLin(DeviceNames[i],Results[i][0],NVECTORS[i],0,1))
		GFLOPS_Plan.append(PlotFlopsLin(DeviceNames[i],Results[i][0],NVECTORS[i],2,3))
		GFLOPS_Execu.append(PlotFlopsFFT(DeviceNames[i],Results[i][0],NVECTORS[i],4,5))
		GFLOPS_MemcpyIn.append(PlotFlopsLin(DeviceNames[i],Results[i][0],NVECTORS[i],6,7))

"""
TimeCanvasmemcpyOut=TCanvas("Time Comparison memcpyOut")
TimeCanvasmemcpyOut.cd()
DrawPlots(Time_MemcpyOut,DeviceNames,"P","Memory Copy to Device","nThreads","m sec")
TimeCanvasmemcpyOut.Print("TimeComparison_memcpyOut.pdf")

TimeCanvasplan=TCanvas("Time Comparison plan")
TimeCanvasplan.cd()
DrawPlots(Time_Plan,DeviceNames,"P","Plan Making","nThreads","m sec")
TimeCanvasplan.Print("TimeComparison_plan.pdf")
"""

TimeCanvasexecu=TCanvas("Time Comparison execu")
TimeCanvasexecu.cd()
DrawPlots(Time_Execu,DeviceNames,"P","Execution of FFT","nThreads","m sec")
#TimeCanvasexecu.Print("TimeComparison_execu.pdf")

#PUT LINEAR AND LOG ON SAME CANVAS

"""
TimeCanvasmemcpyIn=TCanvas("Time Comparison memcpyIn")
TimeCanvasmemcpyIn.cd()
DrawPlots(Time_MemcpyIn,DeviceNames,"P","Memory Copy to Host","nThreads","m sec")
TimeCanvasmemcpyIn.Print("TimeComparison_memcpyIn.pdf")


ComputeCanvasmemcpyOut=TCanvas("Compute Comparison memcpyOut")
ComputeCanvasmemcpyOut.cd()
DrawPlots(GFLOPS_MemcpyOut,DeviceNames,"P","Memory Copy to Device","nThreads","GB/s")
ComputeCanvasmemcpyOut.Print("BandwidthComparison_memcpyOut.pdf")


ComputeCanvasplan=TCanvas("Compute Comparison plan")
ComputeCanvasplan.cd()
DrawPlots(GFLOPS_Plan,DeviceNames,"P","Plan Making","nThreads","GFLOPS")
ComputeCanvasplan.Print("GFLOPSComparison_plan.pdf")
"""
ComputeCanvasexecu=TCanvas("Compute Comparison execu")
ComputeCanvasexecu.cd()
DrawPlots(GFLOPS_Execu,DeviceNames,"P","Execution of FFT","nThreads","GFLOPS")
#ComputeCanvasexecu.Print("GFLOPSComparison_execu.pdf")

#PUT LINEAR AND LOG ON SAME CANVAS

"""
ComputeCanvasmemcpyIn=TCanvas("Compute Comparison memcpyIn")
ComputeCanvasmemcpyIn.cd()
DrawPlots(GFLOPS_MemcpyIn,DeviceNames,"P","Memory Copy to Host","nThreads","GB/s")
ComputeCanvasmemcpyIn.Print("BandwidthComparison_memcpyIn.pdf")
"""
