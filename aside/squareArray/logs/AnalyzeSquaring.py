from ROOT import *

def LoadATree(filename): 
		
		t=TTree()
		t.ReadFile(filename)
		
		return t
	

FileNames=["OPT-NVIDIASDK-K20X.csv",
					 "OPT-AMDSDK-W9100.csv",
					 "OPT-AMDSDK-E52695-edited.csv",
					 "OPT-IntelSDK-E52695.csv",
					 "OPT-IntelSDK-Phi.csv",
					 "AMDSDK-W9100.csv",
					 "AMDSDK-E52695.csv",
					 "IntelSDK-E52695.csv",
					 "IntelSDK-Phi.csv",
					 "CUDAK20X.csv"
					]

DeviceNames=["OPT-NVIDIASDK-K20X",
						 "OPT-AMDSDK-W9100",
						 "OPT-AMDSDK-E52695",
						 "OPT-IntelSDK-E52695",
						 "OPT-IntelSDK-Phi",
						 "AMDSDK-W9100",
						 "AMDSDK-E52695",
						 "IntelSDK-E52695",
						 "IntelSDK-Phi",
						 "CUDA-K20X"
						]

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

# Mean and RMS data
CUMean=0
CURMS=1
FFTMean=2
FFTRMS=3

def GetNVecs(Tree):
	NVecs=[] 
	Bounds={}

	for i in xrange(0,Tree.GetEntries()):
		Tree.GetEntry(i)
		N=Tree.nVectors
		#N=Tree.NVec
		if not (N in NVecs):
			NVecs.append(N)
			NVecs.sort()
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
				
	return [NVecs,Bounds]

def GetMeanAndRMS(Tree,BinFactor=1):
	[NVecs,Bounds]=GetNVecs(Tree)

	#Nvecs is an array of vectorsizes
	#Bounds is a list of vectorsizes where each vectorsize entry has min/max of all seen vectors of that size's plan,execu,memcpy times		
	Results={}
	Hists={}
	Bins=len(NVecs)*BinFactor
	
	for NVec in NVecs:
		if not NVec in Results:
			Results[NVec]=[-1,-1,-1,-1,-1,-1,-1,-1]
			Hists[NVec] = [-1,-1,-1,-1]
			#Results[NVec]=[-1,-1,-1,-1]
			#Hists[NVec]=[-1,-1,-1,-1] ? why was this 4 numbers instead of 2

		histname="memcpyOut"+str(NVec)
		hmemcpyOut=TH1F(histname,histname,Bins,Bounds[NVec][minmemcpyOut]*.9,Bounds[NVec][maxmemcpyOut]*1.1)
		Tree.Draw("memcpyOut>>"+histname,"nVectors=="+str(NVec))
		Results[NVec][0]=hmemcpyOut.GetMean()
		Results[NVec][1]=hmemcpyOut.GetRMS()

		histname="plan"+str(NVec)
		hplan=TH1F(histname,histname,Bins,Bounds[NVec][minplan]*.9,Bounds[NVec][maxplan]*1.1)
		Tree.Draw("plan>>"+histname,"nVectors=="+str(NVec))
		Results[NVec][2]=hplan.GetMean()
		Results[NVec][3]=hplan.GetRMS()

		histname="execu"+str(NVec)
		hexecu=TH1F(histname,histname,Bins,Bounds[NVec][minexecu]*.9,Bounds[NVec][maxexecu]*1.1)
		Tree.Draw("execu>>"+histname,"nVectors=="+str(NVec))
		Results[NVec][4]=hexecu.GetMean()
		Results[NVec][5]=hexecu.GetRMS()

		histname="memcpyIn"+str(NVec)
		hmemcpyIn=TH1F(histname,histname,Bins,Bounds[NVec][minmemcpyIn]*.9,Bounds[NVec][maxmemcpyIn]*1.1)
		Tree.Draw("memcpyIn>>"+histname,"nVectors=="+str(NVec))
		Results[NVec][6]=hmemcpyIn.GetMean()
		Results[NVec][7]=hmemcpyIn.GetRMS()

		Hists[NVec][0]=hmemcpyOut
		Hists[NVec][1]=hplan
		Hists[NVec][2]=hexecu
		Hists[NVec][3]=hmemcpyIn
				
	return [Results,Hists,Bounds]

def PlotTime(name,R,MeanI=0,RMSI=1,BinFactor=1):
	NVecs=R.keys()
	c1=TCanvas("plottime")
	c1.cd()

	Bins=len(NVecs)*BinFactor
	h=TH1F(name,name,Bins,min(NVecs)*4/1048576,max(NVecs)*4/1048576)

	for NVec in NVecs:
		bin=h.FindBin(NVec*4/1048576)
		h.SetBinContent(bin,R[NVec][MeanI])
		h.SetBinError(bin,R[NVec][RMSI])

	return h

def log2(x):
		return log(x)/log(2)
		
		#plot flops on the assumption the function scales linearly or is constant with data size
def PlotFlopsLin(name,R,MeanI=0,RMSI=1,BinFactor=1):
		NVecs=R.keys()
		NVecs.sort()

		Bins=len(NVecs)*BinFactor
		h=TH1F(name,name,Bins,min(NVecs)*4/1048576,max(NVecs)*4/1048576)

		for NVec in NVecs:
				bin=h.FindBin(NVec*4/1048576)
				if R[NVec][MeanI]<=0:
						gflops=.000000001
				else:
						gflops=((4*float(NVec)/1073741824)/R[NVec][MeanI])*1000
				h.SetBinContent(bin,gflops)

		return h

def PrettyHists(h,color,marker):
		h.SetStats(False)
		h.SetLineColor(color)
		h.SetMarkerColor(color)
		h.SetMarkerStyle(marker)
		#was h.SetMinimum(0) but this caused making log graphs a problem
		h.SetMinimum(0)

objs=[]

#[hist1, jist] ["GPU1", "GPU2"]

def DrawPlots(Perf_GPU,Names,opts=None,Title=None,xTitle=None,yTitle=None):

		if opts:
				opts=","+opts

		leg=TLegend(0.8,0.7,0.9,0.9)

		hists=[]

		colors=[1,2,3,4,5,2,3,4,5,1,40,46]
		markers=[2,4,8,25,26,28,33,30,21,29]

		for i in range(len(Perf_GPU)):
				PrettyHists(Perf_GPU[i],colors[i],markers[i])
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

		GFLOPS_MemcpyOut.append(PlotFlopsLin(DeviceNames[i],Results[i][0],0,1))
		GFLOPS_Plan.append(PlotFlopsLin(DeviceNames[i],Results[i][0],2,3))
		GFLOPS_Execu.append(PlotFlopsLin(DeviceNames[i],Results[i][0],4,5))
		GFLOPS_MemcpyIn.append(PlotFlopsLin(DeviceNames[i],Results[i][0],6,7))

TimeCanvasmemcpyOut=TCanvas("Time Comparison memcpyOut")
TimeCanvasmemcpyOut.cd()
#DrawPlots(Time_MemcpyOut,DeviceNames,"P","Memory Copy to Device","Megabytes","m sec")
#TimeCanvasmemcpyOut.Print("TimeComparison_memcpyOut.pdf")

TimeCanvasplan=TCanvas("Time Comparison plan")
TimeCanvasplan.cd()
#DrawPlots(Time_Plan,DeviceNames,"P","Plan Making","Megabytes","m sec")
#TimeCanvasplan.Print("TimeComparison_plan.pdf")

TimeCanvasexecu=TCanvas("Time Comparison execu")
TimeCanvasexecu.cd()
DrawPlots(Time_Execu,DeviceNames,"P","Execution of Squaring","Megabytes","m sec")
#TimeCanvasexecu.Print("TimeComparison_execu.pdf")

TimeCanvasmemcpyIn=TCanvas("Time Comparison memcpyIn")
TimeCanvasmemcpyIn.cd()
#DrawPlots(Time_MemcpyIn,DeviceNames,"P","Memory Copy to Host","Megabytes","m sec")
#TimeCanvasmemcpyIn.Print("TimeComparison_memcpyIn.pdf")


ComputeCanvasmemcpyOut=TCanvas("Compute Comparison memcpyOut")
ComputeCanvasmemcpyOut.cd()
#DrawPlots(GFLOPS_MemcpyOut,DeviceNames,"P","Memory Copy to Device","Megabytes","GB/s")
#ComputeCanvasmemcpyOut.Print("BandwidthComparison_memcpyOut.pdf")

ComputeCanvasplan=TCanvas("Compute Comparison plan")
ComputeCanvasplan.cd()
#DrawPlots(GFLOPS_Plan,DeviceNames,"P","Plan Making","Megabytes","GFLOPS")
#ComputeCanvasplan.Print("GFLOPSComparison_plan.pdf")

ComputeCanvasexecu=TCanvas("Compute Comparison execu")
ComputeCanvasexecu.cd()
DrawPlots(GFLOPS_Execu,DeviceNames,"P","Execution of Squaring","Megabytes","GFLOPS")
#ComputeCanvasexecu.Print("GFLOPSComparison_execu.pdf")

ComputeCanvasmemcpyIn=TCanvas("Compute Comparison memcpyIn")
ComputeCanvasmemcpyIn.cd()
#DrawPlots(GFLOPS_MemcpyIn,DeviceNames,"P","Memory Copy to Host","Megabytes","GB/s")
#ComputeCanvasmemcpyIn.Print("BandwidthComparison_memcpyIn.pdf")
