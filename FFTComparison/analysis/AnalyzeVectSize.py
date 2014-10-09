from ROOT import *

FileNames=["AFSuper/Hawaii.csv",
					 "AFSuper/FFTW3-1.csv",
					 #"AFSuper/FFTW3-12.csv",
					 #"AFSuper/FFTW3-1-Scaled12.csv",
					 #"AFSuper/CudaK20X.csv",
					 #"AFSuper/OpenCLK20X.csv",
					 #"AFSuper/IntelPhi.csv",
					 #"AFSuper/IntelCPU.csv",
					 "AFSuper/AMDCPU.csv"
					]

DeviceNames=["clFFT-W9100",
						 "FFTW3-1-E52695",
						 #"FFTW3-12-E52695",
						 #"FFTW3-1-12-Expectation",
						 #"CUFFT-K20X",
						 #"clFFT-K20X",
						 #"clFFT-Phi",
						 #"clFFT-IntelSDK-E52695",
						 "clFFT-AMDSDK-E52695"
						]

def LoadATree(filename): 
		
		t=TTree()
		t.ReadFile(filename)
		
		return t


"""
input: a devices's tree
output: Nvecs array and Bounds dictionary whose corresponding indices tells us the 
"""
def GetNVecs(Tree):
	NVecs=[] #lists the nVectors found in a tree
	Bounds={} #dictionary of bounds for each NVectors, a key 'N' has an array of bounds at 'Bounds[N]'
	
	# for i from 0 to number of data points (lines)
	for i in xrange(0,Tree.GetEntries()):
		
		Tree.GetEntry(i)
		N=Tree.nVectors
		
		#ignore odd numbers:
		if N%2:
			continue
		
		#check if the current entry's nVectors is in our NVecs array, if not, initialize it with default bounds
		if not (N in NVecs): 
			NVecs.append(N)
			NVecs.sort() #sort nVectors values in the array
			
			Bounds[N] = [10000,0, 10000,0, 10000,0, 10000,0]

		if Tree.memcpyOut>Bounds[N][maxMemCpyOut]:
			Bounds[N][maxMemCpyOut]=Tree.memcpyOut
		if Tree.plan>Bounds[N][maxPlan]:
			Bounds[N][maxPlan]=Tree.plan
		if Tree.execu>Bounds[N][maxExecu]:
			Bounds[N][maxExecu]=Tree.execu
		if Tree.memcpyIn>Bounds[N][maxMemCpyIn]:
			Bounds[N][maxMemCpyIn]=Tree.memcpyIn

		if Tree.memcpyOut<Bounds[N][minMemCpyOut]:
			Bounds[N][minMemCpyOut]=Tree.memcpyOut
		if Tree.plan<Bounds[N][minPlan]:
			Bounds[N][minPlan]=Tree.plan
		if Tree.execu<Bounds[N][minExecu]:
			Bounds[N][minExecu]=Tree.execu
		if Tree.memcpyIn<Bounds[N][minMemCpyIn]:
			Bounds[N][minMemCpyIn]=Tree.memcpyIn
				
	return [NVecs,Bounds]	

def GetMeanAndRMS(Tree,BinFactor=1):
	[NVecs,Bounds]=GetNVecs(Tree)
	#Tree is from device.csv and then device[i]
	#Nvecs is an array of vector sizes
	MeanRMS={} #MeanRMS[N](MeanMemCpyOut,RMSMemCpyOut, MeanPlan, RMSPlan, ... RMSMemCpyIn)
	Hists={} #Hists[N](HistMemCpyOut, HistPlan, HistExecu, HistMemCpyIn)
	Bins=len(NVecs)*BinFactor
	
	for N in NVecs: #make a histogram for each NVector
		if not N in MeanRMS:
			MeanRMS[N]=[-1, -1, -1, -1, -1, -1, -1, -1]
			Hists[N] = [-1, -1, -1, -1]

		histname="memcpyOut"+str(N)
		hmemcpyOut=TH1F(histname,histname,Bins,Bounds[N][minMemCpyOut]*.9,Bounds[N][maxMemCpyOut]*1.1)
		Tree.Draw("memcpyOut>>"+histname,"nVectors=="+str(N))
		MeanRMS[N][0]=hmemcpyOut.GetMean()
		MeanRMS[N][1]=hmemcpyOut.GetRMS()

		histname="plan"+str(N)
		hplan=TH1F(histname,histname,Bins,Bounds[N][minPlan]*.9,Bounds[N][maxPlan]*1.1)
		Tree.Draw("plan>>"+histname,"nVectors=="+str(N))
		MeanRMS[N][2]=hplan.GetMean()
		MeanRMS[N][3]=hplan.GetRMS()

		histname="execu"+str(N)
		hexecu=TH1F(histname,histname,Bins,Bounds[N][minExecu]*.9,Bounds[N][maxExecu]*1.1)
		Tree.Draw("execu>>"+histname,"nVectors=="+str(N))
		MeanRMS[N][4]=hexecu.GetMean()
		MeanRMS[N][5]=hexecu.GetRMS()

		histname="memcpyIn"+str(N)
		hmemcpyIn=TH1F(histname,histname,Bins,Bounds[N][minMemCpyIn]*.9,Bounds[N][maxMemCpyIn]*1.1)
		Tree.Draw("memcpyIn>>"+histname,"nVectors=="+str(N))
		MeanRMS[N][6]=hmemcpyIn.GetMean()
		MeanRMS[N][7]=hmemcpyIn.GetRMS()

		Hists[N][0]=hmemcpyOut
		Hists[N][1]=hplan
		Hists[N][2]=hexecu
		Hists[N][3]=hmemcpyIn
				
	return [MeanRMS,Hists,Bounds]

def PlotTime(name,R,MeanI,RMSI,BinFactor=1):
	NVecs=R.keys()
	c1=TCanvas("plottime")
	c1.cd()

	Bins=len(NVecs)*BinFactor
	h=TH1F(name,name,Bins,min(NVecs)*8/1048576,max(NVecs)*8/1048576)

	for NVec in NVecs:
		bin=h.FindBin(NVec*8/1048576)
		h.SetBinContent(bin,R[NVec][MeanI])
		h.SetBinError(bin,R[NVec][RMSI])

	return h

def log2(x):
		return log(x)/log(2)

#plot flops if the data follows O(nlogn)
def PlotFlopsFFT(name,R,MeanI,RMSI,BinFactor=1):
		NVecs=R.keys()
		NVecs.sort()

		Bins=len(NVecs)*BinFactor
		h=TH1F(name,name,Bins,min(NVecs)*8/1048576,max(NVecs)*8/1048576)

		for NVec in NVecs:
				bin=h.FindBin(NVec*8/1048576)
				#Based on http://www.fftw.org/speed/method.html
				#if R[NVec][MeanI]<=0:
				#		 gflops=.000000001
				#else:
				gflops=(2.5*float(NVec)*log2(float(NVec))/R[NVec][MeanI])/1000000
				h.SetBinContent(bin,gflops)
				if gflops>=25:
					print name, NVec

		return h
		
#plot flops if the data follows O(n)
def PlotFlopsLin(name,R,MeanI,RMSI,BinFactor=1):
		NVecs=R.keys()
		NVecs.sort()

		Bins=len(NVecs)*BinFactor
		h=TH1F(name,name,Bins,min(NVecs)*8/1048576,max(NVecs)*8/1048576)

		for NVec in NVecs:
				bin=h.FindBin(NVec*8/1048576)
				if R[NVec][MeanI]<=0:
						gflops=.000000001
				else:
						gflops=((8*float(NVec)/1073741824)/R[NVec][MeanI])*1000
				h.SetBinContent(bin,gflops)

		return h


def PrettyHists(h,color,marker):
		h.SetStats(False)
		h.SetLineColor(color)
		h.SetMarkerColor(color)
		h.SetMarkerStyle(marker)
		#was h.SetMinimum(0) but this caused making log graphs a problem
		h.SetMinimum(0)


def DrawPlots(Perf_GPU,Names,opts=None,Title=None,xTitle=None,yTitle=None):

		if opts:
				opts=","+opts

		leg=TLegend(0.8,0.7,0.9,0.9)

		hists=[]

		colors=[1,2,4,3,5,6,7,8,11,30,40,46]
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

#Bounds and Mean and RMS Dictionary Index
minMemCpyOut=0
minPlan=2
minExecu=4
minMemCpyIn=6
maxMemCpyOut=1
maxPlan=3
maxExecu=5
maxMemCpyIn=7

#list of trees
Devices=[]

#related to the legend
objs=[]

#list of Results,Hists,Bounds, Results[device]{RMSMean,Hists,Bounds}
#for example, get histogram of 256 vectors execution like: 
#Results[deviceIndex][1][256][2]
Results=[]

#lists of histograms
Time_MemcpyOut=[]
Time_Plan=[]
Time_Execu=[]
Time_MemcpyIn=[]
GFLOPS_MemcpyOut=[]
GFLOPS_Plan=[]
GFLOPS_Execu=[]
GFLOPS_MemcpyIn=[]


for f in FileNames:
		Devices.append(LoadATree(f))
		print f


for i in range(len(Devices)):
		Results.append(GetMeanAndRMS(Devices[i]))
		
#has 4 data points for each device in Devices
		
		Time_MemcpyOut.append(PlotTime(DeviceNames[i], Results[i][0], 0, 1))
		Time_Plan.append(PlotTime(DeviceNames[i],Results[i][0], 2, 3))
		Time_Execu.append(PlotTime(DeviceNames[i],Results[i][0], 4, 5))
		Time_MemcpyIn.append(PlotTime(DeviceNames[i],Results[i][0], 6, 7))

		GFLOPS_MemcpyOut.append(PlotFlopsLin(DeviceNames[i], Results[i][0], 0, 1))
		GFLOPS_Plan.append(PlotFlopsLin(DeviceNames[i], Results[i][0], 2, 3))
		GFLOPS_Execu.append(PlotFlopsFFT(DeviceNames[i], Results[i][0], 4, 5))
		GFLOPS_MemcpyIn.append(PlotFlopsLin(DeviceNames[i], Results[i][0], 6, 7))


TimeCanvasmemcpyOut=TCanvas("Time Comparison memcpyOut")
TimeCanvasmemcpyOut.cd()
DrawPlots(Time_MemcpyOut,DeviceNames,"P","Memory Copy to Device","Megabytes","m sec")
#TimeCanvasmemcpyOut.Print("TimeComparison_memcpyOut.pdf")

TimeCanvasplan=TCanvas("Time Comparison plan")
TimeCanvasplan.cd()
DrawPlots(Time_Plan,DeviceNames,"P","Plan Making","Megabytes","m sec")

TimeCanvasexecu=TCanvas("Time Comparison execu")
TimeCanvasexecu.cd()
DrawPlots(Time_Execu,DeviceNames,"P","Execution of FFT","Megabytes","m sec")

TimeCanvasmemcpyIn=TCanvas("Time Comparison memcpyIn")
TimeCanvasmemcpyIn.cd()
DrawPlots(Time_MemcpyIn,DeviceNames,"P","Memory Copy to Host","Megabytes","m sec")

ComputeCanvasmemcpyOut=TCanvas("Compute Comparison memcpyOut")
ComputeCanvasmemcpyOut.cd()
DrawPlots(GFLOPS_MemcpyOut,DeviceNames,"P","Memory Copy to Device","Megabytes","GB/s")

ComputeCanvasplan=TCanvas("Compute Comparison plan")
ComputeCanvasplan.cd()
DrawPlots(GFLOPS_Plan,DeviceNames,"P","Plan Making","Megabytes","GFLOPS")

ComputeCanvasexecu=TCanvas("Compute Comparison execu")
ComputeCanvasexecu.cd()
DrawPlots(GFLOPS_Execu,DeviceNames,"P","Execution of FFT","Megabytes","GFLOPS")

ComputeCanvasmemcpyIn=TCanvas("Compute Comparison memcpyIn")
ComputeCanvasmemcpyIn.cd()
DrawPlots(GFLOPS_MemcpyIn,DeviceNames,"P","Memory Copy to Host","Megabytes","GB/s")
