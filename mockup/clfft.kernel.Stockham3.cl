/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


__constant float2 twiddles[127] = {
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(0.9807852804032304305792422383092344f, -0.1950903220161282480837883213098394f),
(float2)(0.9238795325112867384831361050601117f, -0.3826834323650897817792326804919867f),
(float2)(0.8314696123025452356714026791451033f, -0.5555702330196021776487214083317667f),
(float2)(0.9238795325112867384831361050601117f, -0.3826834323650897817792326804919867f),
(float2)(0.7071067811865475727373109293694142f, -0.7071067811865474617150084668537602f),
(float2)(0.3826834323650898372903839117498137f, -0.9238795325112867384831361050601117f),
(float2)(0.8314696123025452356714026791451033f, -0.5555702330196021776487214083317667f),
(float2)(0.3826834323650898372903839117498137f, -0.9238795325112867384831361050601117f),
(float2)(-0.1950903220161281925726370900520124f, -0.9807852804032304305792422383092344f),
(float2)(0.7071067811865475727373109293694142f, -0.7071067811865474617150084668537602f),
(float2)(0.0000000000000000612303176911188629f, -1.0000000000000000000000000000000000f),
(float2)(-0.7071067811865474617150084668537602f, -0.7071067811865475727373109293694142f),
(float2)(0.5555702330196022886710238708474208f, -0.8314696123025452356714026791451033f),
(float2)(-0.3826834323650897262680814492341597f, -0.9238795325112867384831361050601117f),
(float2)(-0.9807852804032304305792422383092344f, -0.1950903220161286089062713244857150f),
(float2)(0.3826834323650898372903839117498137f, -0.9238795325112867384831361050601117f),
(float2)(-0.7071067811865474617150084668537602f, -0.7071067811865475727373109293694142f),
(float2)(-0.9238795325112868495054385675757658f, 0.3826834323650896707569302179763326f),
(float2)(0.1950903220161283313505151681965799f, -0.9807852804032304305792422383092344f),
(float2)(-0.9238795325112867384831361050601117f, -0.3826834323650898928015351430076407f),
(float2)(-0.5555702330196021776487214083317667f, 0.8314696123025452356714026791451033f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
(float2)(0.9987954562051724050064649418345653f, -0.0490676743274180149345653489945107f),
(float2)(0.9951847266721969287317506314138882f, -0.0980171403295606036287779261328978f),
(float2)(0.9891765099647810144389836750633549f, -0.1467304744553617479319029826001497f),
(float2)(0.9951847266721969287317506314138882f, -0.0980171403295606036287779261328978f),
(float2)(0.9807852804032304305792422383092344f, -0.1950903220161282480837883213098394f),
(float2)(0.9569403357322088243819280251045711f, -0.2902846772544623310530198523338186f),
(float2)(0.9891765099647810144389836750633549f, -0.1467304744553617479319029826001497f),
(float2)(0.9569403357322088243819280251045711f, -0.2902846772544623310530198523338186f),
(float2)(0.9039892931234433381959547659789678f, -0.4275550934302820849097770405933261f),
(float2)(0.9807852804032304305792422383092344f, -0.1950903220161282480837883213098394f),
(float2)(0.9238795325112867384831361050601117f, -0.3826834323650897817792326804919867f),
(float2)(0.8314696123025452356714026791451033f, -0.5555702330196021776487214083317667f),
(float2)(0.9700312531945439742386838588572573f, -0.2429801799032638709441300761682214f),
(float2)(0.8819212643483550495560052695509512f, -0.4713967368259976420397094898362411f),
(float2)(0.7409511253549591058842338497925084f, -0.6715589548470183300921121372084599f),
(float2)(0.9569403357322088243819280251045711f, -0.2902846772544623310530198523338186f),
(float2)(0.8314696123025452356714026791451033f, -0.5555702330196021776487214083317667f),
(float2)(0.6343932841636454877942696839454584f, -0.7730104533627369933768136434082408f),
(float2)(0.9415440651830208063088889502978418f, -0.3368898533922200511092626129538985f),
(float2)(0.7730104533627369933768136434082408f, -0.6343932841636454877942696839454584f),
(float2)(0.5141027441932216612840989000687841f, -0.8577286100002721180857179206213914f),
(float2)(0.9238795325112867384831361050601117f, -0.3826834323650897817792326804919867f),
(float2)(0.7071067811865475727373109293694142f, -0.7071067811865474617150084668537602f),
(float2)(0.3826834323650898372903839117498137f, -0.9238795325112867384831361050601117f),
(float2)(0.9039892931234433381959547659789678f, -0.4275550934302820849097770405933261f),
(float2)(0.6343932841636454877942696839454584f, -0.7730104533627369933768136434082408f),
(float2)(0.2429801799032639819664325386838755f, -0.9700312531945439742386838588572573f),
(float2)(0.8819212643483550495560052695509512f, -0.4713967368259976420397094898362411f),
(float2)(0.5555702330196022886710238708474208f, -0.8314696123025452356714026791451033f),
(float2)(0.0980171403295607701622316199063789f, -0.9951847266721968177094481688982341f),
(float2)(0.8577286100002721180857179206213914f, -0.5141027441932216612840989000687841f),
(float2)(0.4713967368259978085731631836097222f, -0.8819212643483549385337028070352972f),
(float2)(-0.0490676743274177859510665200559743f, -0.9987954562051724050064649418345653f),
(float2)(0.8314696123025452356714026791451033f, -0.5555702330196021776487214083317667f),
(float2)(0.3826834323650898372903839117498137f, -0.9238795325112867384831361050601117f),
(float2)(-0.1950903220161281925726370900520124f, -0.9807852804032304305792422383092344f),
(float2)(0.8032075314806449428672863177780528f, -0.5956993044924333569056784654094372f),
(float2)(0.2902846772544623310530198523338186f, -0.9569403357322089354042304876202252f),
(float2)(-0.3368898533922201621315650754695525f, -0.9415440651830206952865864877821878f),
(float2)(0.7730104533627369933768136434082408f, -0.6343932841636454877942696839454584f),
(float2)(0.1950903220161283313505151681965799f, -0.9807852804032304305792422383092344f),
(float2)(-0.4713967368259976975508607210940681f, -0.8819212643483550495560052695509512f),
(float2)(0.7409511253549591058842338497925084f, -0.6715589548470183300921121372084599f),
(float2)(0.0980171403295607701622316199063789f, -0.9951847266721968177094481688982341f),
(float2)(-0.5956993044924329128164686153468210f, -0.8032075314806451649118912428093608f),
(float2)(0.7071067811865475727373109293694142f, -0.7071067811865474617150084668537602f),
(float2)(0.0000000000000000612303176911188629f, -1.0000000000000000000000000000000000f),
(float2)(-0.7071067811865474617150084668537602f, -0.7071067811865475727373109293694142f),
(float2)(0.6715589548470183300921121372084599f, -0.7409511253549591058842338497925084f),
(float2)(-0.0980171403295606452621413495762681f, -0.9951847266721969287317506314138882f),
(float2)(-0.8032075314806450538895887802937068f, -0.5956993044924331348610735403781291f),
(float2)(0.6343932841636454877942696839454584f, -0.7730104533627369933768136434082408f),
(float2)(-0.1950903220161281925726370900520124f, -0.9807852804032304305792422383092344f),
(float2)(-0.8819212643483549385337028070352972f, -0.4713967368259978640843144148675492f),
(float2)(0.5956993044924334679279809279250912f, -0.8032075314806448318449838552623987f),
(float2)(-0.2902846772544621645195661585603375f, -0.9569403357322089354042304876202252f),
(float2)(-0.9415440651830206952865864877821878f, -0.3368898533922203286650187692430336f),
(float2)(0.5555702330196022886710238708474208f, -0.8314696123025452356714026791451033f),
(float2)(-0.3826834323650897262680814492341597f, -0.9238795325112867384831361050601117f),
(float2)(-0.9807852804032304305792422383092344f, -0.1950903220161286089062713244857150f),
(float2)(0.5141027441932216612840989000687841f, -0.8577286100002721180857179206213914f),
(float2)(-0.4713967368259976975508607210940681f, -0.8819212643483550495560052695509512f),
(float2)(-0.9987954562051724050064649418345653f, -0.0490676743274179663623080216439121f),
(float2)(0.4713967368259978085731631836097222f, -0.8819212643483549385337028070352972f),
(float2)(-0.5555702330196019556041164833004586f, -0.8314696123025454577160076041764114f),
(float2)(-0.9951847266721969287317506314138882f, 0.0980171403295601456617802682558249f),
(float2)(0.4275550934302821959320795031089801f, -0.9039892931234433381959547659789678f),
(float2)(-0.6343932841636453767719672214298043f, -0.7730104533627371043991161059238948f),
(float2)(-0.9700312531945439742386838588572573f, 0.2429801799032638154329788449103944f),
(float2)(0.3826834323650898372903839117498137f, -0.9238795325112867384831361050601117f),
(float2)(-0.7071067811865474617150084668537602f, -0.7071067811865475727373109293694142f),
(float2)(-0.9238795325112868495054385675757658f, 0.3826834323650896707569302179763326f),
(float2)(0.3368898533922200511092626129538985f, -0.9415440651830208063088889502978418f),
(float2)(-0.7730104533627369933768136434082408f, -0.6343932841636454877942696839454584f),
(float2)(-0.8577286100002721180857179206213914f, 0.5141027441932215502617964375531301f),
(float2)(0.2902846772544623310530198523338186f, -0.9569403357322089354042304876202252f),
(float2)(-0.8314696123025453466937051416607574f, -0.5555702330196021776487214083317667f),
(float2)(-0.7730104533627368823545111808925867f, 0.6343932841636455988165721464611124f),
(float2)(0.2429801799032639819664325386838755f, -0.9700312531945439742386838588572573f),
(float2)(-0.8819212643483549385337028070352972f, -0.4713967368259978640843144148675492f),
(float2)(-0.6715589548470186631590195247554220f, 0.7409511253549588838396289247612003f),
(float2)(0.1950903220161283313505151681965799f, -0.9807852804032304305792422383092344f),
(float2)(-0.9238795325112867384831361050601117f, -0.3826834323650898928015351430076407f),
(float2)(-0.5555702330196021776487214083317667f, 0.8314696123025452356714026791451033f),
(float2)(0.1467304744553617479319029826001497f, -0.9891765099647810144389836750633549f),
(float2)(-0.9569403357322088243819280251045711f, -0.2902846772544623865641710835916456f),
(float2)(-0.4275550934302824734878356593981152f, 0.9039892931234431161513498409476597f),
(float2)(0.0980171403295607701622316199063789f, -0.9951847266721968177094481688982341f),
(float2)(-0.9807852804032304305792422383092344f, -0.1950903220161286089062713244857150f),
(float2)(-0.2902846772544632747425907837168779f, 0.9569403357322086023373231000732630f),
(float2)(0.0490676743274181259568678115101648f, -0.9987954562051724050064649418345653f),
(float2)(-0.9951847266721968177094481688982341f, -0.0980171403295608256733828511642059f),
(float2)(-0.1467304744553623030434152951784199f, 0.9891765099647809034166812125477009f),
};


#define fvect2 float2

#define C8Q  0.70710678118654752440084436210485f
#define C5QA 0.30901699437494742410229341718282f
#define C5QB 0.95105651629515357211643933337938f
#define C5QC 0.50000000000000000000000000000000f
#define C5QD 0.58778525229247312916870595463907f
#define C5QE 0.80901699437494742410229341718282f
#define C3QA 0.50000000000000000000000000000000f
#define C3QB 0.86602540378443864676372317075294f

__attribute__((always_inline)) void 
FwdRad4B1(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	
	T = (*R1); (*R1) = (*R2); (*R2) = T;
	
}

__attribute__((always_inline)) void 
InvRad4B1(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)((*R3).y, -(*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	
	T = (*R1); (*R1) = (*R2); (*R2) = T;
	
}

__attribute__((always_inline)) void 
FwdRad8B1(float2 *R0, float2 *R4, float2 *R2, float2 *R6, float2 *R1, float2 *R5, float2 *R3, float2 *R7)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	(*R5) = (*R4) - (*R5);
	(*R4) = 2.0f * (*R4) - (*R5);
	(*R7) = (*R6) - (*R7);
	(*R6) = 2.0f * (*R6) - (*R7);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + (fvect2)(-(*R7).y, (*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) - C8Q * (fvect2)((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + (fvect2)(-(*R6).y, (*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) - C8Q * (fvect2)((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	
	T = (*R1); (*R1) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R6); (*R6) = T;
	
}

__attribute__((always_inline)) void 
InvRad8B1(float2 *R0, float2 *R4, float2 *R2, float2 *R6, float2 *R1, float2 *R5, float2 *R3, float2 *R7)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	(*R5) = (*R4) - (*R5);
	(*R4) = 2.0f * (*R4) - (*R5);
	(*R7) = (*R6) - (*R7);
	(*R6) = 2.0f * (*R6) - (*R7);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)((*R3).y, -(*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + (fvect2)((*R7).y, -(*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) + C8Q * (fvect2)((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + (fvect2)((*R6).y, -(*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) + C8Q * (fvect2)((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	
	T = (*R1); (*R1) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R6); (*R6) = T;
	
}

__attribute__((always_inline)) void
FwdPass0(uint rw, uint b, uint me, uint inOffset, uint outOffset, __global float2 *bufIn, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*128];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 )*128];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 )*128];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 )*128];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 )*128];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 )*128];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 96 )*128];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 112 )*128];
	}


	FwdRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 )*1] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 )*1] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 )*1] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 )*1] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 )*1] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 )*1] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 )*1] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 )*1] = (*R7).x;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 )*1] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 )*1] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 )*1] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 )*1] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 )*1] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 )*1] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 )*1] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 )*1] = (*R7).y;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
FwdPass1(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{



	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 )*1] = (*R0).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 )*1] = (*R1).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 )*1] = (*R2).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 )*1] = (*R3).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 )*1] = (*R4).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 )*1] = (*R5).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 )*1] = (*R6).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 )*1] = (*R7).x;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 )*1] = (*R0).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 )*1] = (*R1).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 )*1] = (*R2).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 )*1] = (*R3).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 )*1] = (*R4).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 )*1] = (*R5).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 )*1] = (*R6).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 )*1] = (*R7).y;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
FwdPass2(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __global float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{



	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 0];
		float TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 1];
		float TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 2];
		float TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 0];
		float TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 1];
		float TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 2];
		float TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	bufOut[outOffset + ( 2*me + 0 + 0 )*128] = (*R0);
	bufOut[outOffset + ( 2*me + 0 + 32 )*128] = (*R1);
	bufOut[outOffset + ( 2*me + 0 + 64 )*128] = (*R2);
	bufOut[outOffset + ( 2*me + 0 + 96 )*128] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 0 )*128] = (*R4);
	bufOut[outOffset + ( 2*me + 1 + 32 )*128] = (*R5);
	bufOut[outOffset + ( 2*me + 1 + 64 )*128] = (*R6);
	bufOut[outOffset + ( 2*me + 1 + 96 )*128] = (*R7);
	}

}

__attribute__((always_inline)) void
InvPass0(uint rw, uint b, uint me, uint inOffset, uint outOffset, __global float2 *bufIn, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*128];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 )*128];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 )*128];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 )*128];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 )*128];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 )*128];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 96 )*128];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 112 )*128];
	}


	InvRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 )*1] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 )*1] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 )*1] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 )*1] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 )*1] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 )*1] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 )*1] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 )*1] = (*R7).x;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 )*1] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 )*1] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 )*1] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 )*1] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 )*1] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 )*1] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 )*1] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 )*1] = (*R7).y;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
InvPass1(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{



	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		float TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		float TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		float TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		float TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		float TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		float TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 )*1] = (*R0).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 )*1] = (*R1).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 )*1] = (*R2).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 )*1] = (*R3).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 )*1] = (*R4).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 )*1] = (*R5).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 )*1] = (*R6).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 )*1] = (*R7).x;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 )*1] = (*R0).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 )*1] = (*R1).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 )*1] = (*R2).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 )*1] = (*R3).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 )*1] = (*R4).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 )*1] = (*R5).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 )*1] = (*R6).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 )*1] = (*R7).y;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
InvPass2(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __global float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{



	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 0];
		float TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 1];
		float TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 2];
		float TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 0];
		float TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 1];
		float TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 2];
		float TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	bufOut[outOffset + ( 2*me + 0 + 0 )*128] = (*R0) * 6.1035156250000000e-05f;
	bufOut[outOffset + ( 2*me + 0 + 32 )*128] = (*R1) * 6.1035156250000000e-05f;
	bufOut[outOffset + ( 2*me + 0 + 64 )*128] = (*R2) * 6.1035156250000000e-05f;
	bufOut[outOffset + ( 2*me + 0 + 96 )*128] = (*R3) * 6.1035156250000000e-05f;
	bufOut[outOffset + ( 2*me + 1 + 0 )*128] = (*R4) * 6.1035156250000000e-05f;
	bufOut[outOffset + ( 2*me + 1 + 32 )*128] = (*R5) * 6.1035156250000000e-05f;
	bufOut[outOffset + ( 2*me + 1 + 64 )*128] = (*R6) * 6.1035156250000000e-05f;
	bufOut[outOffset + ( 2*me + 1 + 96 )*128] = (*R7) * 6.1035156250000000e-05f;
	}

}

 typedef union  { uint u; int i; } cb_t;

__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_fwd(__constant cb_t *cb __attribute__((max_constant_size(32))), __global const float2 * restrict gbIn, __global float2 * restrict gbOut)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[512];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

	float2 R0, R1, R2, R3, R4, R5, R6, R7;

	uint rw = (me < ((128 * cb[1].u) - batch*4)*16) ? 1 : 0;

	uint b = 0;

	iOffset = ((batch*4 + (me/16))/128)*16384 + ((batch*4 + (me/16))%128)*1;
	oOffset = ((batch*4 + (me/16))/128)*16384 + ((batch*4 + (me/16))%128)*1;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	FwdPass0(rw, b, me%16, 0, (me/16)*128, lwbIn, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass1(rw, b, me%16, (me/16)*128, (me/16)*128, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass2(rw, b, me%16, (me/16)*128, 0, lds, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
}

__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_back(__constant cb_t *cb __attribute__((max_constant_size(32))), __global const float2 * restrict gbIn, __global float2 * restrict gbOut)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[512];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

	float2 R0, R1, R2, R3, R4, R5, R6, R7;

	uint rw = (me < ((128 * cb[1].u) - batch*4)*16) ? 1 : 0;

	uint b = 0;

	iOffset = ((batch*4 + (me/16))/128)*16384 + ((batch*4 + (me/16))%128)*1;
	oOffset = ((batch*4 + (me/16))/128)*16384 + ((batch*4 + (me/16))%128)*1;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	InvPass0(rw, b, me%16, 0, (me/16)*128, lwbIn, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass1(rw, b, me%16, (me/16)*128, (me/16)*128, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass2(rw, b, me%16, (me/16)*128, 0, lds, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
}


