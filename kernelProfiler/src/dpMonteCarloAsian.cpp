
/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "dpMonteCarloAsian.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpMonteCarloAsian::dpMonteCarloAsian(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;

	name = "MonteCarloAsian";
	kernelString = "\n"
"typedef struct _MonteCalroAttribVector                                                                              \n"
"{                                                                                                                   \n"
"    float4 strikePrice;                                                                                             \n"
"    float4 c1;                                                                                                      \n"
"    float4 c2;                                                                                                      \n"
"    float4 c3;                                                                                                      \n"
"    float4 initPrice;                                                                                               \n"
"    float4 sigma;                                                                                                   \n"
"    float4 timeStep;                                                                                                \n"
"}MonteCarloAttribVector;                                                                                            \n"
"                                                                                                                    \n"
"__constant uint mulFactor = 4;                                                                                      \n"
"__constant uint stateMask = 1812433253u;                                                                            \n"
"__constant uint thirty = 30u;                                                                                       \n"
"__constant uint thirteen  = 13u;                                                                                    \n"
"__constant uint fifteen = 15u;                                                                                      \n"
"__constant uint threeBytes = 8u * 3u;                                                                               \n"
"__constant uint mask[4] = {0xfdff37ffu, 0xef7f3f7du, 0xff777b7du, 0x7ff7fb2fu};                                     \n"
"__constant float one = 1.0f;                                                                                        \n"
"__constant float intMax = 4294967296.0f;                                                                            \n"
"__constant float PI = 3.14159265358979f;                                                                            \n"
"__constant float two = 2.0f;                                                                                        \n"
"                                                                                                                    \n"
"/**                                                                                                                 \n"
"* @brief Left shift                                                                                                 \n"
"* @param input input to be shifted                                                                                  \n"
"* @param output result after shifting input                                                                         \n"
"*/                                                                                                                  \n"
"void                                                                                                                \n"
"    lshift128(uint4 input, uint4* output)                                                                           \n"
"{                                                                                                                   \n"
"    unsigned int invshift = 32u - threeBytes;                                                                       \n"
"                                                                                                                    \n"
"    uint4 temp;                                                                                                     \n"
"    temp.x = input.x << threeBytes;                                                                                 \n"
"    temp.y = (input.y << threeBytes) | (input.x >> invshift);                                                       \n"
"    temp.z = (input.z << threeBytes) | (input.y >> invshift);                                                       \n"
"    temp.w = (input.w << threeBytes) | (input.z >> invshift);                                                       \n"
"                                                                                                                    \n"
"    *output = temp;                                                                                                 \n"
"}                                                                                                                   \n"
"                                                                                                                    \n"
"/**                                                                                                                 \n"
"* @brief Right shift                                                                                                \n"
"* @param input input to be shifted                                                                                  \n"
"* @param output result after shifting input                                                                         \n"
"*/                                                                                                                  \n"
"void                                                                                                                \n"
"    rshift128(uint4 input, uint4* output)                                                                           \n"
"{                                                                                                                   \n"
"    unsigned int invshift = 32u - threeBytes;                                                                       \n"
"                                                                                                                    \n"
"    uint4 temp;                                                                                                     \n"
"                                                                                                                    \n"
"    temp.w = input.w >> threeBytes;                                                                                 \n"
"    temp.z = (input.z >> threeBytes) | (input.w << invshift);                                                       \n"
"    temp.y = (input.y >> threeBytes) | (input.z << invshift);                                                       \n"
"    temp.x = (input.x >> threeBytes) | (input.y << invshift);                                                       \n"
"                                                                                                                    \n"
"    *output = temp;                                                                                                 \n"
"}                                                                                                                   \n"
"                                                                                                                    \n"
"/**                                                                                                                 \n"
"* @brief Generates gaussian random numbers by using                                                                 \n"
"*        Mersenenne Twister algo and box muller transformation                                                      \n"
"* @param seed  initial seed value                                                                                   \n"
"* @param gaussianRand1 gaussian random number generatred                                                            \n"
"* @param gaussianRand2 gaussian random number generarted                                                            \n"
"* @param nextRand  generated seed for next usage                                                                    \n"
"*/                                                                                                                  \n"
"void                                                                                                                \n"
"    generateRand_Vector(uint4 seed,                                                                                 \n"
"    float4 *gaussianRand1,                                                                                          \n"
"    float4 *gaussianRand2,                                                                                          \n"
"    uint4 *nextRand)                                                                                                \n"
"{                                                                                                                   \n"
"    uint4 temp[4];                                                                                                  \n"
"                                                                                                                    \n"
"    uint4 state1 = seed;                                                                                            \n"
"    uint4 state2 = (uint4)(0);                                                                                      \n"
"    uint4 state3 = (uint4)(0);                                                                                      \n"
"    uint4 state4 = (uint4)(0);                                                                                      \n"
"    uint4 state5 = (uint4)(0);                                                                                      \n"
"                                                                                                                    \n"
"    uint4 mask4 = (uint4)(stateMask);                                                                               \n"
"    uint4 thirty4 = (uint4)(thirty);                                                                                \n"
"    uint4 one4 = (uint4)(1u);                                                                                       \n"
"    uint4 two4 = (uint4)(2u);                                                                                       \n"
"    uint4 three4 = (uint4)(3u);                                                                                     \n"
"    uint4 four4 = (uint4)(4u);                                                                                      \n"
"                                                                                                                    \n"
"    uint4 r1 = (uint4)(0);                                                                                          \n"
"    uint4 r2 = (uint4)(0);                                                                                          \n"
"                                                                                                                    \n"
"    uint4 a = (uint4)(0);                                                                                           \n"
"    uint4 b = (uint4)(0);                                                                                           \n"
"                                                                                                                    \n"
"    uint4 e = (uint4)(0);                                                                                           \n"
"    uint4 f = (uint4)(0);                                                                                           \n"
"                                                                                                                    \n"
"    float4 r;                                                                                                       \n"
"    float4 phi;                                                                                                     \n"
"                                                                                                                    \n"
"    float4 temp1;                                                                                                   \n"
"    float4 temp2;                                                                                                   \n"
"                                                                                                                    \n"
"    //Initializing states.                                                                                          \n"
"    state2 = mask4 * (state1 ^ (state1 >> thirty4)) + one4;                                                         \n"
"    state3 = mask4 * (state2 ^ (state2 >> thirty4)) + two4;                                                         \n"
"    state4 = mask4 * (state3 ^ (state3 >> thirty4)) + three4;                                                       \n"
"    state5 = mask4 * (state4 ^ (state4 >> thirty4)) + four4;                                                        \n"
"                                                                                                                    \n"
"    uint i = 0;                                                                                                     \n"
"    #pragma unroll 4                                                                                                \n"
"    for(i = 0; i < mulFactor; ++i)                                                                                  \n"
"    {                                                                                                               \n"
"        switch(i)                                                                                                   \n"
"        {                                                                                                           \n"
"        case 0:                                                                                                     \n"
"            r1 = state4;                                                                                            \n"
"            r2 = state5;                                                                                            \n"
"            a = state1;                                                                                             \n"
"            b = state3;                                                                                             \n"
"            break;                                                                                                  \n"
"        case 1:                                                                                                     \n"
"            r1 = r2;                                                                                                \n"
"            r2 = temp[0];                                                                                           \n"
"            a = state2;                                                                                             \n"
"            b = state4;                                                                                             \n"
"            break;                                                                                                  \n"
"        case 2:                                                                                                     \n"
"            r1 = r2;                                                                                                \n"
"            r2 = temp[1];                                                                                           \n"
"            a = state3;                                                                                             \n"
"            b = state5;                                                                                             \n"
"            break;                                                                                                  \n"
"        case 3:                                                                                                     \n"
"            r1 = r2;                                                                                                \n"
"            r2 = temp[2];                                                                                           \n"
"            a = state4;                                                                                             \n"
"            b = state1;                                                                                             \n"
"            break;                                                                                                  \n"
"        default:                                                                                                    \n"
"            break;                                                                                                  \n"
"                                                                                                                    \n"
"        }                                                                                                           \n"
"                                                                                                                    \n"
"        lshift128(a, &e);                                                                                           \n"
"        rshift128(r1, &f);                                                                                          \n"
"                                                                                                                    \n"
"        temp[i].x = a.x ^ e.x ^ ((b.x >> thirteen) & mask[0]) ^ f.x ^ (r2.x << fifteen);                            \n"
"        temp[i].y = a.y ^ e.y ^ ((b.y >> thirteen) & mask[1]) ^ f.y ^ (r2.y << fifteen);                            \n"
"        temp[i].z = a.z ^ e.z ^ ((b.z >> thirteen) & mask[2]) ^ f.z ^ (r2.z << fifteen);                            \n"
"        temp[i].w = a.w ^ e.w ^ ((b.w >> thirteen) & mask[3]) ^ f.w ^ (r2.w << fifteen);                            \n"
"    }                                                                                                               \n"
"                                                                                                                    \n"
"    temp1 = convert_float4(temp[0]) * one / intMax;                                                                 \n"
"    temp2 = convert_float4(temp[1]) * one / intMax;                                                                 \n"
"                                                                                                                    \n"
"    // Applying Box Mullar Transformations.                                                                         \n"
"    r = sqrt((-two) * log(temp1));                                                                                  \n"
"    phi  = two * PI * temp2;                                                                                        \n"
"    *gaussianRand1 = r * cos(phi);                                                                                  \n"
"    *gaussianRand2 = r * sin(phi);                                                                                  \n"
"    *nextRand = temp[2];                                                                                            \n"
"                                                                                                                    \n"
"}                                                                                                                   \n"
"                                                                                                                    \n"
"/**                                                                                                                 \n"
"* @brief   calculates the  price and vega for all trajectories                                                      \n"
"* @param strikePrice	Strike price                                                                                   \n"
"* @param meanDeriv1		Average Derive price (from gaussianRand1)                                                    \n"
"* @param meanDeriv2		Average Derive price (from gaussianRand2)                                                    \n"
"* @param meanPrice1		Average price (from gaussianRand1)                                                           \n"
"* @param meanPrice2		Average price (from gaussianRand2)                                                           \n"
"* @param pathDeriv1		path derive price (for gaussianRand1)                                                        \n"
"* @param pathDeriv2		path derive price (for gaussianRand2)                                                        \n"
"* @param price1			price (for gaussianRand1)                                                                      \n"
"* @param price2			price (for gaussianRand2)                                                                      \n"
"*/                                                                                                                  \n"
"void                                                                                                                \n"
"    calOutputs_Vector(float4 strikePrice,                                                                           \n"
"    float4 meanDeriv1,                                                                                              \n"
"    float4 meanDeriv2,                                                                                              \n"
"    float4 meanPrice1,                                                                                              \n"
"    float4 meanPrice2,                                                                                              \n"
"    float4 *pathDeriv1,                                                                                             \n"
"    float4 *pathDeriv2,                                                                                             \n"
"    float4 *priceVec1,                                                                                              \n"
"    float4 *priceVec2)                                                                                              \n"
"{                                                                                                                   \n"
"    float4 temp1 = (float4)0.0f;                                                                                    \n"
"    float4 temp2 = (float4)0.0f;                                                                                    \n"
"    float4 temp3 = (float4)0.0f;                                                                                    \n"
"    float4 temp4 = (float4)0.0f;                                                                                    \n"
"                                                                                                                    \n"
"    float4 tempDiff1 = meanPrice1 - strikePrice;                                                                    \n"
"    float4 tempDiff2 = meanPrice2 - strikePrice;                                                                    \n"
"    if(tempDiff1.x > 0.0f)                                                                                          \n"
"    {                                                                                                               \n"
"        temp1.x = 1.0f;                                                                                             \n"
"        temp3.x = tempDiff1.x;                                                                                      \n"
"    }                                                                                                               \n"
"    if(tempDiff1.y > 0.0f)                                                                                          \n"
"    {                                                                                                               \n"
"        temp1.y = 1.0f;                                                                                             \n"
"        temp3.y = tempDiff1.y ;                                                                                     \n"
"    }                                                                                                               \n"
"    if(tempDiff1.z > 0.0f)                                                                                          \n"
"    {                                                                                                               \n"
"        temp1.z = 1.0f;                                                                                             \n"
"        temp3.z = tempDiff1.z;                                                                                      \n"
"    }                                                                                                               \n"
"    if(tempDiff1.w > 0.0f)                                                                                          \n"
"    {                                                                                                               \n"
"        temp1.w = 1.0f;                                                                                             \n"
"        temp3.w = tempDiff1.w;                                                                                      \n"
"    }                                                                                                               \n"
"                                                                                                                    \n"
"    if(tempDiff2.x > 0.0f)                                                                                          \n"
"    {                                                                                                               \n"
"        temp2.x = 1.0f;                                                                                             \n"
"        temp4.x = tempDiff2.x;                                                                                      \n"
"    }                                                                                                               \n"
"    if(tempDiff2.y > 0.0f)                                                                                          \n"
"    {                                                                                                               \n"
"        temp2.y = 1.0f;                                                                                             \n"
"        temp4.y = tempDiff2.y;                                                                                      \n"
"    }                                                                                                               \n"
"    if(tempDiff2.z > 0.0f)                                                                                          \n"
"    {                                                                                                               \n"
"        temp2.z = 1.0f;                                                                                             \n"
"        temp4.z = tempDiff2.z;                                                                                      \n"
"    }                                                                                                               \n"
"    if(tempDiff2.w > 0.0f)                                                                                          \n"
"    {                                                                                                               \n"
"        temp2.w = 1.0f;                                                                                             \n"
"        temp4.w = tempDiff2.w;                                                                                      \n"
"    }                                                                                                               \n"
"                                                                                                                    \n"
"    *pathDeriv1 = meanDeriv1 * temp1;                                                                               \n"
"    *pathDeriv2 = meanDeriv2 * temp2;                                                                               \n"
"    *priceVec1 = temp3;                                                                                             \n"
"    *priceVec2 = temp4;	                                                                                           \n"
"}                                                                                                                   \n"
"                                                                                                                    \n"
"                                                                                                                    \n"
"/**                                                                                                                 \n"
"* @brief   Calculates the  price and vega for all trajectories for given random numbers(For Vector Kernel)          \n"
"* @param   attrib  structure of inputs for simulation                                                               \n"
"* @param   priceSamples    array of calculated price samples                                                        \n"
"* @param   pathDeriv   array calculated path derivatives                                                            \n"
"* @param   sData1   array used for blockwise reduction                                                              \n"
"*/                                                                                                                  \n"
"__kernel                                                                                                            \n"
"    void                                                                                                            \n"
"    calPriceVega_Vector(MonteCarloAttribVector attrib,                                                              \n"
"    int noOfSum,                                                                                                    \n"
"    __global uint4 *randArray,                                                                                      \n"
"    __global float4 *priceSamples,                                                                                  \n"
"    __global float4 *pathDeriv,                                                                                     \n"
"    __local float8 *sData1)                                                                                         \n"
"{                                                                                                                   \n"
"    float4 strikePrice = attrib.strikePrice;                                                                        \n"
"    float4 c1 = attrib.c1;                                                                                          \n"
"    float4 c2 = attrib.c2;                                                                                          \n"
"    float4 c3 = attrib.c3;                                                                                          \n"
"    float4 initPrice = attrib.initPrice;                                                                            \n"
"    float4 sigma = attrib.sigma;                                                                                    \n"
"    float4 timeStep = attrib.timeStep;                                                                              \n"
"                                                                                                                    \n"
"    int2 localIdx;                                                                                                  \n"
"    int2 groupIdx;                                                                                                  \n"
"    int2 groupDim;                                                                                                  \n"
"    localIdx.x = (int)get_local_id(0);                                                                              \n"
"    localIdx.y = (int)get_local_id(1);                                                                              \n"
"    groupIdx.x = (int)get_group_id(0);                                                                              \n"
"    groupIdx.y = (int)get_group_id(1);                                                                              \n"
"    groupDim.x = (int)get_local_size(0);                                                                            \n"
"    groupDim.y = (int)get_local_size(1);                                                                            \n"
"                                                                                                                    \n"
"    int xDim = (int)get_global_size(0);                                                                             \n"
"    int yDim = (int)get_global_size(1);                                                                             \n"
"    int xPos = (int)get_global_id(0);                                                                               \n"
"    int yPos = (int)get_global_id(1);                                                                               \n"
"    int gidx=yPos * xDim + xPos;                                                                                    \n"
"    int bidx=groupIdx.y*xDim/groupDim.x+groupIdx.x;                                                                 \n"
"    int lidx=localIdx.y*groupDim.x + localIdx.x;                                                                    \n"
"                                                                                                                    \n"
"    float4 temp = (float4)0.0f;                                                                                     \n"
"    float4 temp1 = (float4)0.0f;                                                                                    \n"
"    float4 temp2 = (float4)0.0f;                                                                                    \n"
"                                                                                                                    \n"
"    float4 price1 = (float4)0.0f;                                                                                   \n"
"    float4 price2 = (float4)0.0f;                                                                                   \n"
"    float4 pathDeriv1 = (float4)0.0f;                                                                               \n"
"    float4 pathDeriv2 = (float4)0.0f;                                                                               \n"
"                                                                                                                    \n"
"    float4 trajPrice1 = initPrice;                                                                                  \n"
"    float4 trajPrice2 = initPrice;                                                                                  \n"
"                                                                                                                    \n"
"    float4 sumPrice1 = initPrice;                                                                                   \n"
"    float4 sumPrice2 = initPrice;                                                                                   \n"
"                                                                                                                    \n"
"    float4 meanPrice1 = temp;                                                                                       \n"
"    float4 meanPrice2 = temp;                                                                                       \n"
"                                                                                                                    \n"
"    float4 sumDeriv1 = temp;                                                                                        \n"
"    float4 sumDeriv2 = temp;                                                                                        \n"
"                                                                                                                    \n"
"    float4 meanDeriv1 = temp;                                                                                       \n"
"    float4 meanDeriv2 = temp;                                                                                       \n"
"                                                                                                                    \n"
"    float4 finalRandf1 = temp;                                                                                      \n"
"    float4 finalRandf2 = temp;                                                                                      \n"
"                                                                                                                    \n"
"    uint4 nextRand = randArray[gidx];                                                                               \n"
"                                                                                                                    \n"
"    //Run the Monte Carlo simulation a total of Num_Sum - 1 times                                                   \n"
"    for(int i = 1; i < noOfSum; i++)                                                                                \n"
"    {                                                                                                               \n"
"        uint4 tempRand = nextRand;                                                                                  \n"
"        generateRand_Vector(tempRand, &finalRandf1, &finalRandf2, &nextRand);                                       \n"
"                                                                                                                    \n"
"        //Calculate the trajectory price and sum price for all trajectories                                         \n"
"                                                                                                                    \n"
"        temp1 += c1 + c2 * finalRandf1;                                                                             \n"
"        temp2 += c1 + c2 * finalRandf2;                                                                             \n"
"        trajPrice1 = trajPrice1 * exp(c1 + c2 * finalRandf1);                                                       \n"
"        trajPrice2 = trajPrice2 * exp(c1 + c2 * finalRandf2);                                                       \n"
"                                                                                                                    \n"
"        sumPrice1 = sumPrice1 + trajPrice1;                                                                         \n"
"        sumPrice2 = sumPrice2 + trajPrice2;                                                                         \n"
"                                                                                                                    \n"
"        temp = c3 * timeStep * i;                                                                                   \n"
"                                                                                                                    \n"
"        // Calculate the derivative price for all trajectories                                                      \n"
"        sumDeriv1 = sumDeriv1 + trajPrice1                                                                          \n"
"            * (temp1 - temp) / sigma;                                                                               \n"
"                                                                                                                    \n"
"        sumDeriv2 = sumDeriv2 + trajPrice2                                                                          \n"
"            * (temp2 - temp) / sigma;                                                                               \n"
"    }                                                                                                               \n"
"                                                                                                                    \n"
"    //Calculate the average price and average derivative of each simulated path                                     \n"
"    meanPrice1 = sumPrice1 / noOfSum;                                                                               \n"
"    meanPrice2 = sumPrice2 / noOfSum;                                                                               \n"
"    meanDeriv1 = sumDeriv1 / noOfSum;                                                                               \n"
"    meanDeriv2 = sumDeriv2 / noOfSum;                                                                               \n"
"                                                                                                                    \n"
"    calOutputs_Vector(strikePrice, meanDeriv1, meanDeriv2, meanPrice1,                                              \n"
"        meanPrice2, &pathDeriv1, &pathDeriv2, &price1, &price2);                                                    \n"
"                                                                                                                    \n"
"     //Do the reduction blockwise and store the result in sData1[0]                                                 \n"
"     sData1[lidx]=(float8)(price1+price2,pathDeriv1+pathDeriv2);                                                    \n"
"     barrier(CLK_LOCAL_MEM_FENCE);                                                                                  \n"
"                                                                                                                    \n"
"    for(unsigned int s = (groupDim.x*groupDim.y)>> 1; s > 0; s >>= 1)                                               \n"
"    {                                                                                                               \n"
"        if(lidx < s)                                                                                                \n"
"        {                                                                                                           \n"
"            sData1[lidx] += sData1[lidx + s];                                                                       \n"
"        }                                                                                                           \n"
"          barrier(CLK_LOCAL_MEM_FENCE);	                                                                           \n"
"    }                                                                                                               \n"
"                                                                                                                    \n"
"    // Write the reduction result of  this block to global memory                                                   \n"
"    if ((localIdx.y==0) && (localIdx.x==0) ){                                                                       \n"
"    priceSamples[bidx] = sData1[0].lo;                                                                              \n"
"    pathDeriv[bidx] = sData1[0].hi;                                                                                 \n"
"                                                                                                                    \n"
"    }                                                                                                               \n"
"}                                                                                                                   \n";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	clErrChk(err);
	programCheck(err, context, program);
	kernel = clCreateKernel(program, "calPriceVega_Vector", &err); clErrChk(err);
}

void dpMonteCarloAsian::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = xLocal;
	localSize[1] = yLocal;
	localSize[2] = 1;
	
	noOfTraj = (int) sqrt(dataMB*1048576*8/sizeof(cl_float4));
	
	if (noOfTraj%4 != 0)
		noOfTraj = noOfTraj - noOfTraj%4 + 4;
	
	width = noOfTraj / 4;
	height = noOfTraj / 2;
	
	if ((width==0) || (height==0)){ //making sure kernel won't launch with too small of a data size
		noOfTraj = 512;
		width = 128;
		height = 256;
	}
	
	//MB = noOfTraj^2 /8 *sizeof(cl_float4);
	MB=(width*height*sizeof(cl_float4))/(float) 1048576;
	
}

void dpMonteCarloAsian::init(){

	noOfSum = 12;
	vectorWidth = 4;
	steps = 2;
	initPrice = 50.f;
	strikePrice = 55.f;
	interest = 0.06f;
	maturity = 1.f;
	iterations = 1;
	
	steps = (steps < 4) ? 4 : steps;
	steps = (steps / 2) * 2;
	
	dataParameters.push_back(width);
	dataParameters.push_back(height);
	dataParameters.push_back(steps);
	
	dataNames.push_back("width");
	dataNames.push_back("height");
	dataNames.push_back("steps");

	const cl_float finalValue = 0.8f;
	const cl_float stepValue = finalValue / (cl_float)steps;

	// Allocate and init memory used by host
	sigma = (cl_float*)malloc(steps * sizeof(cl_float));

	sigma[0] = 0.01f;
	for(int i = 1; i < steps; i++){
		sigma[i] = sigma[i - 1] + stepValue;
	}

	price = (cl_float*) malloc(steps * sizeof(cl_float));
	memset((void*)price, 0, steps * sizeof(cl_float));

	vega = (cl_float*) malloc(steps * sizeof(cl_float));
	memset((void*)vega, 0, steps * sizeof(cl_float));
	
	randNum = (cl_uint*)memalign(16, width * height * sizeof(cl_uint4) * steps);

	// Generate random data
	srand(time(NULL));
	for(int i = 0; i < (width * height * 4 * steps); i++){
		randNum[i] = (cl_uint)rand();
	}

	priceVals = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
	memset((void*)priceVals, 0, width * height * 2 * sizeof(cl_float4));

	priceDeriv = (cl_float*)malloc(width * height * 2 * sizeof(cl_float4));
	memset((void*)priceDeriv, 0, width * height * 2 * sizeof(cl_float4));

}

void dpMonteCarloAsian::memoryCopyOut(){
	// create Normal Buffer, if persistent memory is not in used
	randBuf = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,sizeof(cl_uint4) * width  * height,randNum,&err);
	clErrChk(err);

	// create Buffer for PriceBuf
	priceBuf = clCreateBuffer(context,CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_float4) * width * height * 2,NULL,&err);
	clErrChk(err);
																	
	// create Buffer for PriceDeriveBuffer
	priceDerivBuf = clCreateBuffer(context,CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_float4) * width * height * 2,NULL,&err);
	clErrChk(err);
	clErrChk(clFinish(queue));
}

void dpMonteCarloAsian::plan(){
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&noOfSum));
	globalSize[0] = width;
	globalSize[1] = height;
} 

int dpMonteCarloAsian::execute(){

	size_t sizeAfterReduction = width * height  / localSize[0] / localSize[1] * 4;
	
	for(int k = 0; k < steps; k++){
		clErrChk(clEnqueueWriteBuffer(queue,randBuf,CL_TRUE,0, width * height * sizeof(cl_float4),(randNum+(k*width*height*4)),0,NULL,NULL));
		setKernelArgs(k, &randBuf, &priceBuf, &priceDerivBuf);
		
		// Enqueue a kernel run call.
		err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
		clErrChk(err);
		if(err<0)
			return -1;
		clErrChk(err);
		
		clErrChk(clFlush(queue));

		// Enqueue the results to application pointer
		clErrChk(clEnqueueReadBuffer(queue,priceBuf,CL_TRUE,0,width * height * 2 * sizeof(cl_float4),priceVals,0,NULL,NULL));

		// Enqueue the results to application pointer
		clErrChk(clEnqueueReadBuffer(queue,priceDerivBuf,CL_TRUE,0,width * height * 2 * sizeof(cl_float4),priceDeriv,0,NULL,NULL));

		for(size_t i = 0; i < sizeAfterReduction; i++){
			price[k] += priceVals[i];
			vega[k] += priceDeriv[i];
		}

		price[k] /= (noOfTraj * noOfTraj);
		vega[k] /= (noOfTraj * noOfTraj);

		price[k] = exp(-interest * maturity) * price[k];
		vega[k] = exp(-interest * maturity) * vega[k];
	}
	
	clErrChk(clFinish(queue));
	return 0;
}

void dpMonteCarloAsian::memoryCopyIn(){
	
	clErrChk(clFinish(queue));
}

void dpMonteCarloAsian::cleanUp(){
	   // Releases OpenCL resources (Context, Memory etc.)
	clErrChk(clReleaseMemObject(priceBuf));
	clErrChk(clReleaseMemObject(priceDerivBuf));
	clErrChk(clReleaseMemObject(randBuf));
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program)); 

	// Release program resources (input memory etc.)
	free(randNum);
	free(sigma);
	free(price);
	free(vega);
	free(priceVals);
	free(priceDeriv);
}

void dpMonteCarloAsian::setKernelArgs(int step, cl_mem *rand, cl_mem *price, cl_mem *priceDeriv){
    float timeStep = maturity / (noOfSum - 1);

    // Set up arguments required for kernel 1
    float c1 = (interest - 0.5f * sigma[step] * sigma[step]) * timeStep;
    float c2 = sigma[step] * sqrt(timeStep);
    float c3 = (interest + 0.5f * sigma[step] * sigma[step]);

		const cl_float4 c1F4 = {{c1, c1, c1, c1}};
		attributes.c1 = c1F4;

		const cl_float4 c2F4 = {{c2, c2, c2, c2}};
		attributes.c2 = c2F4;

		const cl_float4 c3F4 = {{c3, c3, c3, c3}};
		attributes.c3 = c3F4;

		const cl_float4 initPriceF4 = {{initPrice, initPrice, initPrice, initPrice}};
		attributes.initPrice = initPriceF4;

		const cl_float4 strikePriceF4 = {{strikePrice, strikePrice, strikePrice, strikePrice}};
		attributes.strikePrice = strikePriceF4;

		const cl_float4 sigmaF4 = {{sigma[step], sigma[step], sigma[step], sigma[step]}};
		attributes.sigma = sigmaF4;

		const cl_float4 timeStepF4 = {{timeStep, timeStep, timeStep, timeStep}};
		attributes.timeStep = timeStepF4;

    // Set appropriate arguments to the kernel
    clErrChk(clSetKernelArg(kernel, 0, sizeof(attributes), (void*)&attributes));
    clErrChk(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)rand));
    clErrChk(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)price));
    clErrChk(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)priceDeriv));
    clErrChk(clSetKernelArg(kernel, 5, localSize[0]*localSize[1]*sizeof(cl_float8),NULL));
}

