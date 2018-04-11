//  Copyright (c) 2018, TU Darmstadt.
//  All rights reserved.

//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree.

#include <THC/THC.h>
#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define THREADS 128
#define idx(X, Y)	((px * 2 + (X)) + (py * 2 + (Y)) * (I_width))
#define channel	blockIdx.y
#define image	blockIdx.z
#include "common.h"
#if 0 // 1 == fast math, 0 == normal math
#define nn_exp __expf
#define nn_pow __powf
#define nn_log __logf
#else
#define nn_exp exp
#define nn_pow pow
#define nn_log log
#endif
//-------------------------------------------------------------------
__device__ __forceinline__ void operator+=(float2& out, const float2 in) {
	out.x += in.x;
	out.y += in.y;
}

//-------------------------------------------------------------------
__device__ __forceinline__ float calcVMAX(const float I_q, const float It_p, const float epsilon) {
	float x = I_q - It_p;
	return x * x + epsilon;
}

//-------------------------------------------------------------------
__device__ __forceinline__ float2 calcSUM(const float I_q, const float It_p, const float epsilon, const float alpha_e, const float lambda_e, const float vmax) {
	float x = I_q - It_p;
	float a = (x * x + epsilon) / vmax;
	float b = nn_pow(a, lambda_e);
	float weight = b + alpha_e;
	
	return make_float2(I_q * weight, weight);
}

//-------------------------------------------------------------------
template<bool overlap>
__global__ void GPU_FORWARD(
	const	float* 			__restrict__ 	I,
	const	float* 			__restrict__ 	It, // I~
			float* 			__restrict__ 	O,
	const	float* const 	__restrict__	lambda,
	const	float* 	 		__restrict__	alpha,
	const	float 							epsilon,
	const	int 						I_width,
	const	int 						I_height,
	const	int 						numChannels
) {
	const uint32_t 	O_width 	= I_width / 2;
	const uint32_t 	O_height	= I_height / 2;
	const int 		tid 		= blockIdx.x * THREADS + threadIdx.x;
	const int 		px 			= tid % O_width;
	const int 		py			= tid / O_width;
	
	// bail out if this is outside the image
	if(py >= O_height)
		return;
	
	// move ptr to correct image, channel (and patch for It and O)
	I	+= (image * numChannels + channel) * I_width * I_height;
	It	+= (image * numChannels + channel) * O_width * O_height + py * O_width + px;
	O	+= (image * numChannels + channel) * O_width * O_height + py * O_width + px;
	
	// precompute some values
	const float alpha_e  = nn_exp(alpha[channel]);
	const float lambda_e = nn_exp(lambda[channel]) / 2.0f;
	
	// Stage 1: calculate VMAX
	float vmax;

	vmax =  calcVMAX(I[idx(0, 0)], *It, epsilon);
	vmax += calcVMAX(I[idx(0, 1)], *It, epsilon);
	vmax += calcVMAX(I[idx(1, 0)], *It, epsilon);
	vmax += calcVMAX(I[idx(1, 1)], *It, epsilon);

	if(overlap) {
		if(px != 0) {
			vmax += calcVMAX(I[idx(-1, 0)], *It, epsilon);
			vmax += calcVMAX(I[idx(-1, 1)], *It, epsilon);
		}
		
		if(py != 0) { 
			vmax += calcVMAX(I[idx(0, -1)], *It, epsilon);
			vmax += calcVMAX(I[idx(1, -1)], *It, epsilon);
		}
		
		if(px != 0 && py != 0) {
			vmax += calcVMAX(I[idx(-1, -1)], *It, epsilon);
		}
	}
	
	// Stage 2: calculate O(p)
	float2 sums;
	
	sums =  calcSUM(I[idx(0, 0)], *It, epsilon, alpha_e, lambda_e, vmax);
	sums += calcSUM(I[idx(0, 1)], *It, epsilon, alpha_e, lambda_e, vmax);
	sums += calcSUM(I[idx(1, 0)], *It, epsilon, alpha_e, lambda_e, vmax);
	sums += calcSUM(I[idx(1, 1)], *It, epsilon, alpha_e, lambda_e, vmax);

	if(overlap) {
		if(px != 0) {
			sums += calcSUM(I[idx(-1, 0)], *It, epsilon, alpha_e, lambda_e, vmax);
			sums += calcSUM(I[idx(-1, 1)], *It, epsilon, alpha_e, lambda_e, vmax);
		}
		
		if(py != 0) { 
			sums += calcSUM(I[idx(0, -1)], *It, epsilon, alpha_e, lambda_e, vmax);
			sums += calcSUM(I[idx(1, -1)], *It, epsilon, alpha_e, lambda_e, vmax);
		}
		
		if(px != 0 && py != 0) {
			sums += calcSUM(I[idx(-1, -1)], *It, epsilon, alpha_e, lambda_e, vmax);
		}
	}
	
	// store result
	*O = sums.x / sums.y;
}

//-------------------------------------------------------------------
__device__ void reduce(float* out, float total) {
	const uint32_t numWarps = THREADS / 32;
	
	__shared__ float sum[numWarps];
	
	total += __shfl_down(total, 16);
	total += __shfl_down(total, 8);
	total += __shfl_down(total, 4);
	total += __shfl_down(total, 2);
	total += __shfl_down(total, 1);
	
	if(threadIdx.x % 32 == 0)
		sum[threadIdx.x / 32] = total;
	
	__syncthreads();
	
	if(threadIdx.x < numWarps) {
		total = sum[threadIdx.x];
		
		total += __shfl_down(total, 2);
		total += __shfl_down(total, 1);
		
		if(threadIdx.x == 0)
			atomicAdd(out, total);
	}
}

//-------------------------------------------------------------------------------
//-------------------------------------------------------------------
void check(cudaError err) {
	if(err != cudaSuccess) {
		std::cerr << "CUDA_ERROR: " << (int)err << " " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
}

//-------------------------------------------------------------------
void calcLaunchConfig(const uint32_t I_width, const uint32_t I_height, const uint32_t numImages, const uint32_t numChannels, dim3& threads, dim3& blocks) {
	// calc launch config
	const uint32_t O_width = I_width / 2;
	const uint32_t O_height = I_height / 2;
	
	const uint32_t numPixels = O_width * O_height;
	threads = dim3(THREADS, 1, 1);
	blocks  = dim3((uint32_t)std::ceil(numPixels / (float)THREADS), numChannels, numImages);
}
//-----------------------------------------------------------------------------------------------------------------------

extern "C"
void SpatialInverseBilateralPooling_updateOutput(THCState* state, THCudaTensor* I, THCudaTensor* It,THCudaTensor* lambda,THCudaTensor* alpha, 
	THCudaTensor* output, int kW, int kH, int dW, int dH)
{
	//VISINF_assertSameGPU(state, 5, I,It,lambda,alpha, output);
	long nInputCols, nInputRows, nInputPlane, batchSize;

	if (I->nDimension == 3) {
		nInputCols = I->size[2];
		nInputRows = I->size[1];
		nInputPlane = I->size[0];
		batchSize = 1;
	}
        else if (I->nDimension == 2){
		nInputCols = 1;
		nInputRows = 1;
		nInputPlane = I->size[1];
		batchSize = I->size[0];
	}
	else
	{
		nInputCols = I->size[3];
		nInputRows = I->size[2];
		nInputPlane = I->size[1];
		batchSize = I->size[0];
	}

	long nOutputCols = ceil(float(nInputCols - kW) / float(dW)) + 1;
	long nOutputRows = ceil(float(nInputRows - kH) / float(dH)) + 1;

	I = THCudaTensor_newContiguous(state, I);
	It = THCudaTensor_newContiguous(state, It);
	lambda = THCudaTensor_newContiguous(state, lambda);
	alpha = THCudaTensor_newContiguous(state, alpha);
	float* I_data = THCudaTensor_data(state, I);
	float* It_data = THCudaTensor_data(state, It);
	float* lambda_data = THCudaTensor_data(state, lambda);
	float* alpha_data = THCudaTensor_data(state, alpha);

	THCudaTensor_resize4d(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
	THCudaTensor_resizeAs(state, It, output);

	float* output_data = THCudaTensor_data(state, output);
	float epsilon = 0.001f;

	// get launch config
	dim3 threads, blocks;
	calcLaunchConfig(nInputCols, nInputRows, batchSize, nInputPlane, threads, blocks);
			
	// init some variables
	check(cudaMemset(output_data,  0, sizeof(float) * batchSize * nInputPlane * (nInputCols / 2) * (nInputRows / 2)));
	GPU_FORWARD<false> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(I_data, It_data, output_data, lambda_data, alpha_data, epsilon, nInputCols, nInputRows, nInputPlane);
	
	// wait for kernels
	//check(cudaDeviceSynchronize());
        //check(cudaStreamSynchronize(THCState_getCurrentStream(state)));
	THCudaCheck(cudaGetLastError());

	if(I->nDimension == 3)
		THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);

	THCudaTensor_free(state, I);
	THCudaTensor_free(state, It);
	THCudaTensor_free(state, lambda);
	THCudaTensor_free(state, alpha);
}
//-----------------------------------------------------------------------------------------------------------------------------------------
__global__ void GPU_BACKWARD_I_V1(
	const	float* 			__restrict__ 	gradOutput,
	const	float* 			__restrict__ 	I,
			float*			__restrict__	I_gradient,
	const	float* 			__restrict__ 	Itilde,
			float*			__restrict__	Itilde_gradient,
	const	float* 	 		__restrict__	lambda,
	const	float* 	 		__restrict__	alpha,
	const	float 					epsilon,
	const	uint32_t 				I_width,
	const	uint32_t 				I_height,
	const	uint32_t 				numChannels
) {
	const uint32_t 			O_width 	= I_width / 2;
	const uint32_t   	O_height	= I_height / 2;
	const int 		tid 		= blockIdx.x * THREADS + threadIdx.x;
	const int 		px 		= tid % O_width;
	const int 		py		= tid / O_width;
	
	// bail out if this is outside the image
	if(py >= O_height)
		return;
	
	// move ptr to correct image, channel (and patch for It and O)
	I			+= (image * numChannels + channel) * I_width * I_height;
	I_gradient		+= (image * numChannels + channel) * I_width * I_height;
	Itilde			+= (image * numChannels + channel) * O_width * O_height + py * O_width + px;
		gradOutput		+= (image * numChannels + channel) * O_width * O_height + py * O_width + px;
	Itilde_gradient			+= (image * numChannels + channel) * O_width * O_height + py * O_width + px;
	
	// constants
	const float It	= *Itilde;
		const float gOut= *gradOutput;
	const float ea 	= nn_exp(alpha[channel]);
	const float el 	= nn_exp(lambda[channel]) / 2.0f;
	
	// general
	float n 		= 0.0f;
	float A			= 0.0f;
	float C			= 0.0f;
	
	//---------------------------------------------------------------
	// iterate all pixels
	#pragma unroll
	for(int dy = 0; dy <= 1; ++dy) {
		#pragma unroll
		for(int dx = 0; dx <= 1; ++dx) {
			// values
			float In = I[idx(dx, dy)];
			float x = In - It;					// (In - It)
			float x2 = x * x;					// (In - It)^2
			float v  = 2.0f * In - 2.0f * It;					// 2In - 2It
			n++;
			
			// general
			A += x2;
			C += v;
		}
	}
	
	A += n * epsilon;
	
	//---------------------------------------------------------------
	float B				= 0.0f;
	float D	 			= 0.0f;
	float tilde_0			= 0.0f;
	float tilde_1			= 0.0f;
	
	float image_0[2][2] = {0};
	float image_1[2][2] = {0};
	float image_2[2][2] = {0};
	float image_3[2][2] = {0};
	
	// iterate all pixels
	#pragma unroll
	for(int dy = 0; dy <= 1; ++dy) {
		#pragma unroll
		for(int dx = 0; dx <= 1; ++dx) {
			// values
//printf("------------------------------------------------------------------------------------------ \n");
			float In = I[idx(dx, dy)];
					  
	
			float x  = In - It;								// (In - It)
			float x2 = x * x;								// (In - It)^2
			float y  = x2 + epsilon;							// (In - It)^2 + 1/1000
			float yA = y / A;								// ((ln - It)^2 + 1/1000) / A
			float z  = nn_pow(yA, el);							// (((ln - It)^2 + 1/1000) / A)^((e^ln)/2)
			float z1 = nn_pow(yA, el-1.0f);							// (((ln - It)^2 + 1/1000) / A)^((e^ln)/2 - 1)
			float v  = 2.0f * In - 2.0f * It;						// 2In - 2It
			float w  = v / A - (y * C) / (A * A); 						// (2In - 2It)/A - (y*C)/A^2
			float ww  = v / A - (y * v) / (A * A); 						// (2In - 2It)/A - (y*v)/A^2
			float k  = nn_log(yA);								// log(yA)
						
			// general
			B += z;
		
			D += In * (ea + z);
		
			// tilde
			float tt 	= el * w * z1;
		
			tilde_0		+= tt;
		
			tilde_1		+= In * tt;
		
			
			// image
			image_0[dy][dx] = ea + z + (In * el * ww * z1);
			image_2[dy][dx] = -(el * ww * z1) ;
			
			float it1 = (In * el * y * z1) / ( A * A);
			float it3 = (el * y * z1) / (A * A);
			  //printf("ea: %f el: %f I: %f It: %f w: %f z: %f z1: %f B: %f D: %f  \n", ea, el, In, It, w, z, z1,B, D);
			#pragma unroll
			for(int iy = 0; iy <= 1; ++iy) {
				#pragma unroll
				for(int ix = 0; ix <= 1; ++ix) {
					if(ix == dx && iy == dy) continue;
					
					image_1[iy][ix] += 2.0f * (I[idx(ix, iy)]-It) * it1;
					image_3[iy][ix] += 2.0f * (I[idx(ix, iy)]-It) * it3;
				}
			}
//printf("------------------------------------------------------------------------------------------ \n");
		}		
	}
	
	B += n * ea;
	//printf("final B: %f \n",B);
	//D *= n * ea;
	
	//---------------------------------------------------------------
	// final image

//printf("image_0[0][0]: %f image_0[0][1]: %f image_0[1][0]: %f image_0[1][1]: %f \n image_1[0][0]: %f image_1[0][1]: %f image_1[1][0]: %f image_1[1][1]: %f \n image_2[0][0]: %f image_2[0][1]: %f image_2[1][0]: %f image_2[1][1]: %f \n image_3[0][0]: %f image_3[0][1]: %f image_3[1][0]: %f image_3[1][1]: %f \n B: %f D: %f \n --------------------------------------------------------------\n",image_0[0][0],image_0[0][1],image_0[1][0],image_0[1][1],image_1[0][0],image_1[0][1],image_1[1][0],image_1[1][1],image_2[0][0],image_2[0][1],image_2[1][0],image_2[1][1],image_3[0][0],image_3[0][1],image_3[1][0],image_3[1][1],B,D);

	#pragma unroll
	for(int dy = 0; dy <= 1; ++dy) {
		#pragma unroll
		for(int dx = 0; dx <= 1; ++dx) {
			I_gradient[idx(dx, dy)] = gOut * ((image_0[dy][dx] - image_1[dy][dx]) / B + ((image_2[dy][dx] + image_3[dy][dx]) * D) / (B * B));
		}
	}
	
	//---------------------------------------------------------------
	// final tilde
	*Itilde_gradient =  gOut * (D * tilde_0 / (B*B) - tilde_1 / B);
}


extern "C"
void SpatialInverseBilateralPooling_updateGradInput(THCState* state, THCudaTensor* I,THCudaTensor* It, THCudaTensor* lambda,THCudaTensor* alpha,
	THCudaTensor* gradI, THCudaTensor* gradIt, THCudaTensor* gradOutput, int kW, int kH, int dW, int dH)
{
//VISINF_assertSameGPU(state, 7, I,It,lambda,alpha,gradI,gradIt, gradOutput);
  long nInputCols, nInputRows, nInputPlane, batchSize;

  if (I->nDimension == 3) {
	nInputCols = I->size[2];
	nInputRows = I->size[1];
	nInputPlane = I->size[0];
	batchSize = 1;
  }
 else if (I->nDimension == 2){
	nInputCols = 1;
	nInputRows = 1;
	nInputPlane = I->size[1];
	batchSize = I->size[0];
  }
  else
  {
	nInputCols = I->size[3];
	nInputRows = I->size[2];
	nInputPlane = I->size[1];
	batchSize = I->size[0];
  }
  float epsilon = 0.001f;
  long nOutputCols = ceil(float(nInputCols - kW) / float(dW)) + 1;
  long nOutputRows = ceil(float(nInputRows - kH) / float(dH)) + 1;

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  THCudaTensor_resizeAs(state, gradI, I);
  THCudaTensor_resizeAs(state, gradIt, It);
  
 // get launch config
	dim3 threads, blocks;
	calcLaunchConfig(nInputCols, nInputRows, batchSize, nInputPlane, threads, blocks);
			
	// VARIANT 1: without overlap
	GPU_BACKWARD_I_V1<<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, gradOutput), THCudaTensor_data(state, I), THCudaTensor_data(state, gradI), THCudaTensor_data(state, It), THCudaTensor_data(state, gradIt), THCudaTensor_data(state, lambda), THCudaTensor_data(state, alpha), epsilon, nInputCols, nInputRows, nInputPlane);
	
	// VARIANT 2: with overlap
	//GPU_BACKWARD_I_V2<<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(I, gradI, It, gradIt, lambda, alpha, epsilon, nInputCols, nInputRows, nInputPlane);
	
	// wait for kernels
	//check(cudaDeviceSynchronize());
	THCudaCheck(cudaGetLastError());
  

  THCudaTensor_free(state, gradOutput);
}




//-------------------------------------------------------------------
__global__ void GPU_BACKWARD_LA_V1(
	  	const	float* 			__restrict__ 	gradOutput,
	const	float* 			__restrict__ 	I,
	const	float* 			__restrict__ 	Itilde,
	const	float* 	 		__restrict__	lambda,
			float* 	 		__restrict__	lambda_gradient,
	const	float* 	 		__restrict__	alpha,
		float*			__restrict__	alpha_gradient,
	const	float 					epsilon,
	const	uint32_t 				I_width,
	const	uint32_t 				I_height,
	const	uint32_t 				numChannels
) {
	const uint32_t 			O_width 	= I_width / 2;
	const uint32_t 			O_height	= I_height / 2;
	const int 		tid 		= blockIdx.x * THREADS + threadIdx.x;
	const int 		px 			= tid % O_width;
	const int 		py			= tid / O_width;
	
	// move ptr to correct image, channel (and patch for It and O)
	I			+= (image * numChannels + channel) * I_width * I_height;
	Itilde			+= (image * numChannels + channel) * O_width * O_height + py * O_width + px;
 	gradOutput		+= (image * numChannels + channel) * O_width * O_height + py * O_width + px;
	alpha_gradient			+= channel;
	lambda_gradient			+= channel;
	
	float n 		= 0.0f;
	float A			= 0.0f;
	float C			= 0.0f;
	float alpha_0 			= 0.0f;
	float B			= 0.0f;
	float Dea	 	= 0.0f;
		float D	 		= 0.0f;
	float lamdba_0			= 0.0f;
	float lamdba_1			= 0.0f;
	
	// bail out if this is outside the image
	if(py < O_height) {	
		// constants
		const float It		= *Itilde;
		const float gOut	= *gradOutput;
		const float ea 		= nn_exp(alpha[channel]);
		const float el 		= nn_exp(lambda[channel]) / 2.0f;
		
		//---------------------------------------------------------------
		// iterate all pixels
		#pragma unroll
		for(int dy = 0; dy <= 1; ++dy) {
			#pragma unroll
			for(int dx = 0; dx <= 1; ++dx) {
				// values
				float In = I[idx(dx, dy)];
				float x  = In - It;					// (In - It)
				float x2 = x * x;					// (In - It)^2
				float v  = 2.0f * In - 2.0f * It;					// 2In - 2It
				n++;
				
				// general
				A += x2;
				C += v;
				
				// alpha
				alpha_0 += In;
			}
		}
		
		alpha_0 	*= ea;
		A 			+= n * epsilon;
		
		//---------------------------------------------------------------
		// iterate all pixels
		#pragma unroll
		for(int dy = 0; dy <= 1; ++dy) {
			#pragma unroll
			for(int dx = 0; dx <= 1; ++dx) {
				// values
				float In = I[idx(dx, dy)];
				float x  = In - It;										// (In - It)
				float x2 = x * x;										// (In - It)^2
				float y  = x2 + epsilon;									// (In - It)^2 + 1/1000
				float yA = y / A;										// ((ln - It)^2 + 1/1000) / A
				float z  = nn_pow(yA, el);									// (((ln - It)^2 + 1/1000) / A)^((e^ln)/2)
				float z1 = nn_pow(yA, el-1.0f);										// (((ln - It)^2 + 1/1000) / A)^((e^ln)/2 - 1)
				float k  = nn_log(yA);										// log(yA)
							
				// general
				B += z;
				D += In * (ea + z);
				
				// lamdba
				lamdba_0	+= In * k * el * z;
				lamdba_1	+= k * el * z ;
			}		
		}
		
		B		+= n * ea;
		Dea 	  = D * n * ea * gOut;
		alpha_0  *= gOut;
		lamdba_0 *= gOut;
		lamdba_1 *= gOut;
	}
	
	//---------------------------------------------------------------
	// prevent division by 0!
	float final_alpha, final_lambda;
	if(B == 0.0f) {
		final_alpha = 0.0f;
		final_lambda = 0.0f;
	} else {
		final_alpha =  alpha_0 / B - Dea / (B * B);
		final_lambda = lamdba_0 / B - (D * lamdba_1 / (B * B));
	}
	
	// final alpha
	reduce(alpha_gradient, final_alpha);
	
	// final lamdba
	reduce(lambda_gradient, final_lambda);
}

//-------------------------------------------------------------------
extern "C"
void SpatialInverseBilateralPooling_accGradParameters(THCState* state, THCudaTensor* I,THCudaTensor* It, THCudaTensor* lambda,THCudaTensor* alpha,
	THCudaTensor* lambda_gradient, THCudaTensor* alpha_gradient, THCudaTensor* gradOutput, int kW, int kH, int dW, int dH)
{
//VISINF_assertSameGPU(state, 7, I,It,lambda,alpha,gradLambda,gradAlpha, gradOutput);
  long nInputCols, nInputRows, nInputPlane, batchSize;

  if (I->nDimension == 3) {
	nInputCols = I->size[2];
	nInputRows = I->size[1];
	nInputPlane = I->size[0];
	batchSize = 1;
  }
  else if (I->nDimension == 2){
	nInputCols = 1;
	nInputRows = 1;
	nInputPlane = I->size[1];
	batchSize = I->size[0];
  }
  else

  {
	nInputCols = I->size[3];
	nInputRows = I->size[2];
	nInputPlane = I->size[1];
	batchSize = I->size[0];
  }
  float epsilon = 0.001f;
  long nOutputCols = ceil(float(nInputCols - kW) / float(dW)) + 1;
  long nOutputRows = ceil(float(nInputRows - kH) / float(dH)) + 1;

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  THCudaTensor_resizeAs(state, lambda_gradient, lambda);
  THCudaTensor_resizeAs(state, alpha_gradient, alpha);
  
		// get launch config
	dim3 threads, blocks;
	calcLaunchConfig(nInputCols, nInputRows, batchSize, nInputPlane, threads, blocks);
			
	// init some variables
	check(cudaMemset(THCudaTensor_data(state, alpha_gradient),   0, sizeof(float) * nInputPlane));
	check(cudaMemset(THCudaTensor_data(state, lambda_gradient),  0, sizeof(float) * nInputPlane));
	
	// VARIANT 1: without overlap
	GPU_BACKWARD_LA_V1<<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, gradOutput), THCudaTensor_data(state, I), THCudaTensor_data(state, It), THCudaTensor_data(state, lambda), THCudaTensor_data(state, lambda_gradient), THCudaTensor_data(state, alpha), THCudaTensor_data(state, alpha_gradient), epsilon, nInputCols, nInputRows, nInputPlane);

	// VARIANT 2: with overlap
	//GPU_BACKWARD_LA_V2<<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(I, It, lambda, lambda_gradient, alpha, alpha_gradient, epsilon, I_width, I_height, numChannels);
	
	// wait for kernels
	//check(cudaDeviceSynchronize());
	THCudaCheck(cudaGetLastError());
	THCudaTensor_free(state, gradOutput);
}
