/*
 *  Copyright 2014 Maxim Milakov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_common.h"
#include "../int_fastdiv.h"

template<typename divisor_type>
__global__ void throughput_test(
	divisor_type d1,
	divisor_type d2,
	divisor_type d3,
	int dummy,
	int * buf)
{
	int elem_id = blockIdx.x * blockDim.x + threadIdx.x;

	int x1 = elem_id / d1;
	int x2 = elem_id / d2;
	int x3 = elem_id / d3;

	int aggregate = x1 + x2 + x3;	
	if (aggregate & dummy == 1)
		buf[0] = aggregate;
}

template<typename divisor_type>
__global__ void latency_test(
	divisor_type d1,
	divisor_type d2,
	divisor_type d3,
	divisor_type d4,
	divisor_type d5,
	divisor_type d6,
	divisor_type d7,
	divisor_type d8,
	divisor_type d9,
	divisor_type d10,
	int dummy,
	int * buf)
{
	int elem_id = blockIdx.x * blockDim.x + threadIdx.x;

	int x = elem_id;
	x = x / d1;
	x = x / d2;
	x = x / d3;
	x = x / d4;
	x = x / d5;
	x = x / d6;
	x = x / d7;
	x = x / d8;
	x = x / d9;
	x = x / d10;

	if (x & dummy == 1)
		buf[0] = x;
}

int main(int argc, char* argv[])
{
	int grid_size = 32 * 1024;
	int threadblock_size = 256;

	cudaEvent_t start, stop;
	float elapsed_time_slow;
	float elapsed_time_fast;

	cuda_safe_call(cudaEventCreate(&start));
	cuda_safe_call(cudaEventCreate(&stop));

	{
		std::cout << "THROUGHPUT TEST" << std::endl;

		std::cout << "Benchmarking plain division by constant... ";
		cuda_safe_call(cudaEventRecord(start, 0));
		throughput_test<int><<<grid_size, threadblock_size>>>(3, 5, 7, 0, 0);
		cuda_safe_call(cudaEventRecord(stop, 0));
		cuda_safe_call(cudaEventSynchronize(stop));
		cuda_safe_call(cudaEventElapsedTime(&elapsed_time_slow, start, stop));
		std::cout << elapsed_time_slow << " milliseconds" << std::endl;

		std::cout << "Benchmarking fast division by constant... ";
		cuda_safe_call(cudaEventRecord(start, 0));
		throughput_test<int_fastdiv><<<grid_size, threadblock_size>>>(3, 5, 7, 0, 0);
		cuda_safe_call(cudaEventRecord(stop, 0));
		cuda_safe_call(cudaEventSynchronize(stop));
		cuda_safe_call(cudaEventElapsedTime(&elapsed_time_fast, start, stop));
		std::cout << elapsed_time_fast << " milliseconds" << std::endl;

		std::cout << "Speedup = " << elapsed_time_slow / elapsed_time_fast << std::endl;
	}

	{
		std::cout << "LATENCY TEST" << std::endl;

		std::cout << "Benchmarking plain division by constant... ";
		cuda_safe_call(cudaEventRecord(start, 0));
		latency_test<int><<<grid_size, threadblock_size>>>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
		cuda_safe_call(cudaEventRecord(stop, 0));
		cuda_safe_call(cudaEventSynchronize(stop));
		cuda_safe_call(cudaEventElapsedTime(&elapsed_time_slow, start, stop));
		std::cout << elapsed_time_slow << " milliseconds" << std::endl;

		std::cout << "Benchmarking fast division by constant... ";
		cuda_safe_call(cudaEventRecord(start, 0));
		latency_test<int_fastdiv><<<grid_size, threadblock_size>>>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
		cuda_safe_call(cudaEventRecord(stop, 0));
		cuda_safe_call(cudaEventSynchronize(stop));
		cuda_safe_call(cudaEventElapsedTime(&elapsed_time_fast, start, stop));
		std::cout << elapsed_time_fast << " milliseconds" << std::endl;

		std::cout << "Speedup = " << elapsed_time_slow / elapsed_time_fast << std::endl;
	}

	cuda_safe_call(cudaEventDestroy(start));
	cuda_safe_call(cudaEventDestroy(stop));

	return 0;
}
