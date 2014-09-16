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
#include <string>

#include "cuda_common.h"
#include "../int_fastdiv.h"

__global__ void check(int_fastdiv divisor, int * results)
{
	int divident = blockIdx.x * blockDim.x + threadIdx.x;

	int quotient = divident / (int)divisor;
	int fast_quotient = divident / divisor;

	if (quotient != fast_quotient)
	{
		int error_id = atomicAdd(&results[0], 1);
		if (error_id == 0)
		{
			results[1] = divident;
			results[2] = quotient;
			results[3] = fast_quotient;
		}
	}

	divident = -divident;

	quotient = divident / (int)divisor;
	fast_quotient = divident / divisor;

	if (quotient != fast_quotient)
	{
		int error_id = atomicAdd(&results[0], 1);
		if (error_id == 0)
		{
			results[1] = divident;
			results[2] = quotient;
			results[3] = fast_quotient;
		}
	}
}

int gpu_check()
{
	const int threadblock_size = 256;
	const int divisor_count = 100000;
	const int divident_count = 1000000;

	int grid_size = (divident_count + threadblock_size - 1) / threadblock_size;

	int buf[4];
	int * buf_d;
	cuda_safe_call(cudaMalloc(&buf_d, sizeof(int) * 4));

	std::cout << "Running GPU functional test on " << divisor_count << " divisors, with " << grid_size * threadblock_size << " dividents for each divisor" << std::endl;
	for(int d = 1; d < divisor_count; ++d)
	{
		for(int sign = 1; sign >= -1; sign -= 2)
		{
			int divisor = d * sign;

			std::cout << "Checking divisor " << divisor << "... ";

			cuda_safe_call(cudaMemset(buf_d, 0, sizeof(int) * 4));
			check<<<grid_size,threadblock_size>>>(divisor, buf_d);
			cuda_safe_call(cudaMemcpy(buf, buf_d, sizeof(int) * 4, cudaMemcpyDeviceToHost));
			cuda_safe_call(cudaDeviceSynchronize());

			if (buf[0] > 0)
			{
				std::cout << buf[0] << " wrong results, one of them is for divident " << buf[1] << ", correct quotient = " << buf[2] << ", fast computed quotient = " << buf[3] << std::endl;
				return 1;
			}

			std::cout << "done" << std::endl;
		}
	}

	cuda_safe_call(cudaFree(buf_d));

	return 0;
}

int cpu_check()
{
	const int divisor_count = 100000;
	const int divident_count = 1000000;
	std::cout << "Running CPU functional test on " << divisor_count << " divisors, with " << divident_count << " dividents for each divisor" << std::endl;
	for(int d = 1; d < divisor_count; ++d)
	{
		for(int sign = 1; sign >= -1; sign -= 2)
		{
			int divisor = d * sign;

			std::cout << "Checking divisor " << divisor << "... ";

			int_fastdiv fast_divisor(divisor);

			for(int dd = 0; dd < divident_count; ++dd)
			{
				for(int ss = 1; ss >= -1; ss -= 2)
				{
					int divident = dd * ss;

					int quotient = divident / divisor;
					int fast_quotient = divident / fast_divisor;
					if (quotient != fast_quotient)
					{
						std::cout << "wrong result for divident " << divident << ", correct quotient = " << quotient << ", fast computed quotient = " << fast_quotient << std::endl;
						return 1;
					}
				}
			}

			std::cout << "done" << std::endl;
		}
	}

	return 0;
}

int main(int argc, char* argv[])
{
	bool run_gpu = false;

	if (argc > 1)
		run_gpu = (strcmp(argv[1], "gpu") == 0);

	int res;
	if (run_gpu)
		res = gpu_check();
	else
		res = cpu_check();

	return res;
}
