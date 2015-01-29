all: performance_test functional_test

performance_test: performance_test.cu ../int_fastdiv.h Makefile
	nvcc -o performance_test performance_test.cu -lineinfo -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=\"compute_50,sm_50\"

functional_test: functional_test.cu ../int_fastdiv.h Makefile
	nvcc -o functional_test functional_test.cu -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=\"compute_50,sm_50\"

clean:
	nvcc -o performance_test performance_test.cu -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=\"compute_50,sm_50\" -clean
	nvcc -o functional_test functional_test.cu -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=\"compute_50,sm_50\" -clean
	$(RM) performance_test functional_test
