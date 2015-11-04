Fast integer division
=====================

Integer division is known to be relatively slow on modern CPUs and GPUs. The compiler generates ~30 instructions for a single integer division operation:

```c++
int q = n / d;
```

But if the divisor is known at compile time then the compiler calculates a pair of magic numbers `M` and `s`, such that

```c++
q = hi32bits(n * M) >> s; // it works for all integer n
```

Well, it is a little bit more complex: there are some corner cases requring additional operations. Nevertheless these multiplication and right shift remain the core of this fast division.

*int_fastdiv* class
=================

What if you have integer division and the divisor is not known at compile time? If you do integer division by the same divisor multiple times then you might use the same trick the compiler does, here in runtime. And you don't have to do it manually - **int_fastdiv** class does all the dirty work, calculating those magic numbers. All you need to do is to `#include "int_fastdiv.h"` and replace `int` type of the divisor with `int_fastdiv`.

The class has all the necessary constructors and operators defined to allow you using objects of this class as if they were integers. Specifically, it overrides / and % operators to utilize precomputed magic numbers.

Example
=======

I created this class with CUDA kernels in mind, but you should be able to use it in plain C++ code. Suppose you have a kernel which accepts integer parameter `d` and divides by this `d` somewhere in kernel code, and C code which runs this kernel:

```c++
__global__ kernel_name(int d)
{
  int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
  int q = elem_id / d;
  ...
}
...
kernel_name<<<grid_size,threadblock_size>>>(rand());
```

Here we add `#include` directive and replace the type of the kernel's parameter with `int_fastdiv`:

```c++
#include "int_fastdiv.h"
...
__global__ kernel_name(int_fastdiv d)
{
  int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
  int q = elem_id / d;
  ...
}
...
kernel_name<<<grid_size,threadblock_size>>>(rand());
```

That's it. `int_fastdiv` object will be constructed right when you call the kernel - on the host, once. Each thread of the CUDA kernel will utlizie fast integer division procedure when dividing `elem_id` by `d` of type `int_fastdiv`.

Performance
===========

Experiments show that division by `int_fastdiv` is about 2x faster than plain division by integer.

License
=======

It is an open-source software distributed under the [Apache License v2.0](http://www.apache.org/licenses/LICENSE-2.0).
