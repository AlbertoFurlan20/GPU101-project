## Optimization steps:

### 1) Tiling with shared memory

Why doing this:
- Shared memory allows threads in the same block to share data without accessing slower global memory repeatedly. 
- This is especially useful for problems like matrix multiplication or convolution, where neighboring data points are reused.


Key steps:
1. Load a tile (block of data) from global memory into shared memory.
2.	Synchronize all threads within the block using __syncthreads().
3.	Perform computations using shared memory.
4.	Write results back to global memory.


### 2) Strided access

Why doing this:
- When a dataset is larger than the number of available threads, strided access ensures each thread can process multiple elements. 
- Threads work on disjoint subsets of the data. 
- This reduces idle threads and balances the workload.

### 3) Loop Unrolling
Why doing this:
- By unrolling loops manually, you reduce the number of branch instructions and improve pipeline efficiency. 
- This is particularly effective in small, fixed-size loops.

### 4) Cooperative groups
Why doing this:
- CUDA cooperative groups allow explicit synchronization and collaboration between threads and blocks beyond the standard __syncthreads. 
- They are useful for complex algorithms like reductions, where multiple blocks need to coordinate.

### 5) Warp-level intrinsics
Warp-level programming lets threads in the same warp exchange data without shared memory.<br>
Intrinsics like __shfl_sync provide low-latency thread communication.

### 6) Overlapping computations & memory transfers
By leveraging CUDA streams, memory transfers can overlap with kernel execution.<br>
This technique uses asynchronous memory transfers (cudaMemcpyAsync) and stream-based kernels.

### 7) Dynamic parallelism
CUDA dynamic parallelism allows kernels to launch other kernels.<br> 
This is useful for recursive or adaptive algorithms, where the workload is unknown until runtime.

### 8) Thread coarsening
Thread coarsening assigns multiple computations to a single thread to reduce the overhead of thread scheduling.