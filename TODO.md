## Optimization steps:

### 1) Tiling with shared memory

#### Why doing this:
- Shared memory allows threads in the same block to share data without accessing slower global memory repeatedly. 
- This is especially useful for problems like matrix multiplication or convolution, where neighboring data points are reused.

#### Key steps:
1. Load a tile (block of data) from global memory into shared memory.
```c++
// 1. declaration
__shared__ float tile_A[TILE_SIZE][TILE_SIZE]; 
__shared__ float tile_B[TILE_SIZE][TILE_SIZE];

// 2. cumpute the indexes
int row = threadIdx.y + blockIdx.y * TILE_SIZE;
int col = threadIdx.x + blockIdx.x * TILE_SIZE;

// 3. fill-up
if(row < N && (i * TILE_SIZE + threadIdx.x) < N){
    tile_A[threadIdx.y][threadIdx.x] = A[row * N + i * TILE_SIZE + threadIdx.x]; }
else {
    tile_A[threadIdx.y][threadIdx.x] = 0.0f;
}
```
2.	Synchronize all threads within the block using __syncthreads().
3.	Perform computations using shared memory.
```c++
// Example of multiplication
for(int j = 0; j < TILE_SIZE; j++){    
 val += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x]; 
}
```
4.	Write results back to global memory.

#### Known bugs:
- for input matrixes that are too large (_o.o.m.: (10^5)+_) the kernel goes "memory access error" since it does not naturally fit into the shared memory.
- <ins>Why</ins>:
  - it all comes up to HW constraints, defined as follows: (_wrt a Nvidia Tesla T4_) you get the HW constraints:
    - max block size 1024 x 1024 x 64;
    - max grid size 2.147.483.647, 65535, 65535;
    - max shared memory size: 49152;
  - let's say for instance your load is a 100k x 100k matrix, you'll have:
    - block size: 16 x 16 x 1 (_since a 2D configuration_);
    - grid size: 6250 x 6250 x 1;
    - shared memory size: 2304;
  - issue breakdown:
    - total threads (_for x-axis_) = 6250 * 16 (= 100k) that is << the HW bound (2.147.483.647) => __OK__;
    - total threads (_for y-axis_) = 6250 * 16 (= 100k) that is >> the HW bound (65535) => __NOT OK__;
  - basically what happens is that you're correctly sizing the grid in order to fit the input, but it's too high for the HW of the device.
- <ins>Possible fix</ins>:
  1. Tiling:
     - basically you split the input into tiles in order to fit all the input into the shared memory.
     - although this way you can overcome the y-axis out-of-bound problem, there'll be a single kernel launch (_differently from cuda streams_). 
  2. CUDA Streams:
     - you split the input into chunks, each one is assigned to a stream that will launch a kernel.
     - all the kernels created by the splitting of the input will be executed in parallel.
     - doesn't solve grid-sizing by itself, why?
       - even though each single kernel will have its own grid and block sizes, at the exact moment you start them all together in parallel you're gonna end up in the exact same situation as before.
       - their grid size sum is _de facto_ equal to the grid size of the whole input, so the HW grid bound will be broken THEORETICALLY.
       - WHAT is going to happen in practice is the GPU understanding the imminent bounds-break will reschedule the streams in order not to break the bounds.
       - SO the only time the error would happen is when each stream has ITS grid size larger than the bounds.
     - you would need some scheduling mechanism in order to take advantage of streams.
     - moreover:
       - chunks need their memory space in order not to overlap.
       - you have to take care of not launching too many streams at once so not to saturate other GPU resources.
  3. Grid-striding loops:

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