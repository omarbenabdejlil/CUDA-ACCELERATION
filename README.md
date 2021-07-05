# tuto-cuda

# ACCELERATE APPLICATIONS WITH CUDA 

|NAME |LAST NAME|CLASS
|---|---|---|
|BENABDEJLIL|OMAR|CCDAD.1

#

## Here is my file structure : 
#
<a href="https://imgbb.com/"><img src="https://i.ibb.co/KjscyQ8/acceleration.png" alt="acceleration" border="0"></a>

` The first program is created to check the if the element is doubled or not , here is the code : `

```c
#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElements(int *a, int N)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  /*
   * The previous code (now commented out) attempted
   * to access an element outside the range of `a`.
   */

  // for (int i = idx; i < N + stride; i += stride)
  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  /*
   * The previous code (now commented out) attempted to launch
   * the kernel with more than the maximum number of threads per
   * block, which is 1024.
   */

  size_t threads_per_block = 1024;
  /* size_t threads_per_block = 2048; */
  size_t number_of_blocks = 32;

  cudaError_t syncErr, asyncErr;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);

  /*
   * Catch errors for both the kernel launch above and any
   * errors that occur during the asynchronous `doubleElements`
   * kernel execution.
   */

  syncErr = cudaGetLastError();
  asyncErr = cudaDeviceSynchronize();

  /*
   * Print errors should they exist.
   */

  if (syncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(syncErr));
  if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
```
- __Runnin code from my machine :__
 
<a href="https://ibb.co/3RK5SWj"><img src="https://i.ibb.co/vX9CLmT/1.png" alt="1" border="0"></a>

` exemple of basic parallel solution  5 grid per 5 threds for each one: `

```c
#include <stdio.h>

__global__ void firstParallel()
{
  printf("This is running in parallel.\n");
}

int main()
{
  firstParallel<<<5, 5>>>();
  cudaDeviceSynchronize();
}
```
- __running code from my host :__ 

  <a href="https://ibb.co/qFhQmX5"><img src="https://i.ibb.co/G3GSCBT/2.png" alt="2" border="0"></a>

`exemple of Heat confuction solution :`
```c
#include <stdio.h>
#include <math.h>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

__global__
void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  // loop over all points in domain (except boundary)
  if (j > 0 && i > 0 && j < nj-1 && i < ni-1) {
    // find indices into linear memory
    // for central point and neighbours
    i00 = I2D(ni, i, j);
    im10 = I2D(ni, i-1, j);
    ip10 = I2D(ni, i+1, j);
    i0m1 = I2D(ni, i, j-1);
    i0p1 = I2D(ni, i, j+1);

    // evaluate derivatives
    d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
    d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

    // update temperatures
    temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
  }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;


  // loop over all points in domain (except boundary)
  for ( int j=1; j < nj-1; j++ ) {
    for ( int i=1; i < ni-1; i++ ) {
      // find indices into linear memory
      // for central point and neighbours
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
  }
}

int main()
{
  int istep;
  int nstep = 200; // number of time steps

  // Specify our 2D dimensions
  const int ni = 200;
  const int nj = 100;
  float tfac = 8.418e-5; // thermal diffusivity of silver

  float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;

  const int size = ni * nj * sizeof(float);

  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);
  cudaMallocManaged(&temp1, size);
  cudaMallocManaged(&temp2, size);

  // Initialize with random data
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }

  // Execute the CPU-only reference version
  for (istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    // swap the temperature pointers
    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp;
  }

  dim3 tblocks(32, 16, 1);
  dim3 grid((nj/tblocks.x)+1, (ni/tblocks.y)+1, 1);
  cudaError_t ierrSync, ierrAsync;

  // Execute the modified version using same data
  for (istep=0; istep < nstep; istep++) {
    step_kernel_mod<<< grid, tblocks >>>(ni, nj, tfac, temp1, temp2);

    ierrSync = cudaGetLastError();
    ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
    if (ierrSync != cudaSuccess) { printf("Sync error: %s\n", cudaGetErrorString(ierrSync)); }
    if (ierrAsync != cudaSuccess) { printf("Async error: %s\n", cudaGetErrorString(ierrAsync)); }

    // swap the temperature pointers
    temp_tmp = temp1;
    temp1 = temp2;
    temp2= temp_tmp;
  }

  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1[i]-temp1_ref[i]); }
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  free( temp1_ref );
  free( temp2_ref );
  cudaFree( temp1 );
  cudaFree( temp2 );

  return 0;
}
```
- __running code from my host :__

<a href="https://ibb.co/1fBTrG5"><img src="https://i.ibb.co/C54QbKZ/3.png" alt="3" border="0"></a>

`Exemple of hello gpu solution : `

```c
#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}


__global__ void helloGPU()
{
  printf("Hello from the GPU.\n");
}

int main()
{
  helloCPU();


  /*
   * Add an execution configuration with the <<<...>>> syntax
   * will launch this function as a kernel on the GPU.
   */

  helloGPU<<<1, 1>>>();

  /*
   * `cudaDeviceSynchronize` will block the CPU stream until
   * all GPU kernels have completed.
   */

  cudaDeviceSynchronize();
}
```
- __running code from my host__ :

<a href="https://ibb.co/hsNKHvf"><img src="https://i.ibb.co/H7yrYQd/4.png" alt="4" border="0"></a>

`Generating multiple 2d matrix : `
```c
#include <stdio.h>

#define N  64

__global__ void matrixMulGPU( int * a, int * b, int * c )
{
  int val = 0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < N && col < N)
  {
    for ( int k = 0; k < N; ++k )
      val += a[row * N + k] * b[k * N + col];
    c[row * N + col] = val;
  }
}

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu;

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  dim3 threads_per_block (16, 16, 1); // A 16 x 16 block threads
  dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

  cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

  // Call the CPU version to check our work
  matrixMulCPU( a, b, c_cpu );

  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}
```
- __running code from my host :__

<a href="https://ibb.co/zS8gWCd"><img src="https://i.ibb.co/7GgDxBh/5.png" alt="5" border="0"></a>

`Single block loop test : `
```c
#include <stdio.h>


/*
 * Notice the absence of the previously expected argument `N`.
 */

__global__ void loop()
{
  /*
   * This kernel does the work of only 1 iteration
   * of the original for loop. Indication of which
   * "iteration" is being executed by this kernel is
   * still available via `threadIdx.x`.
   */

  printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
  /*
   * It is the execution context that sets how many "iterations"
   * of the "loop" will be done.
   */

  loop<<<1, 10>>>();
  cudaDeviceSynchronize();
}
```
- __running code from my host :__ 

<a href="https://ibb.co/7yjKk1L"><img src="https://i.ibb.co/127Tzmj/6.png" alt="6" border="0"></a>

` Checking win condition between thread per bloc : `
```c
#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  }
}

int main()
{
  /*
   * This is one possible execution context that will make
   * the kernel launch print its success message.
   */

  printSuccessForCorrectExecutionConfiguration<<<256, 1024>>>();

  /*
   * Don't forget kernel execution is asynchronous and you must
   * sync on its completion.
   */

  cudaDeviceSynchronize();
}
```

- __running code from my host :__

<a href="https://ibb.co/VjVVrwG"><img src="https://i.ibb.co/2nqqmZr/7.png" alt="7" border="0"></a>

`Adding vector exemple (simple solution) :`
```c
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  checkCuda( cudaMallocManaged(&a, size) );
  checkCuda( cudaMallocManaged(&b, size) );
  checkCuda( cudaMallocManaged(&c, size) );

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );

  checkElementsAre(7, c, N);

  checkCuda( cudaFree(a) );
  checkCuda( cudaFree(b) );
  checkCuda( cudaFree(c) );
}
```

`Mis-matched loop solution : `
```c
#include <stdio.h>


__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
  {
    a[i] = initialValue;
  }
}

int main()
{
  /*
   * Do not modify `N`.
   */

  int N = 1000;

  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);

  /*
   * Assume we have reason to want the number of threads
   * fixed at `256`: do not modify `threads_per_block`.
   */

  size_t threads_per_block = 256;

  /*
   * The following is idiomatic CUDA to make sure there are at
   * least as many threads in the grid as there are `N` elements.
   */

  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  int initialValue = 6;

  initializeElementsTo<<<number_of_blocks, threads_per_block>>>(initialValue, a, N);
  cudaDeviceSynchronize();

  /*
   * Check to make sure all values in `a`, were initialized.
   */

  for (int i = 0; i < N; ++i)
  {
    if(a[i] != initialValue)
    {
      printf("FAILURE: target value: %d\t a[%d]: %d\n", initialValue, i, a[i]);
      exit(1);
    }
  }
  printf("SUCCESS!\n");

  cudaFree(a);
}
```
- __running code from my host :__ 

<a href="https://ibb.co/XfPsptB"><img src="https://i.ibb.co/j9YMGDK/8.png" alt="8" border="0"></a>

`Grid stride with double solution : `
```c
#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElements(int *a, int N)
{

  /*
   * Use a grid-stride loop so each thread does work
   * on more than one element in the array.
   */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  size_t threads_per_block = 256;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
```
- __running code from my host :__ 

<a href="https://ibb.co/ZSdrVpm"><img src="https://i.ibb.co/XDXPbr8/9.png" alt="9" border="0"></a>

#
