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
