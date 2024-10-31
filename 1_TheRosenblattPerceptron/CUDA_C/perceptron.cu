/*
nvcc perceptron.cu -o ./bin/perceptron.exe
*/
#include <stdio.h>

static void ErrorHandler(cudaError_t err, const char* file, int line)
{
  if (err != cudaSuccess)
  {
    printf("file(%s) at line %d: %s", file, line, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#define ERROR_HANDLER( err ) (ErrorHandler(err, __FILE__, __LINE__))
#define N_INPUT 3

__global__ void DotProduct(double*a, double*b, double*c)
{
  int id = blockIdx.x;
  if (id < N_INPUT)
  {
    c[id] = a[id] * b[id];
  }
}

// Not the best utilization of having the GPU perform a dot product of this size.
// This is just to implement the perceptron function in CUDA C
int main(void)
{
  double* weight_vector;
  double* input_vector;
  double* result_vector;

  double* dev_w, *dev_x, *dev_result;

  // allocate memory on the CPU side 
  weight_vector = (double*)malloc(N_INPUT * sizeof(double));
  input_vector = (double*)malloc(N_INPUT * sizeof(double));
  result_vector = (double*)malloc(N_INPUT * sizeof(double));

  weight_vector[0] = 0.9;
  weight_vector[1] = -0.6;
  weight_vector[2] = -0.5;

  input_vector[0] = 1.0;
  input_vector[1] = 1.0;
  input_vector[2] = 1.0;

  result_vector[0] = 0.0;
  result_vector[1] = 0.0;
  result_vector[2] = 0.0;


  // allocate the memory on the GPU
  ERROR_HANDLER(cudaMalloc((void**)&dev_w, N_INPUT*sizeof(double)));
  ERROR_HANDLER(cudaMalloc((void**)&dev_x, N_INPUT*sizeof(double)));
  ERROR_HANDLER(cudaMalloc((void**)&dev_result, N_INPUT*sizeof(double)));

  // copy data from CPU to GPU
  ERROR_HANDLER(cudaMemcpy(dev_w, weight_vector, N_INPUT * sizeof(double), cudaMemcpyHostToDevice));
  ERROR_HANDLER(cudaMemcpy(dev_x, input_vector, N_INPUT * sizeof(double), cudaMemcpyHostToDevice));

  DotProduct<<<N_INPUT,1>>>(dev_w, dev_x, dev_result);

  ERROR_HANDLER(cudaMemcpy(result_vector, dev_result, N_INPUT * sizeof(double), cudaMemcpyDeviceToHost));

  double z = 0.0;

  for (int i = 0; i < N_INPUT; i++)
  {
    printf("%.2f| ", result_vector[i]);
    z += result_vector[i];
  }
  printf("z: %.2f\n", z);

  if (z < 0)
  {
    printf("%d\n", -1);
  }
  else 
  {
    printf("%d\n", 1);
  }
  
  return 0;
}