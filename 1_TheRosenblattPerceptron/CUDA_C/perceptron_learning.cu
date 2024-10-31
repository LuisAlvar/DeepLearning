/*
nvcc perceptron_learning.cu -o ./bin/perceptron_learning.exe
*/
#include <stdio.h>
#include <time.h>

#define N_INPUT_PERCEPTRON 2
const double LEARNING_RATE = 0.1;

/// @brief Implementation of Perceptron Function
/// @param w Is the weight vector
/// @param x Is the input vector
/// @return Return -1 or 1
double compute_output(double* w, double* x, int size)
{
  double z = 0.0;
  double* w_arry_ptr = w;
  double* x_arry_ptr = x;
  for (int index = 0; index < size; ++index)
  {
    z += (*w_arry_ptr) * (*x_arry_ptr); // Compute sum of weighted inputs
    ++w_arry_ptr;
    ++x_arry_ptr;
  }
  // Apply sign function
  if (z < 0)
  {
    return -1.0;
  }
  else {
    return 1.0;
  }
}

/// @brief Shows the current values of the weight array
/// @param w A point to an array of doubles - the weight array
/// @param size The size of the array of doubles
void showLearning(double* w, int size)
{
  for (int i = 0; i < size; ++i)
  {
    printf("w%d = %.2f |", i, w[i]);
    fflush(stdout);
  }
  printf("\n");
}

/// @brief Swaps the two given integers using XOR method
/// @param a Any given integer
/// @param b Any given integer
void Swap(int* a, int* b)
{
  (*a)^=(*b);
  (*b)^=(*a);
  (*a)^=(*b);
}

/// @brief Shuffle the given array of integers
/// @param array A pointer to an array of integers
/// @param size The size of the given array
void arrayShuffle(int* array, int size)
{
  srand(time(NULL));
  for (int index = size - 1; index > 0; --index)
  {
    int rand_within_range_j = rand() % index;
    Swap(&array[index], &array[rand_within_range_j]);
  }
}

// This is just to implement the perceptron function in CUDA C
// ONLY used the CPU. If w
int main(void)
{

  double weights[N_INPUT_PERCEPTRON+1] = {0.2, -0.6, 0.25}; // randomized weight vector
  double x_train[4][N_INPUT_PERCEPTRON+1] = {
    {1.0,-1.0,-1.0},
    {1.0,-1.0,1.0},
    {1.0,1.0,-1.0},
    {1.0,1.0,1.0},
  };
  double y_train[4] = {1.0,1.0,1.0,-1.0};

  double* w_vector_ptr, *input_vector_ptr, *output_vector_ptr;

  w_vector_ptr = weights;
  output_vector_ptr = y_train;
  showLearning(w_vector_ptr, N_INPUT_PERCEPTRON+1);

  // Perceptron Learning Algo
  int index_list[4] = {0, 1, 2, 3};
  bool all_correct = false;
  while (!all_correct)
  {
    all_correct = true;
    arrayShuffle(index_list, sizeof(index_list)/sizeof(int));
    for (size_t i = 0; i < sizeof(index_list)/sizeof(int); ++i)
    {
      input_vector_ptr = x_train[i];
      double y = y_train[i];
      double p_output = compute_output(w_vector_ptr, input_vector_ptr, N_INPUT_PERCEPTRON + 1);
      if (p_output != y)
      {
        for (size_t i = 0; i < sizeof(weights)/sizeof(double); ++i)
        {
          weights[i] += (y * LEARNING_RATE * (*input_vector_ptr));
          ++input_vector_ptr;
        }
        all_correct = false;
        showLearning(w_vector_ptr, N_INPUT_PERCEPTRON + 1);
      }
    } 
  }
  
  return 0;
}