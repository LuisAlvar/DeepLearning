/***************************************************************
 * perceptron_learning - implementaion of a two-input perceptron learning algorithm
 * 
 * Usage
 *      perceptron_learning
 * 
 * Note:
 *  Within this c program file, we attempt to find the weight vector 
 *  that will yield to the NAND gate behavior. We start by always setting 
 *  the initial weight vector with random values. Next, we apply the 
 *  perceptron learning alforithm to find a weight vector that works.
 * 
 *  NAND gate
 *  x1 | x2 | output
 *  ---|----|-------|
 *  -1 | -1 |   1
 *  1  | -1 |   1
 *  -1 | 1  |   1
 *  1  | 1  |   -1
 * 
 * Compile:
 * >> gcc -g -D__USE_FIXED_PROTOTYPES__ -ansi -std=c11 -o ./bin/perceptron_learning perceptron_learning.c -lm
 * 
 * Run:
 * >> ./bin/perceptron_learning
 ***************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define N_INPUT_PERCEPTRON 2
const double LEARNING_RATE = 0.1;

/// @brief Implementation of Perceptron Function
/// @param w Is the weight vector
/// @param x Is the input vector
/// @return Return -1 or 1
int compute_output(double* w, double* x, int size)
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
    return -1;
  }
  else {
    return 1;
  }
}

/// @brief Shows the current values of the weight array
/// @param w A point to an array of doubles - the weight array
/// @param size The size of the array of doubles
void show_learning(double* w, int size)
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
void swap(int* a, int* b)
{
  (*a)^=(*b);
  (*b)^=(*a);
  (*a)^=(*b);
}

/// @brief Shuffle the given array of integers
/// @param array A pointer to an array of integers
/// @param size The size of the given array
void array_shuffle(int* array, int size)
{
  srand(time(NULL));
  for (int index = size - 1; index > 0; --index)
  {
    int rand_within_range_j = rand() % index;
    swap(&array[index], &array[rand_within_range_j]);
  }
}

int main(void) 
{
  int index_list[4] = {0, 1, 2, 3};
  double x_train[4][3] = {
    {1.0,-1.0,-1.0},
    {1.0,-1.0,1.0},
    {1.0,1.0,-1.0},
    {1.0,1.0,1.0},
  };
  int y_train[4] = {1.0, 1.0, 1.0, -1.0};
  double weights[N_INPUT_PERCEPTRON+1] = {0.2,-0.6,0.25};
  show_learning(weights, sizeof(weights)/sizeof(double));

  // Perceptron training loop:
  bool all_correct = false;
  while (!all_correct)
  {
    all_correct = true;
    array_shuffle(index_list, sizeof(index_list)/sizeof(int));
    for (int i = 0; i < sizeof(index_list)/sizeof(int); ++i)
    {
      double* x = x_train[i];
      int y = y_train[i];
      int p_output =  compute_output(weights, x, N_INPUT_PERCEPTRON+1);
      if (p_output != y)
      {
        for (int j = 0; j < sizeof(weights)/sizeof(double); ++j)
        {
          weights[j] += (y * LEARNING_RATE * (*x));
          ++x;
        }
        all_correct = false;
        show_learning(weights, sizeof(weights)/sizeof(double));
      }
    }  
  }
  
  return 0;
}