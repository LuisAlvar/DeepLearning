/***************************************************************
 * perceptron - implementaion of a two-input perceptron
 * 
 * Usage
 *      perceptron
 * 
 * Note:
 *  Within this c program file, we carefully set the weight vector 
 *  to [0.9,-0.6,-0.5], with the expectation that it will yield to
 *  NAND gate behavior - given any possible commbination of 2 inputs:
 *  either -1(false) and 1(true).
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
 * >> make
 * 
 * Run:
 * >> ./bin/perceptron
 * 
 ***************************************************************/
#include <stdio.h>

#define N_INPUT_PERCEPTRON 2

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
    printf("%.1f|", (*w_arry_ptr) * (*x_arry_ptr));
    ++w_arry_ptr;
    ++x_arry_ptr;
  }
  printf("%.1f|", z);
  // Apply sign function
  if (z < 0)
  {
    return -1;
  }
  else {
    return 1;
  }
}


int main(void) 
{
  double result;
  /* Using these particular weight values {0.9,-0.6,-0.5} results in an NAND gate */
  double w_vector[N_INPUT_PERCEPTRON+1] = {0.9,-0.6,-0.5};
  // A 2-input vector where the x subnot is the bias term set to 1 - for now.
  double x_vector[N_INPUT_PERCEPTRON+1] = {1.0,-1.0,-1.0};

  result = compute_output(w_vector, x_vector, N_INPUT_PERCEPTRON+1);
  printf("%.1f\n", result);

  x_vector[1] = 1.0;
  x_vector[2] = -1.0; 
  result = compute_output(w_vector, x_vector, N_INPUT_PERCEPTRON+1);
  printf("%.1f\n", result);

  x_vector[1] = -1.0;
  x_vector[2] = 1.0; 
  result = compute_output(w_vector, x_vector, N_INPUT_PERCEPTRON+1);
  printf("%.1f\n", result);

  x_vector[1] = 1.0;
  x_vector[2] = 1.0; 
  result = compute_output(w_vector, x_vector, N_INPUT_PERCEPTRON+1);
  printf("%.1f\n", result);

  return 0;
}