#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"
#include <x86intrin.h>
#include <omp.h>

#define NI 4096
#define NJ 4096
#define NK 4096

/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_and_valid_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  float golden_sum = 27789682688.000000;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

  if ( abs(sum-golden_sum)/sum > 0.00001 ) // more than 0.001% error rate
    printf("Incorrect sum of C array. Expected sum: %f, your sum: %f\n", golden_sum, sum);
  else
    printf("Correct result. Sum of C array = %f\n", sum);
}


/* Main computational kernel: baseline. The whole function will be timed,
   including the call and return. DO NOT change the baseline.*/
static
void gemm_base(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
    for (j = 0; j < NJ; j++) {
      for (k = 0; k < NK; ++k) {
	C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}

/* Main computational kernel: with tiling optimization. */
#define TILE_SIZE 64

void gemm_tile(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
    int i, j, k, i1, j1, k1;

    for (i = 0; i < NI; i+=TILE_SIZE) 
    {
        for (j = 0; j < NJ; j+=TILE_SIZE) 
        {
            for (k = 0; k < NK; k+=TILE_SIZE) 
            {
                for (i1 = i; i1 < i + TILE_SIZE && i1 < NI; ++i1)
                {
                    for (j1 = j; j1 < j + TILE_SIZE && j1 < NJ; ++j1)
                    {
                        float temp = C[i1*NJ+j1] * beta;
                        for (k1 = k; k1 < k + TILE_SIZE && k1 < NK; ++k1)
                        {
                            temp += alpha * A[i1*NK+k1] * B[k1*NJ+j1];
                        }
                        C[i1*NJ+j1] = temp;
                    }
                }
            }
        }
    }
}


/* Main computational kernel: with tiling and simd optimizations. */
void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
    int i, j, k, i1, j1, k1;

    for (i = 0; i < NI; i+=TILE_SIZE) 
    {
        for (j = 0; j < NJ; j+=TILE_SIZE) 
        {
            for (k = 0; k < NK; k+=TILE_SIZE) 
            {
                for (i1 = i; i1 < i + TILE_SIZE && i1 < NI; ++i1)
                {
                    for (j1 = j; j1 < j + TILE_SIZE && j1 < NJ; ++j1)
                    {
                        __m256 vA = _mm256_loadu_ps(&A[i1*NK+k1]);
                        __m256 vB = _mm256_loadu_ps(&B[k1*NJ+j1]);
                        __m256 vC = _mm256_loadu_ps(&C[i1*NJ+j1]);

                        vC = _mm256_add_ps(_mm256_mul_ps(vA, vB), vC);

                        _mm256_storeu_ps(&C[i1*NJ+j1], vC);
                    }
                }
            }
        }
    }
}


/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
void gemm_tile_simd_par(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NI; i+=TILE_SIZE) 
    {
        for (int j = 0; j < NJ; j+=TILE_SIZE) 
        {
            for (int k = 0; k < NK; k+=TILE_SIZE) 
            {
                for (int i1 = i; i1 < i + TILE_SIZE && i1 < NI; ++i1)
                {
                    for (int j1 = j; j1 < j + TILE_SIZE && j1 < NJ; ++j1)
                    {
                        __m256 vA = _mm256_loadu_ps(&A[i1*NK+k]);
                        __m256 vB = _mm256_loadu_ps(&B[k*NJ+j1]);
                        __m256 vC = _mm256_loadu_ps(&C[i1*NJ+j1]);

                        vC = _mm256_add_ps(_mm256_mul_ps(vA, vB), vC);

                        _mm256_storeu_ps(&C[i1*NJ+j1], vC);
                    }
                }
            }
        }
    }
}


int main(int argc, char** argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  /* opt selects which gemm version to run */
  int opt = 0;
  if(argc == 2) {
    opt = atoi(argv[1]);
  }
  //printf("option: %d\n", opt);
  
  /* Initialize array(s). */
  init_array (C, A, B);

  /* Start timer. */
  timespec timer = tic();

  switch(opt) {
  case 0: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
    break;
  case 1: // tiling
    /* Run kernel. */
    gemm_tile (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling time");
    break;
  case 2: // tiling and simd
    /* Run kernel. */
    gemm_tile_simd (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd time");
    break;
  case 3: // tiling, simd, and parallelization
    /* Run kernel. */
    gemm_tile_simd_par (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd-par time");
    break;
  default: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
  }
  /* Print results. */
  print_and_valid_array_sum(C);

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}
