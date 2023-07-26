#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"
#include <x86intrin.h>
#include <omp.h>

#define BLOCK_SIZE 64

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
void gemm_base(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta) {
    int i, j, k, l, m, n;

    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);

    for (i = 0; i < NI; i += BLOCK_SIZE) {
        for (j = 0; j < NJ; j += BLOCK_SIZE) {
            for (k = 0; k < NK; k += BLOCK_SIZE) {

                // Inner loops for the tiles
                for (l = i; l < i + BLOCK_SIZE && l < NI; ++l) {
                    for (m = j; m < j + BLOCK_SIZE && m < NJ; m+=8) { // Increment by 8 for AVX2

                        __m256 c_val;
                        if (k == 0) {
                            c_val = _mm256_loadu_ps(&C[l*NJ + m]);
                            c_val = _mm256_mul_ps(c_val, beta_vec);
                        } else {
                            c_val = _mm256_loadu_ps(&C[l*NJ + m]);
                        }

                        // Now perform the dot product for this specific element of C
                        for (n = k; n < k + BLOCK_SIZE && n < NK; ++n) {
                            __m256 a_val = _mm256_set1_ps(A[l*NK + n]);
                            __m256 b_val = _mm256_loadu_ps(&B[n*NJ + m]);
                            
                            __m256 prod = _mm256_mul_ps(a_val, b_val);
                            prod = _mm256_mul_ps(prod, alpha_vec);
                            c_val = _mm256_add_ps(c_val, prod);
                        }

                        _mm256_storeu_ps(&C[l*NJ + m], c_val);
                    }
                }
            }
        }
    }
}

/* Main computational kernel: with tiling optimization. */
static void gemm_tile(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
    int i, j, k, i_blk, j_blk, k_blk;

    // Loop over the blocks of matrices A, B, and C using tiling
    for (i_blk = 0; i_blk < NI; i_blk += BLOCK_SIZE) {
        for (j_blk = 0; j_blk < NJ; j_blk += BLOCK_SIZE) {
            for (k_blk = 0; k_blk < NK; k_blk += BLOCK_SIZE) {
                // Process a block of matrix C
                for (i = i_blk; i < i_blk + BLOCK_SIZE && i < NI; i++) {
                    for (j = j_blk; j < j_blk + BLOCK_SIZE && j < NJ; j++) {
                        // Scale the current element of C by beta
                        C[i * NJ + j] *= beta;

                        // Compute the matrix multiplication for the current element of C
                        for (k = k_blk; k < k_blk + BLOCK_SIZE && k < NK; k++) {
                            C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
                        }
                    }
                }
            }
        }
    }
}


/* Main computational kernel: with tiling and simd optimizations. */
static
void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j += 8) {  // Process 8 elements at once
            __m256 c_vec = _mm256_loadu_ps(&C[i * NJ + j]);
            c_vec = _mm256_mul_ps(c_vec, beta_vec);
            for (k = 0; k < NK; ++k) {
                __m256 a_vec = _mm256_set1_ps(A[i * NK + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * NJ + j]);
                c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, _mm256_mul_ps(b_vec, alpha_vec)));
            }
            _mm256_storeu_ps(&C[i * NJ + j], c_vec);
        }
    }
}

/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static
void gemm_tile_simd_par(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);

    // #pragma omp parallel for
    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j += 8) {
            __m256 c_vec = _mm256_loadu_ps(&C[i * NJ + j]);
            c_vec = _mm256_mul_ps(c_vec, beta_vec);
            for (k = 0; k < NK; ++k) {
                __m256 a_vec = _mm256_set1_ps(A[i * NK + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * NJ + j]);
                c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, _mm256_mul_ps(b_vec, alpha_vec)));
            }
            _mm256_storeu_ps(&C[i * NJ + j], c_vec);
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
