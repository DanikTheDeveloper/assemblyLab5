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

  if ( abs(sum-golden_sum)/golden_sum > 0.00001 ) // more than 0.001% error rate
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
#define L1_TILE 16
#define L2_TILE 64
#define L3_TILE 256

static void gemm_tile(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta) {
    int i, j, k, i1, j1, k1, i2, j2, k2, i3, j3, k3;

    // Zero out C first
    for (i = 0; i < NI; i++)
        for (j = 0; j < NJ; j++)
            C[i*NJ+j] *= beta;

    // L3 tiling
    for (i1 = 0; i1 < NI; i1 += L3_TILE) {
        for (j1 = 0; j1 < NJ; j1 += L3_TILE) {
            for (k1 = 0; k1 < NK; k1 += L3_TILE) {

                // L2 tiling
                for (i2 = i1; i2 < i1 + L3_TILE && i2 < NI; i2 += L2_TILE) {
                    for (j2 = j1; j2 < j1 + L3_TILE && j2 < NJ; j2 += L2_TILE) {
                        for (k2 = k1; k2 < k1 + L3_TILE && k2 < NK; k2 += L2_TILE) {

                            // L1 tiling
                            for (i3 = i2; i3 < i2 + L2_TILE && i3 < NI; i3 += L1_TILE) {
                                for (j3 = j2; j3 < j2 + L2_TILE && j3 < NJ; j3 += L1_TILE) {
                                    for (k3 = k2; k3 < k2 + L2_TILE && k3 < NK; k3 += L1_TILE) {

                                        // The core multiplication loop
                                        for (i = i3; i < i3 + L1_TILE && i < NI; i++) {
                                            for (j = j3; j < j3 + L1_TILE && j < NJ; j++) {
                                                for (k = k3; k < k3 + L1_TILE && k < NK; k++) {
                                                    C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


/* Main computational kernel: with tiling and simd optimizations. */
static void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta) {
    int i, j, k, i1, j1, k1, i2, j2, k2, i3, j3, k3;

    for (i = 0; i < NI; i++)
        for (j = 0; j < NJ; j++)
            C[i*NJ+j] *= beta;

    for (i1 = 0; i1 < NI; i1 += L3_TILE) {
        for (j1 = 0; j1 < NJ; j1 += L3_TILE) {
            for (k1 = 0; k1 < NK; k1 += L3_TILE) {

                for (i2 = i1; i2 < i1 + L3_TILE && i2 < NI; i2 += L2_TILE) {
                    for (j2 = j1; j2 < j1 + L3_TILE && j2 < NJ; j2 += L2_TILE) {
                        for (k2 = k1; k2 < k1 + L3_TILE && k2 < NK; k2 += L2_TILE) {

                            for (i3 = i2; i3 < i2 + L2_TILE && i3 < NI; i3 += L1_TILE) {
                                for (j3 = j2; j3 < j2 + L2_TILE && j3 < NJ; j3 += L1_TILE) {
                                    for (k3 = k2; k3 < k2 + L2_TILE && k3 < NK; k3 += L1_TILE) {

                                        // Vectorized inner loop using AVX2
                                        for (i = i3; i < i3 + L1_TILE && i < NI; i++) {
                                            for (j = j3; j < j3 + L1_TILE && j < NJ; j += 8) { // process 8 elements at once
                                                __m256 sum = _mm256_setzero_ps();

                                                for (k = k3; k < k3 + L1_TILE && k < NK; k++) {
                                                    __m256 a_val = _mm256_broadcast_ss(&A[i*NK+k]);
                                                    __m256 b_val = _mm256_loadu_ps(&B[k*NJ+j]);
                                                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val, b_val));
                                                }

                                                // Multiply with alpha and add to C manually without FMA
                                                __m256 c_val = _mm256_loadu_ps(&C[i*NJ+j]);
                                                __m256 alpha_val = _mm256_set1_ps(alpha);
                                                __m256 res = _mm256_add_ps(_mm256_mul_ps(alpha_val, sum), c_val);
                                                _mm256_storeu_ps(&C[i*NJ+j], res);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static void gemm_tile_simd_par(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta) {
    omp_set_num_threads(256);

    int i, j, k, i1, j1, k1, i2, j2, k2, i3, j3, k3;

    #pragma omp parallel for private(j) collapse(2)
    for (i = 0; i < NI; i++)
        for (j = 0; j < NJ; j++)
            C[i*NJ+j] *= beta;

    __m256 alpha_val = _mm256_set1_ps(alpha);

    #pragma omp parallel for private(j1, k1, i2, j2, k2, i3, j3, k3, i, j, k) collapse(3)
    for (i1 = 0; i1 < NI; i1 += L3_TILE) {
        for (j1 = 0; j1 < NJ; j1 += L3_TILE) {
            for (k1 = 0; k1 < NK; k1 += L3_TILE) {

                for (i2 = i1; i2 < i1 + L3_TILE && i2 < NI; i2 += L2_TILE) {
                    for (j2 = j1; j2 < j1 + L3_TILE && j2 < NJ; j2 += L2_TILE) {
                        for (k2 = k1; k2 < k1 + L3_TILE && k2 < NK; k2 += L2_TILE) {

                            for (i3 = i2; i3 < i2 + L2_TILE && i3 < NI; i3 += L1_TILE) {
                                for (j3 = j2; j3 < j2 + L2_TILE && j3 < NJ; j3 += L1_TILE) {
                                    for (k3 = k2; k3 < k2 + L2_TILE && k3 < NK; k3 += L1_TILE) {

                                        for (i = i3; i < i3 + L1_TILE && i < NI; i++) {
                                            for (j = j3; j < j3 + L1_TILE && j < NJ; j += 8) {
                                                __m256 sum = _mm256_setzero_ps();

                                                // Unrolling the k loop by a factor of 4
                                                for (k = k3; k < k3 + L1_TILE - 4 && k < NK; k+=4) {
                                                    __m256 a_val1 = _mm256_broadcast_ss(&A[i*NK+k]);
                                                    __m256 a_val2 = _mm256_broadcast_ss(&A[i*NK+k+1]);
                                                    __m256 a_val3 = _mm256_broadcast_ss(&A[i*NK+k+2]);
                                                    __m256 a_val4 = _mm256_broadcast_ss(&A[i*NK+k+3]);

                                                    __m256 b_val1 = _mm256_loadu_ps(&B[k*NJ+j]);
                                                    __m256 b_val2 = _mm256_loadu_ps(&B[(k+1)*NJ+j]);
                                                    __m256 b_val3 = _mm256_loadu_ps(&B[(k+2)*NJ+j]);
                                                    __m256 b_val4 = _mm256_loadu_ps(&B[(k+3)*NJ+j]);

                                                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val1, b_val1));
                                                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val2, b_val2));
                                                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val3, b_val3));
                                                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val4, b_val4));
                                                }

                                                // Handle remaining iterations
                                                for (; k < k3 + L1_TILE && k < NK; k++) {
                                                    __m256 a_val = _mm256_broadcast_ss(&A[i*NK+k]);
                                                    __m256 b_val = _mm256_loadu_ps(&B[k*NJ+j]);
                                                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val, b_val));
                                                }

                                                __m256 c_val = _mm256_loadu_ps(&C[i*NJ+j]);
                                                __m256 res = _mm256_add_ps(_mm256_mul_ps(alpha_val, sum), c_val);
                                                _mm256_storeu_ps(&C[i*NJ+j], res);
                                            }
                                        }
                                    }
                                }
                            }
                        }
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
