
/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_sgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab sgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */




#include <stdio.h>

#include "bl_sgemm.h"
#include "bl_config.h"
#include "omp.h"
#define __AXV2
inline void packA_mcxkc(
        int    m,
        int    k,
        float *XA,
        int    ldXA,
        float *packA,
        int ldPackA,
        int offset
        )
{
    int    i, p;

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            *(packA + p * ldPackA + offset + i) = *(XA + p * ldXA + i);
        }
    }
}

inline void packA_mcxkc_origin(
        int    m,
        int    k,
        float *XA,
        int    ldXA,
        float *packA
        )
{
    int    i, p;

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            *packA ++ = *(XA + p * ldXA + i);
        }
    }
}

/*
 * --------------------------------------------------------------------------
 */

inline void packB_kcxnc(
        int    n,
        int    k,
        float *XB,
        int    ldXB,
        float *packB
        )
{
    int    j, p;

    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            *packB ++ = *(XB + j * ldXB + p);
        }
    }
}




#if(defined(__AVX2))
#include <immintrin.h> // AVX2 header
int sgemm_mat_opt_macc_avx(int bm, int bn, int bk, float* ba, float* bb, float* c, int ldc)
{
    float *pInA = ba;              // Input data matrix pointer A
    float *pInB = bb;              // Input data matrix pointer B
    float *px = c;                // Temporary output data matrix pointer
    int ii, jj, kk;

    __m256 va0m8, vres0m8, vres1m8, vres2m8, vres3m8; // 定义256的向量寄存器
    
    // 先计算数据满足256位的情况，即8个32位浮点数
    for (jj = bn / 4; jj > 0; jj--) {
        px = c;
        pInA = ba;

        // 主循环：处理对齐的8个浮点数块
        for (ii = bm / 8 * 8; ii > 0; ii -= 8) {
            pInB = bb;

            vres0m8 = _mm256_set1_ps(0.0f);
            vres1m8 = _mm256_set1_ps(0.0f);
            vres2m8 = _mm256_set1_ps(0.0f);
            vres3m8 = _mm256_set1_ps(0.0f);

            for (kk = bk; kk > 0; kk--) {
                va0m8 = _mm256_loadu_ps(pInA);  // 对齐加载

                vres0m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 0)), va0m8, vres0m8);
                vres1m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + bk)), va0m8, vres1m8);
                vres2m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 2 * bk)), va0m8, vres2m8);
                vres3m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 3 * bk)), va0m8, vres3m8);
                pInA += bm;
                pInB += 1;
            }

            __m256 tmp = _mm256_loadu_ps(px);
            tmp = _mm256_add_ps(tmp, vres0m8);
            _mm256_storeu_ps(px, tmp);

            tmp = _mm256_loadu_ps(px + ldc);
            tmp = _mm256_add_ps(tmp, vres1m8);
            _mm256_storeu_ps(px + ldc, tmp);

            tmp = _mm256_loadu_ps(px + 2 * ldc);
            tmp = _mm256_add_ps(tmp, vres2m8);
            _mm256_storeu_ps(px + 2 * ldc, tmp);

            tmp = _mm256_loadu_ps(px + 3 * ldc);
            tmp = _mm256_add_ps(tmp, vres3m8);
            _mm256_storeu_ps(px + 3 * ldc, tmp);

            px += 8;
            pInA = ba + (bm/8*8) - ii + 8; // 移动到下8个A数据
        }

        // 处理剩余不足8个浮点数的数据
        int remainder = bm % 8;
        if (remainder > 0) {
            __m256i mask = _mm256_set_epi32(
                remainder > 7 ? -1 : 0,
                remainder > 6 ? -1 : 0,
                remainder > 5 ? -1 : 0,
                remainder > 4 ? -1 : 0,
                remainder > 3 ? -1 : 0,
                remainder > 2 ? -1 : 0,
                remainder > 1 ? -1 : 0,
                remainder > 0 ? -1 : 0
            );

            pInA = ba + bm - remainder;
            pInB = bb;

            vres0m8 = _mm256_set1_ps(0.0f);
            vres1m8 = _mm256_set1_ps(0.0f);
            vres2m8 = _mm256_set1_ps(0.0f);
            vres3m8 = _mm256_set1_ps(0.0f);

            for (kk = bk; kk > 0; kk--) {
                va0m8 = _mm256_maskload_ps(pInA, mask);

                vres0m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 0)), va0m8, vres0m8);
                vres1m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + bk)), va0m8, vres1m8);
                vres2m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 2 * bk)), va0m8, vres2m8);
                vres3m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 3 * bk)), va0m8, vres3m8);
                pInA += bm;
                pInB += 1;
            }

            __m256 tmp = _mm256_maskload_ps(px, mask);
            tmp = _mm256_add_ps(tmp, vres0m8);
            _mm256_maskstore_ps(px, mask, tmp);

            tmp = _mm256_maskload_ps(px + ldc, mask);
            tmp = _mm256_add_ps(tmp, vres1m8);
            _mm256_maskstore_ps(px + ldc, mask, tmp);

            tmp = _mm256_maskload_ps(px + 2 * ldc, mask);
            tmp = _mm256_add_ps(tmp, vres2m8);
            _mm256_maskstore_ps(px + 2 * ldc, mask, tmp);

            tmp = _mm256_maskload_ps(px + 3 * ldc, mask);
            tmp = _mm256_add_ps(tmp, vres3m8);
            _mm256_maskstore_ps(px + 3 * ldc, mask, tmp);
        }

        bb += (bk << 2);
        c += (ldc << 2);
    }


    // ch = 2, mul = 8
    bn = bn & 3;
    for (jj = bn / 2; jj > 0; jj--) {
        px = c;
        pInA = ba;

        // 主循环：处理对齐的8个浮点数块
        for (ii = bm / 8 * 8; ii > 0; ii -= 8) {
            pInB = bb;

            vres0m8 = _mm256_set1_ps(0.0f);
            vres1m8 = _mm256_set1_ps(0.0f);

            for (kk = bk; kk > 0; kk--) {
                va0m8 = _mm256_loadu_ps(pInA);  // 对齐加载

                vres0m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 0)), va0m8, vres0m8);
                vres1m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + bk)), va0m8, vres1m8);
                pInA += bm;
                pInB += 1;
            }

            __m256 tmp = _mm256_loadu_ps(px);
            tmp = _mm256_add_ps(tmp, vres0m8);
            _mm256_storeu_ps(px, tmp);

            tmp = _mm256_loadu_ps(px + ldc);
            tmp = _mm256_add_ps(tmp, vres1m8);
            _mm256_storeu_ps(px + ldc, tmp);

            px += 8;
            pInA = ba + (bm/8*8) - ii + 8; // 移动到下8个A数据
        }

         // 处理剩余不足8个浮点数的数据
        int remainder = bm % 8;
        if (remainder > 0) {
            __m256i mask = _mm256_set_epi32(
                remainder > 7 ? -1 : 0,
                remainder > 6 ? -1 : 0,
                remainder > 5 ? -1 : 0,
                remainder > 4 ? -1 : 0,
                remainder > 3 ? -1 : 0,
                remainder > 2 ? -1 : 0,
                remainder > 1 ? -1 : 0,
                remainder > 0 ? -1 : 0
            );

            pInA = ba + bm - remainder;
            pInB = bb;

            vres0m8 = _mm256_set1_ps(0.0f);
            vres1m8 = _mm256_set1_ps(0.0f);

            for (kk = bk; kk > 0; kk--) {
                va0m8 = _mm256_maskload_ps(pInA, mask);

                vres0m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 0)), va0m8, vres0m8);
                vres1m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + bk)), va0m8, vres1m8);
                pInA += bm;
                pInB += 1;
            }

            __m256 tmp = _mm256_maskload_ps(px, mask);
            tmp = _mm256_add_ps(tmp, vres0m8);
            _mm256_maskstore_ps(px, mask, tmp);

            tmp = _mm256_maskload_ps(px + ldc, mask);
            tmp = _mm256_add_ps(tmp, vres1m8);
            _mm256_maskstore_ps(px + ldc, mask, tmp);
        }

        bb += (bk << 1);
        c += (ldc << 1);
    }

    // ch = 1, mul = 8
    bn = bn & 1;

    for (jj = bn; jj > 0; jj--) {
        px = c;
        pInA = ba;

        // 主循环：处理对齐的8个浮点数块
        for (ii = bm / 8 * 8; ii > 0; ii -= 8) {
            pInB = bb;

            vres0m8 = _mm256_set1_ps(0.0f);

            for (kk = bk; kk > 0; kk--) {
                va0m8 = _mm256_loadu_ps(pInA);  // 对齐加载

                vres0m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 0)), va0m8, vres0m8);
                pInA += bm;
                pInB += 1;
            }

            __m256 tmp = _mm256_loadu_ps(px);
            tmp = _mm256_add_ps(tmp, vres0m8);
            _mm256_storeu_ps(px, tmp);

            px += 8;
             pInA = ba + (bm/8*8) - ii + 8; // 移动到下8个A数据
        }

         // 处理剩余不足8个浮点数的数据
        int remainder = bm % 8;
        if (remainder > 0) {
            __m256i mask = _mm256_set_epi32(
                remainder > 7 ? -1 : 0,
                remainder > 6 ? -1 : 0,
                remainder > 5 ? -1 : 0,
                remainder > 4 ? -1 : 0,
                remainder > 3 ? -1 : 0,
                remainder > 2 ? -1 : 0,
                remainder > 1 ? -1 : 0,
                remainder > 0 ? -1 : 0
            );

            pInA = ba + bm - remainder;
            pInB = bb;

            vres0m8 = _mm256_set1_ps(0.0f);

            for (kk = bk; kk > 0; kk--) {
                va0m8 = _mm256_maskload_ps(pInA, mask);

                vres0m8 = _mm256_fmadd_ps(_mm256_set1_ps(*(pInB + 0)), va0m8, vres0m8);
                pInA += bm;
                pInB += 1;
            }

            __m256 tmp = _mm256_maskload_ps(px, mask);
            tmp = _mm256_add_ps(tmp, vres0m8);
            _mm256_maskstore_ps(px, mask, tmp);

        }

        
        bb += bk;
        c += ldc;
    }
    return 0;
}
#endif



/*
 * --------------------------------------------------------------------------
 */
void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        float *packA,
        float *packB,
        float *C,
        int    ldc
        )
{
#if (defined(__defualt))
    int    i, p, j;

    for ( j = 0; j < n; j ++ ) {            // Start 2-nd loop
      for ( p = 0; p < k; p ++ ) {          // Start 1-st loop
          float elem_B = packB[ j * k + p ];
          float *p_elemA = &(packA[ p * m]);
          float *p_elemC = &(C[ j * ldc]);
          for ( i = 0; i < m; i ++ ) {      // Start 0-th loop
              *p_elemC++ += *p_elemA++ * elem_B;
          }                                 // End   0-th loop
      }                                     // End   1-st loop
  }
                                           // 2-th loop around micro-kernel
#elif (defined(__AVX2))
    sgemm_mat_opt_macc_avx(m, n, k, packA, packB, C, ldc);
#endif  
}

// C must be aligned
void bl_sgemm(
        int    m,
        int    n,
        int    k,
        float *XA,
        int    lda,
        float *XB,
        int    ldb,
        float *C,        // must be aligned
        int    ldc        // ldc must also be aligned
        )
{
    int    i, j, p;
    int    ic, ib, jc, jb, pc, pb;
    int    ir, jr;
    float *packA, *packB, *packA_origin;
    char   *str;
    int mp_num_threads = 8;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        printf( "bl_sgemm(): early return\n" );
        return;
    }

    // Allocate packing buffers
    packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * mp_num_threads, sizeof(float) );
    packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 ), sizeof(float) );
    // packA_origin = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ), sizeof(float) );
    
    
    for ( jc = 0; jc < n; jc += DGEMM_NC ) {                                 // 5-th loop around micro-kernel
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += pb ) {                                   // 4-th loop around micro-kernel
            pb = min( k - pc, DGEMM_KC );
            #pragma omp parallel for num_threads(mp_num_threads)
            for(j = 0; j < jb; j += DGEMM_NR){
                 packB_kcxnc(
                    min(jb - j, DGEMM_NR),
                    pb,
                    &XB[ (jc + j) * ldb +  pc],
                    ldb,
                    &packB[ j * pb]
                    );
            }
            
            // packB_kcxnc(
            //         jb,
            //         pb,
            //         &XB[ jc * ldb +  pc],
            //         ldb,
            //         packB
            //         );
           
            #pragma omp parallel for num_threads(mp_num_threads)
            for ( ic = 0; ic < m; ic += DGEMM_MC ) {                               // 3-rd loop around micro-kernel
                ib = min( m - ic, DGEMM_MC );
                int tid = omp_get_thread_num();
                packA_mcxkc_origin(
                        ib,
                        pb,
                        &XA[ pc * lda + ic],
                        lda,
                        &packA[tid * DGEMM_MC * DGEMM_KC]
                        );
                // #pragma omp parallel for num_threads(mp_num_threads)
                // for(i = 0; i < ib; i += DGEMM_MR){
                //     packA_mcxkc(
                //         min(ib - i,DGEMM_MR),
                //         pb,
                //         &XA[ pc * lda + ic + i],
                //         lda,
                //         packA,
                //         ib,
                //         i
                //         );
                // }
                
                bl_macro_kernel(
                        ib,
                        jb,
                        pb,
                        &packA[tid * DGEMM_MC * DGEMM_KC],
                        packB,
                        &C[ jc * ldc + ic ], 
                        ldc
                        );
            }                                                                     // End 3.rd loop around micro-kernel
        }                                                                         // End 4.th loop around micro-kernel
    }                                                                             // End 5.th loop around micro-kernel

    free( packA );
    free( packB );
}
