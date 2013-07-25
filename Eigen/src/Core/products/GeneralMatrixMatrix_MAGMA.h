/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to MAGMA
 *   General matrix-matrix product functionality based on ?GEMM.
 ********************************************************************************
*/

#ifndef EIGEN_GENERAL_MATRIX_MATRIX_MAGMA_H
#define EIGEN_GENERAL_MATRIX_MATRIX_MAGMA_H

#include <stdio.h>
#include <magma.h>

namespace Eigen {

namespace internal {

/**********************************************************************
* This file implements general matrix-matrix multiplication using BLAS
* gemm function via partial specialization of
* general_matrix_matrix_product::run(..) method for float, double,
* std::complex<float> and std::complex<double> types
**********************************************************************/

// gemm specialization

#define GEMM_SPECIALIZATION(EIGTYPE, EIGPREFIX, MAGMATYPE, MAGMAPREFIX, MAGMAPREFIXLOW) \
template< \
  typename Index, \
  int LhsStorageOrder, bool ConjugateLhs, \
  int RhsStorageOrder, bool ConjugateRhs> \
struct general_matrix_matrix_product<Index,EIGTYPE,LhsStorageOrder,ConjugateLhs,EIGTYPE,RhsStorageOrder,ConjugateRhs,ColMajor> \
{ \
static void run(Index rows, Index cols, Index depth, \
  const EIGTYPE* _lhs, Index lhsStride, \
  const EIGTYPE* _rhs, Index rhsStride, \
  EIGTYPE* res, Index resStride, \
  EIGTYPE alpha, \
  level3_blocking<EIGTYPE, EIGTYPE>& /*blocking*/, \
  GemmParallelInfo<Index>* /*info = 0*/) \
{ \
 	const MAGMATYPE *h_A, *h_B; \
 	MAGMATYPE *h_C; \
 	MAGMATYPE *d_A, *d_B, *d_C; \
 	magma_int_t M, N, K; \
 	magma_int_t Am, An, Bm, Bn; \
 	magma_int_t sizeA, sizeB, sizeC; \
 	magma_int_t lda, ldb, ldc, ldda, lddb, lddc; \
	MAGMATYPE alpha_, beta_; \
	EIGTYPE one(1); \
\
	M = rows, N = cols, K = depth; \
\
	h_A = _lhs; \
	h_B = _rhs; \
	h_C = res; 	\
\
	magma_trans_t transA = (LhsStorageOrder == RowMajor) ? ((ConjugateLhs) ? MagmaConjTrans : MagmaTrans) : MagmaNoTrans; \
	magma_trans_t transB = (RhsStorageOrder == RowMajor) ? ((ConjugateRhs) ? MagmaConjTrans : MagmaTrans) : MagmaNoTrans; \
\
	if ( transA == MagmaNoTrans ) { \
		lda = Am = M; \
		An = K; \
	} else { \
		lda = Am = K; \
		An = M; \
	} \
\
	if ( transB == MagmaNoTrans ) { \
		ldb = Bm = K; \
		Bn = N; \
	} else { \
 		ldb = Bm = N; \
		Bn = K; \
	} \
	ldc = M; \
\
	ldda = ((lda+31)/32)*32; \
	lddb = ((ldb+31)/32)*32; \
	lddc = ((ldc+31)/32)*32; \
\
	sizeA = lda*An; \
	sizeB = ldb*Bn; \
	sizeC = ldc*N; \
\
	MAGMA_DEVALLOC( d_A, MAGMATYPE, ldda*An ); \
	MAGMA_DEVALLOC( d_B, MAGMATYPE, lddb*Bn ); \
	MAGMA_DEVALLOC( d_C, MAGMATYPE, lddc*N  ); \
\
	magma_dsetmatrix( Am, An, h_A, lda, d_A, ldda ); \
	magma_dsetmatrix( Bm, Bn, h_B, ldb, d_B, lddb ); \
	magma_dsetmatrix( M, N, h_C, ldc, d_C, lddc ); \
\
	/* Set alpha_ & beta_ */ \
	assign_scalar_eig2magma(alpha_, alpha); \
	assign_scalar_eig2magma(beta_ , one); \
\
	cublas##MAGMAPREFIX##gemm( transA, transB, M, N, K, \
		alpha_, d_A, ldda, 		\
			d_B, lddb, 		\
		beta_,  d_C, lddc ); 	\
\
	magma_##MAGMAPREFIXLOW##getmatrix( M, N, d_C, lddc, h_C, ldc ); \
\
	MAGMA_DEVFREE( d_A ); \
	MAGMA_DEVFREE( d_B ); \
	MAGMA_DEVFREE( d_C ); \
\
}};

GEMM_SPECIALIZATION(double,   d,  double, D, d)
GEMM_SPECIALIZATION(float,    f,  float,  S, s)
GEMM_SPECIALIZATION(dcomplex, cd, magmaDoubleComplex, Z, z)
GEMM_SPECIALIZATION(scomplex, cf, magmaFloatComplex,  C, c)

} // end namespase internal

} // end namespace Eigen

#endif // EIGEN_GENERAL_MATRIX_MATRIX_MAGMA_H
