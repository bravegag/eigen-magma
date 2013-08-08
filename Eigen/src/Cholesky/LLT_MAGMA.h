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
 *   Content : Eigen bindings to Intel(R) MAGMA
 *     LLt decomposition based on LAPACKE_?potrf function.
 ********************************************************************************
*/

#ifndef EIGEN_LLT_MAGMA_H
#define EIGEN_LLT_MAGMA_H

#include "Eigen/src/Core/util/MAGMA_support.h"
#include <iostream>

namespace Eigen {

namespace internal {

template<typename Scalar> struct magma_llt;

#define EIGEN_MAGMA_LLT(EIGTYPE, MAGMATYPE, MAGMAPREFIX) \
template<> struct magma_llt<EIGTYPE> \
{ \
  template<typename MatrixType> \
  static inline typename MatrixType::Index potrf(MatrixType& m, char uplo) \
  { \
    magma_int_t matrix_order; \
    magma_int_t N, size, lda, ldda, info, StorageOrder; \
    MAGMATYPE *h_A, *h_R; \
    MAGMATYPE *d_A; \
    EIGTYPE* a; \
    MAGMATYPE c_neg_one = MAGMA_D_NEG_ONE; \
    eigen_assert(m.rows()==m.cols()); \
    /* Set up parameters for ?potrf */ \
    size = m.rows(); \
    N = m.rows(); \
    lda = m.outerStride(); \
    ldda = ((lda+31)/32)*32; \
    StorageOrder = MatrixType::Flags&RowMajorBit?RowMajor:ColMajor; \
    /* TODO: matrix_order = StorageOrder==RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR; */ \
    a = &(m.coeffRef(0,0)); \
    h_A = a; \
\
    MAGMA_DEVALLOC(  d_A, double, ldda*N ); \
    magma_dsetmatrix( N, N, h_A, lda, d_A, ldda ); \
\
    magma_##MAGMAPREFIX##potrf_gpu( uplo, N, d_A, ldda, &info ); \
    info = (info==0) ? Success : NumericalIssue; \
\
    magma_dgetmatrix( N, N, d_A, ldda, h_A, lda ); \
    MAGMA_DEVFREE(  d_A ); \
\
    return info; \
  } \
}; \
template<> struct llt_inplace<EIGTYPE, Lower> \
{ \
  template<typename MatrixType> \
  static typename MatrixType::Index blocked(MatrixType& m) \
  { \
    return magma_llt<EIGTYPE>::potrf(m, 'L'); \
  } \
  template<typename MatrixType, typename VectorType> \
  static typename MatrixType::Index rankUpdate(MatrixType& mat, const VectorType& vec, const typename MatrixType::RealScalar& sigma) \
  { return Eigen::internal::llt_rank_update_lower(mat, vec, sigma); } \
}; \
template<> struct llt_inplace<EIGTYPE, Upper> \
{ \
  template<typename MatrixType> \
  static typename MatrixType::Index blocked(MatrixType& m) \
  { \
    return magma_llt<EIGTYPE>::potrf(m, 'U'); \
  } \
  template<typename MatrixType, typename VectorType> \
  static typename MatrixType::Index rankUpdate(MatrixType& mat, const VectorType& vec, const typename MatrixType::RealScalar& sigma) \
  { \
    Transpose<MatrixType> matt(mat); \
    return llt_inplace<EIGTYPE, Lower>::rankUpdate(matt, vec.conjugate(), sigma); \
  } \
};

EIGEN_MAGMA_LLT(double,	  double,		d)
EIGEN_MAGMA_LLT(float,	  float,		s)
EIGEN_MAGMA_LLT(dcomplex, magmaDoubleComplex,	z)
EIGEN_MAGMA_LLT(scomplex, magmaFloatComplex,	c)

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_LLT_MAGMA_H
