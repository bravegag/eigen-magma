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
 *   General matrix-vector product functionality based on ?GEMV.
 ********************************************************************************
*/

#ifndef EIGEN_GENERAL_MATRIX_VECTOR_MAGMA_H
#define EIGEN_GENERAL_MATRIX_VECTOR_MAGMA_H

#include <stdio.h>
#include <magma.h>
#include "Eigen/src/Core/util/MAGMA_support.h"

namespace Eigen {

namespace internal {

/**********************************************************************
* This file implements general matrix-vector multiplication using BLAS
* gemv function via partial specialization of
* general_matrix_vector_product::run(..) method for float, double,
* std::complex<float> and std::complex<double> types
**********************************************************************/

// gemv specialization

template<typename Index, typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs, typename RhsScalar, bool ConjugateRhs>
struct general_matrix_vector_product_gemv :
  general_matrix_vector_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,ConjugateRhs,BuiltIn> {};

#define EIGEN_MAGMA_GEMV_SPECIALIZE(Scalar) \
template<typename Index, bool ConjugateLhs, bool ConjugateRhs> \
struct general_matrix_vector_product<Index,Scalar,ColMajor,ConjugateLhs,Scalar,ConjugateRhs,Specialized> { \
static EIGEN_DONT_INLINE void run( \
  Index rows, Index cols, \
  const Scalar* lhs, Index lhsStride, \
  const Scalar* rhs, Index rhsIncr, \
  Scalar* res, Index resIncr, Scalar alpha) \
{ \
  if (ConjugateLhs) { \
    general_matrix_vector_product<Index,Scalar,ColMajor,ConjugateLhs,Scalar,ConjugateRhs,BuiltIn>::run( \
      rows, cols, lhs, lhsStride, rhs, rhsIncr, res, resIncr, alpha); \
  } else { \
    general_matrix_vector_product_gemv<Index,Scalar,ColMajor,ConjugateLhs,Scalar,ConjugateRhs>::run( \
      rows, cols, lhs, lhsStride, rhs, rhsIncr, res, resIncr, alpha); \
  } \
} \
}; \
template<typename Index, bool ConjugateLhs, bool ConjugateRhs> \
struct general_matrix_vector_product<Index,Scalar,RowMajor,ConjugateLhs,Scalar,ConjugateRhs,Specialized> { \
static EIGEN_DONT_INLINE void run( \
  Index rows, Index cols, \
  const Scalar* lhs, Index lhsStride, \
  const Scalar* rhs, Index rhsIncr, \
  Scalar* res, Index resIncr, Scalar alpha) \
{ \
    general_matrix_vector_product_gemv<Index,Scalar,RowMajor,ConjugateLhs,Scalar,ConjugateRhs>::run( \
      rows, cols, lhs, lhsStride, rhs, rhsIncr, res, resIncr, alpha); \
} \
}; \

EIGEN_MAGMA_GEMV_SPECIALIZE(double)
EIGEN_MAGMA_GEMV_SPECIALIZE(float)
EIGEN_MAGMA_GEMV_SPECIALIZE(dcomplex)
EIGEN_MAGMA_GEMV_SPECIALIZE(scomplex)

#define EIGEN_MAGMA_GEMV_SPECIALIZATION(EIGTYPE,MAGMATYPE,MAGMAPREFIX,MAGMAPREFIXLOW) \
template<typename Index, int LhsStorageOrder, bool ConjugateLhs, bool ConjugateRhs> \
struct general_matrix_vector_product_gemv<Index,EIGTYPE,LhsStorageOrder,ConjugateLhs,EIGTYPE,ConjugateRhs> \
{ \
typedef Matrix<EIGTYPE,Dynamic,1,ColMajor> GEMVVector;\
\
static EIGEN_DONT_INLINE void run( \
  Index rows, Index cols, \
  const EIGTYPE* lhs, Index lhsStride, \
  const EIGTYPE* rhs, Index rhsIncr, \
  EIGTYPE* res, Index resIncr, EIGTYPE alpha) \
{ \
  magma_int_t M=rows, N=cols, lda=lhsStride, ldda, incx=rhsIncr, incy=resIncr; \
  MAGMATYPE alpha_, beta_ = 1; \
  const EIGTYPE *x_ptr; \
  if (LhsStorageOrder==RowMajor) { \
    M=cols; \
    N=rows; \
  }\
  assign_scalar_eig2magma(alpha_, alpha); \
  GEMVVector x_tmp; \
  if (ConjugateRhs) { \
    Map<const GEMVVector, 0, InnerStride<> > map_x(rhs,cols,1,InnerStride<>(incx)); \
    x_tmp=map_x.conjugate(); \
    x_ptr=x_tmp.data(); \
    incx=1; \
  } else x_ptr=rhs; \
\
  magma_trans_t trans = (LhsStorageOrder == RowMajor) ? ((ConjugateLhs) ? MagmaConjTrans : MagmaTrans) : MagmaNoTrans; \
  magma_int_t Xm, Ym, sizeA, sizeX, sizeY; \
  ldda = ((lda+31)/32)*32; \
\
  if ( trans == MagmaNoTrans ) { \
    Xm = N; \
    Ym = M; \
  } else {  \
    Xm = M; \
    Ym = N; \
  } \
\
  sizeA = ldda*N;  \
  sizeX = incx*Xm; \
  sizeY = incy*Ym; \
\
  const MAGMATYPE *hA, *hX; \
  MAGMATYPE *hY; \
  MAGMATYPE *dA, *dX, *dY; \
  hA = lhs; \
  hX = x_ptr; \
  hY = res; \
\
  MAGMA_DEVALLOC( dA, MAGMATYPE, sizeA ); \
  MAGMA_DEVALLOC( dX, MAGMATYPE, sizeX ); \
  MAGMA_DEVALLOC( dY, MAGMATYPE, sizeY ); \
\
  magma_dsetmatrix( M, N, hA, lda, dA, ldda ); \
  magma_dsetvector( Xm, hX, incx, dX, incx ); \
  magma_dsetvector( Ym, hY, incy, dY, incy ); \
\
  cublas##MAGMAPREFIX##gemv( trans, M, N, alpha_, dA, ldda, dX, incx, beta_, dY, incy ); \
\
  magma_dgetvector( Ym, dY, incy, hY, incy ); \
\
  MAGMA_DEVFREE( dA ); \
  MAGMA_DEVFREE( dX ); \
  MAGMA_DEVFREE( dY ); \
}\
};

EIGEN_MAGMA_GEMV_SPECIALIZATION(double,   double, 	      D, d)
EIGEN_MAGMA_GEMV_SPECIALIZATION(float,    float,	      S, s)
EIGEN_MAGMA_GEMV_SPECIALIZATION(dcomplex, magmaDoubleComplex, Z, z)
EIGEN_MAGMA_GEMV_SPECIALIZATION(scomplex, magmaFloatComplex,  C, c)

} // end namespase internal

} // end namespace Eigen

#endif // EIGEN_GENERAL_MATRIX_VECTOR_MAGMA_H
