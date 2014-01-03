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
 *   Triangular matrix * matrix product functionality based on ?TRMM.
 ********************************************************************************
*/

#ifndef EIGEN_TRIANGULAR_SOLVER_MATRIX_MAGMA_H
#define EIGEN_TRIANGULAR_SOLVER_MATRIX_MAGMA_H

namespace Eigen {

namespace internal {

// implements LeftSide op(triangular)^-1 * general
#define EIGEN_MAGMA_TRSM_L(EIGTYPE, MAGMATYPE, MAGMAPREFIX) \
template <typename Index, int Mode, bool Conjugate, int TriStorageOrder> \
struct triangular_solve_matrix<EIGTYPE,Index,OnTheLeft,Mode,Conjugate,TriStorageOrder,ColMajor> \
{ \
  enum { \
    IsLower = (Mode&Lower) == Lower, \
    IsUnitDiag  = (Mode&UnitDiag) ? 1 : 0, \
    IsZeroDiag  = (Mode&ZeroDiag) ? 1 : 0, \
    conjA = ((TriStorageOrder==ColMajor) && Conjugate) ? 1 : 0 \
  }; \
  static EIGEN_DONT_INLINE void run( \
      Index size, Index otherSize, \
      const EIGTYPE* _tri, Index triStride, \
      EIGTYPE* _other, Index otherStride, level3_blocking<EIGTYPE,EIGTYPE>& /*blocking*/) \
  { \
   magma_int_t M = size, N = otherSize; \
   magma_side_t side = MagmaLeft; \
   magma_int_t lda, ldb, ldda, lddb, Ak; \
   const MAGMATYPE *h_A; \
   MAGMATYPE *h_B, *d_A, *d_B; \
\
   /* Set alpha_ */ \
   MAGMATYPE alpha; \
   EIGTYPE myone(1); \
   assign_scalar_eig2magma(alpha, myone); \
   ldb = otherStride;\
\
   const EIGTYPE *a; \
/* Set trans */ \
   magma_trans_t transA = (TriStorageOrder == RowMajor) ? ((Conjugate) ? MagmaConjTrans : MagmaTrans) : MagmaNoTrans; \
/* Set uplo */ \
   magma_uplo_t uplo = IsLower ? MagmaLower : MagmaUpper; \
   if (TriStorageOrder==RowMajor) uplo = (uplo == MagmaLower) ? MagmaUpper : MagmaLower; \
/* Set a, lda */ \
   typedef Matrix<EIGTYPE, Dynamic, Dynamic, TriStorageOrder> MatrixTri; \
   Map<const MatrixTri, 0, OuterStride<> > tri(_tri,size,size,OuterStride<>(triStride)); \
   MatrixTri a_tmp; \
\
   if (conjA) { \
     a_tmp = tri.conjugate(); \
     a = a_tmp.data(); \
     lda = a_tmp.outerStride(); \
   } else { \
     a = _tri; \
     lda = triStride; \
   } \
   magma_diag_t diag = (IsUnitDiag) ? 'U' : 'N'; \
\
   if ( side == MagmaLeft ) { \
	   lda = M; \
	   Ak = M; \
   } else { \
	   lda = N; \
	   Ak = N; \
   } \
   ldb = M; \
\
   ldda = ((lda+31)/32)*32; \
   lddb = ((ldb+31)/32)*32; \
\
   MAGMA_DEVALLOC( d_A, MAGMATYPE, ldda*Ak ); \
   MAGMA_DEVALLOC( d_B, MAGMATYPE, lddb*N  ); \
\
   h_A = (const MAGMATYPE*)a; \
   h_B = (MAGMATYPE*)_other; \
\
   magma_dsetmatrix( Ak, Ak, h_A, lda, d_A, ldda ); \
   magma_dsetmatrix( M, N, h_B, ldb, d_B, lddb ); \
\
/* call ?trsm*/ \
   cublas##MAGMAPREFIX##trsm( side, uplo, transA, diag, M, N, alpha, d_A, ldda, d_B, lddb ); \
\
   magma_dgetmatrix( M, N, d_B, lddb, h_B, ldb ); \
\
   MAGMA_DEVFREE( d_A ); \
   MAGMA_DEVFREE( d_B ); \
 } \
};

EIGEN_MAGMA_TRSM_L(double,	double,			D)
EIGEN_MAGMA_TRSM_L(dcomplex, 	magmaDoubleComplex,	Z)
EIGEN_MAGMA_TRSM_L(float,	float,			S)
EIGEN_MAGMA_TRSM_L(scomplex, 	magmaFloatComplex,	C)


// implements RightSide general * op(triangular)^-1
#define EIGEN_MAGMA_TRSM_R(EIGTYPE, MAGMATYPE, MAGMAPREFIX) \
template <typename Index, int Mode, bool Conjugate, int TriStorageOrder> \
struct triangular_solve_matrix<EIGTYPE,Index,OnTheRight,Mode,Conjugate,TriStorageOrder,ColMajor> \
{ \
  enum { \
    IsLower = (Mode&Lower) == Lower, \
    IsUnitDiag  = (Mode&UnitDiag) ? 1 : 0, \
    IsZeroDiag  = (Mode&ZeroDiag) ? 1 : 0, \
    conjA = ((TriStorageOrder==ColMajor) && Conjugate) ? 1 : 0 \
  }; \
  static EIGEN_DONT_INLINE void run( \
      Index size, Index otherSize, \
      const EIGTYPE* _tri, Index triStride, \
      EIGTYPE* _other, Index otherStride, level3_blocking<EIGTYPE,EIGTYPE>& /*blocking*/) \
  { \
   magma_int_t M = otherSize, N = size, lda, ldb; \
   magma_side_t side = MagmaRight; \
   magma_int_t ldda, lddb, Ak; \
   const MAGMATYPE *h_A; \
   MAGMATYPE *h_B, *d_A, *d_B; \
\
   /* Set alpha_ */ \
   MAGMATYPE alpha; \
   EIGTYPE myone(1); \
   assign_scalar_eig2magma(alpha, myone); \
   ldb = otherStride;\
\
   const EIGTYPE *a; \
/* Set trans */ \
   magma_trans_t transA = (TriStorageOrder == RowMajor) ? ((Conjugate) ? MagmaConjTrans : MagmaTrans) : MagmaNoTrans; \
/* Set uplo */ \
   magma_uplo_t uplo = IsLower ? MagmaLower : MagmaUpper; \
   if (TriStorageOrder==RowMajor) uplo = (uplo == MagmaLower) ? MagmaUpper : MagmaLower; \
/* Set a, lda */ \
   typedef Matrix<EIGTYPE, Dynamic, Dynamic, TriStorageOrder> MatrixTri; \
   Map<const MatrixTri, 0, OuterStride<> > tri(_tri,size,size,OuterStride<>(triStride)); \
   MatrixTri a_tmp; \
\
   if (conjA) { \
     a_tmp = tri.conjugate(); \
     a = a_tmp.data(); \
     lda = a_tmp.outerStride(); \
   } else { \
     a = _tri; \
     lda = triStride; \
   } \
   magma_diag_t diag = (IsUnitDiag) ? 'U' : 'N'; \
\
   if ( side == MagmaLeft ) { \
	lda = M; \
	Ak = M; \
   } else { \
	lda = N; \
	Ak = N; \
   } \
   ldb = M; \
\
   ldda = ((lda+31)/32)*32; \
   lddb = ((ldb+31)/32)*32; \
\
   MAGMA_DEVALLOC( d_A, MAGMATYPE, ldda*Ak ); \
   MAGMA_DEVALLOC( d_B, MAGMATYPE, lddb*N  ); \
\
   h_A = (const MAGMATYPE*)a; \
   h_B = (MAGMATYPE*)_other; \
\
   magma_dsetmatrix( Ak, Ak, h_A, lda, d_A, ldda ); \
   magma_dsetmatrix( M, N, h_B, ldb, d_B, lddb ); \
\
   /* call ?trsm*/ \
   cublas##MAGMAPREFIX##trsm( side, uplo, transA, diag, M, N, alpha, d_A, ldda, d_B, lddb ); \
\
   magma_dgetmatrix( M, N, d_B, lddb, h_B, ldb ); \
\
   MAGMA_DEVFREE( d_A ); \
   MAGMA_DEVFREE( d_B ); \
  } \
};

EIGEN_MAGMA_TRSM_R(double,	double,			D)
EIGEN_MAGMA_TRSM_R(dcomplex,	magmaDoubleComplex,	Z)
EIGEN_MAGMA_TRSM_R(float,	float,			S)
EIGEN_MAGMA_TRSM_R(scomplex,	magmaFloatComplex,	C)


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TRIANGULAR_SOLVER_MATRIX_MAGMA_H
