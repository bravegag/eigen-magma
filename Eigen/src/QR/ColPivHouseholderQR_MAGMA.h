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
 *    Householder QR decomposition of a matrix with column pivoting based on
 *    magma_?geqp3_gpu function.
 ********************************************************************************
*/

#ifndef EIGEN_COLPIVOTINGHOUSEHOLDERQR_MAGMA_H
#define EIGEN_COLPIVOTINGHOUSEHOLDERQR_MAGMA_H

#include "Eigen/src/Core/util/MAGMA_support.h"
#include <stdio.h>

namespace Eigen {

/** \internal Specialization for the data types supported by MAGMA */

#define EIGEN_MAGMA_QR_COLPIV(EIGTYPE, MAGMATYPE, MAGMAPREFIX, EIGCOLROW, MAGMACOLROW) \
template<> inline \
ColPivHouseholderQR<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> >& \
ColPivHouseholderQR<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> >::compute( \
              const Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>& matrix) \
\
{ \
	using std::abs; \
	magma_int_t checkres; \
	MAGMATYPE *tau, *h_work; \
	magma_int_t *jpvt; \
\
	/* Matrix size */ \
	magma_int_t M = 0, N = 0, n2, lda, lwork; \
	magma_int_t i, j, info = 2, min_mn, nb; \
	magma_int_t ione = 1; \
	MAGMATYPE *d_R, *dtau, *d_work; \
\
	typedef Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> MatrixType; \
	typedef MatrixType::Scalar Scalar; \
	typedef MatrixType::RealScalar RealScalar; \
	Index rows = matrix.rows(); \
	Index cols = matrix.cols(); \
	Index size = matrix.diagonalSize(); \
\
	m_qr = matrix; \
	m_hCoeffs.resize(size); \
\
	m_colsTranspositions.resize(cols); \
	/* Index number_of_transpositions = 0; */ \
\
	m_nonzero_pivots = 0; \
	m_maxpivot = RealScalar(0); \
	m_colsPermutation.resize(cols); \
	m_colsPermutation.indices().setZero(); \
\
	lda = m_qr.outerStride(); \
\
	M = rows, N = cols; \
	min_mn = M < N ? M : N; \
	n2     = lda*N; \
\
	MAGMA_MALLOC( tau,  MAGMATYPE,   min_mn ); \
	nb 	  = magma_get_##MAGMAPREFIX##geqp3_nb(min_mn); \
	lwork = ( N+1 )*nb; \
	lwork += 2*N; \
\
	jpvt = (magma_int_t*)m_colsPermutation.indices().data(); \
\
	/* allocate gpu workspaces */ \
	MAGMA_DEVALLOC( d_R,    MAGMATYPE, lda*N  ); \
	MAGMA_DEVALLOC( dtau,   MAGMATYPE, min_mn ); \
	MAGMA_DEVALLOC( d_work, MAGMATYPE, lwork  ); \
\
	/* copy A to gpu */ \
	magma_##MAGMAPREFIX##setmatrix( M, N, (MAGMATYPE*)m_qr.data(), lda, d_R, lda );     \
	magma_##MAGMAPREFIX##setvector( min_mn, (MAGMATYPE*)m_hCoeffs.data(), 1, dtau, 1 ); \
\
	magma_##MAGMAPREFIX##geqp3_gpu( M, N, d_R, lda, jpvt, dtau, d_work, lwork, &info ); \
\
	/* copy outputs to cpu */ \
	magma_##MAGMAPREFIX##getmatrix( M, N, d_R, lda, (MAGMATYPE*)m_qr.data(), lda ); \
	magma_##MAGMAPREFIX##getvector( min_mn, dtau, 1, (MAGMATYPE*)m_hCoeffs.data(), 1 ); \
\
	/* cleanup */ \
	MAGMA_DEVFREE( d_work ); \
	MAGMA_DEVFREE( dtau ); \
	MAGMA_DEVFREE( d_R ); \
\
	m_isInitialized = true; \
	m_maxpivot=m_qr.diagonal().cwiseAbs().maxCoeff(); \
	m_hCoeffs.adjointInPlace(); \
	RealScalar premultiplied_threshold = abs(m_maxpivot) * threshold(); \
	magma_int_t *perm = m_colsPermutation.indices().data(); \
	for(i=0;i<size;i++) { \
		m_nonzero_pivots += (abs(m_qr.coeff(i,i)) > premultiplied_threshold); \
	} \
	for(i=0;i<cols;i++) perm[i]--; \
\
	/*m_det_pq = (number_of_transpositions%2) ? -1 : 1;  // TODO: It's not needed now; fix upon availability in Eigen */ \
\
	return *this; \
}

EIGEN_MAGMA_QR_COLPIV  (double,   double,	      d, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MAGMA_QR_COLPIV  (float,    float,              s, ColMajor, LAPACK_COL_MAJOR)
//EIGEN_MAGMA_QR_COLPIV(dcomplex, magmaDoubleComplex, z, ColMajor, LAPACK_COL_MAJOR)
//EIGEN_MAGMA_QR_COLPIV(scomplex, magmaFloatComplex,  c, ColMajor, LAPACK_COL_MAJOR)

EIGEN_MAGMA_QR_COLPIV  (double,   double,             d, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MAGMA_QR_COLPIV  (float,    float,              s, RowMajor, LAPACK_ROW_MAJOR)
//EIGEN_MAGMA_QR_COLPIV(dcomplex, magmaDoubleComplex, z, RowMajor, LAPACK_ROW_MAJOR)
//EIGEN_MAGMA_QR_COLPIV(scomplex, magmaFloatComplex,  c, RowMajor, LAPACK_ROW_MAJOR)

} // end namespace Eigen

#endif // EIGEN_COLPIVOTINGHOUSEHOLDERQR_MAGMA_H
