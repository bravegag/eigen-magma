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
 *    Householder QR decomposition of a matrix w/o pivoting based on
 *    magma_?geqrf_gpu function.
 ********************************************************************************
*/

#ifndef EIGEN_QR_MAGMA_H
#define EIGEN_QR_MAGMA_H

#include "Eigen/src/Core/util/MAGMA_support.h"

namespace Eigen {

namespace internal {

/** \internal Specialization for the data types supported by MAGMA */

#define EIGEN_MAGMA_QR_NOPIV(EIGTYPE, MAGMATYPE, MAGMAPREFIX) \
template<typename MatrixQR, typename HCoeffs> \
void householder_qr_inplace_blocked(MatrixQR& mat, HCoeffs& hCoeffs, \
                                       typename MatrixQR::Index maxBlockSize=32, \
                                       EIGTYPE* tempData = 0) \
{ \
	MAGMATYPE *h_A = (MAGMATYPE*)mat.data(), *h_tau = (MAGMATYPE*)hCoeffs.data(); \
	MAGMATYPE *d_A, *d_T; \
	MAGMATYPE *h_work; \
	magma_int_t info; \
	magma_int_t M 	 = mat.rows(); \
	magma_int_t N 	 = mat.cols(); \
	magma_int_t lda  = mat.outerStride(); \
	magma_int_t nb   = magma_get_##MAGMAPREFIX##geqrf_nb( lda ); \
	/* MAGMATYPE tmp[1]; */ \
\
	/* magma_int_t lwork = -1; */ \
	/* lapackf77_##MAGMAPREFIX##geqrf(&M, &N, h_A, &lda, h_tau, tmp, &lwork, &info); */ \
	/* lwork = (magma_int_t)MAGMA_D_REAL(tmp[0]); */ \
	/* lwork = max(lwork, max(N*nb, 2*nb*nb)); */ \
	/* MAGMA_MALLOC( h_work, MAGMATYPE, lwork ); */ \
\
	magma_int_t ldda = ((lda+31)/32)*32; \
	MAGMA_DEVALLOC(  d_A, MAGMATYPE, ldda*N ); \
	magma_dsetmatrix( M, N, h_A, lda, d_A, ldda ); \
\
	magma_int_t size = (2*min(M, N) + (N+31)/32)*nb; \
	magma_dmalloc( &d_T, size ); \
\
	/* magma_##MAGMAPREFIX##geqrf( M, N, h_A, lda, h_tau, h_work, lwork, &info); */ \
	magma_##MAGMAPREFIX##geqrf3_gpu( M, N, d_A, ldda, h_tau, d_T, &info); \
	hCoeffs.adjointInPlace(); \
\	
	magma_free(  d_T ); \
	MAGMA_DEVFREE(  d_A ); \
	/* MAGMA_FREE( h_work ); */ \
\
}

EIGEN_MAGMA_QR_NOPIV(double,	double,			d)
EIGEN_MAGMA_QR_NOPIV(float,	float,			s)
EIGEN_MAGMA_QR_NOPIV(dcomplex,	magmaDoubleComplex,	z)
EIGEN_MAGMA_QR_NOPIV(scomplex,	magmaFloatComplex,	c)

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_QR_MAGMA_H
