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
 *    Singular Value Decomposition - SVD.
 ********************************************************************************
*/

#ifndef EIGEN_JACOBISVD_MAGMA_H
#define EIGEN_JACOBISVD_MAGMA_H

#include "Eigen/src/Core/util/MAGMA_support.h"

namespace Eigen {

/** \internal Specialization for the data types supported by MAGMA */

#define EIGEN_MAGMA_SVD(EIGTYPE, MAGMATYPE, MAGMARTYPE, MAGMAPREFIX, EIGCOLROW) \
template<> inline \
JacobiSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, ColPivHouseholderQRPreconditioner>& \
JacobiSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, ColPivHouseholderQRPreconditioner>::compute(const Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>& matrix, unsigned int computationOptions) \
{ \
  typedef Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> MatrixType; \
  typedef MatrixType::Scalar Scalar; \
  typedef MatrixType::RealScalar RealScalar; \
  allocate(matrix.rows(), matrix.cols(), computationOptions); \
\
  /*const RealScalar precision = RealScalar(2) * NumTraits<Scalar>::epsilon();*/ \
  m_nonzeroSingularValues = m_diagSize; \
\
  magma_int_t lda = matrix.outerStride(), ldu; \
  magma_int_t ldvt, lwork, M, N, min_mn, nb, n2, info; \
  MAGMATYPE *h_A, *h_R, *h_U, *h_VT, *h_work; \
  MAGMATYPE *h_S1, *h_S2; \
  char jobu, jobvt; \
  MAGMATYPE *u, *vt, dummy; \
\
  M = m_rows; \
  N = m_cols; \
\
  jobu  = (m_computeFullU) ? 'A' : (m_computeThinU) ? 'S' : 'N'; \
  jobvt = (m_computeFullV) ? 'A' : (m_computeThinV) ? 'S' : 'N'; \
  if (computeU()) { \
    ldu  = m_matrixU.outerStride(); \
    u    = (MAGMATYPE*)m_matrixU.data(); \
  } else { \
    ldu  = 1; \
    u    = &dummy; \
  } \
  MatrixType localV; \
  ldvt = (m_computeFullV) ? m_cols : (m_computeThinV) ? m_diagSize : 1; \
  if (computeV()) { \
    localV.resize(ldvt, m_cols); \
    vt   = (MAGMATYPE*)localV.data(); \
  } else { ldvt=1; vt=&dummy; }\
  Matrix<MAGMARTYPE, Dynamic, Dynamic> superb; superb.resize(m_diagSize, 1); \
  MatrixType m_temp; m_temp = matrix; \
\
  h_R  = (MAGMATYPE*)m_temp.data(); \
  h_S1 = (MAGMARTYPE*)m_singularValues.data(); \
  h_U  = (MAGMARTYPE*)u; \
  h_VT = (MAGMARTYPE*)vt; \
  n2 = M*N; \
  min_mn = M < N ? M : N; \
  nb = magma_get_##MAGMAPREFIX##gesvd_nb(N); \
  lwork = (M+N)*nb + 3*min_mn + 2*min_mn*min_mn; \
  MAGMA_HOSTALLOC( h_work, MAGMATYPE, lwork ); \
\
  magma_##MAGMAPREFIX##gesvd( jobu, jobvt, M, N, h_R, lda, h_S1, h_U, ldu, h_VT, N, h_work, lwork, &info ); \
  if (info != 0) { \
      fprintf(stderr, "magma_?gesvd returned error %d: %s.\n", (int) info, magma_strerror( info )); \
  } \
  MAGMA_HOSTFREE( h_work ); \
  if (computeV()) m_matrixV = localV.adjoint(); \
 /* for(int i=0;i<m_diagSize;i++) if (m_singularValues.coeffRef(i) < precision) { m_nonzeroSingularValues--; m_singularValues.coeffRef(i)=RealScalar(0);}*/ \
  m_isInitialized = true; \
  return *this; \
}

EIGEN_MAGMA_SVD(double,   double,		double, d, ColMajor)
EIGEN_MAGMA_SVD(float,    float,		float , s, ColMajor)
//EIGEN_MAGMA_SVD(dcomplex, magmaDoubleComplex,	double, z, ColMajor)
//EIGEN_MAGMA_SVD(scomplex, magmaFloatComplex,	float , c, ColMajor)

EIGEN_MAGMA_SVD(double,   double,		double, d, RowMajor)
EIGEN_MAGMA_SVD(float,    float,		float , s, RowMajor)
//EIGEN_MAGMA_SVD(dcomplex, magmaDoubleComplex,	double, z, RowMajor)
//EIGEN_MAGMA_SVD(scomplex, magmaFloatComplex,	float , c, RowMajor)

} // end namespace Eigen

#endif // EIGEN_JACOBISVD_MAGMA_H
