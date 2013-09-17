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
 *   Include file with common MAGMA declarations
 ********************************************************************************
*/

#ifndef EIGEN_MAGMA_SUPPORT_H
#define EIGEN_MAGMA_SUPPORT_H

#ifdef EIGEN_USE_MAGMA_ALL
  #ifndef EIGEN_USE_CUBLAS
    #define EIGEN_USE_CUBLAS
  #endif
#endif

#if defined(EIGEN_USE_CUBLAS) || defined(EIGEN_USE_MAGMA_VML)
  #define EIGEN_USE_MAGMA
  #define HAVE_CUBLAS 1
#endif

#if defined(EIGEN_USE_MAGMA)

#include <cuda_runtime_api.h>
#include <cublas.h>
#include <magma.h>
//#include <magma_lapack.h>

#define EIGEN_MAGMA_VML_THRESHOLD 128

#define MAGMA_INIT()                                                       \
    magma_init();                                                          \
    if( CUBLAS_STATUS_SUCCESS != cublasInit() ) {                          \
        fprintf(stderr, "ERROR: cublasInit failed\n");                     \
        magma_finalize();                                                  \
        exit(-1);                                                          \
    }                                                                      \
    magma_print_devices();

#define MAGMA_FINALIZE()                                                   \
    magma_finalize();                                                      \
    cublasShutdown();

#define MAGMA_INIT_MGPU()                                                  \
{                                                                          \
    magma_init();                                                          \
    int ndevices;                                                          \
    cudaGetDeviceCount( &ndevices );                                       \
    for( int idevice = 0; idevice < ndevices; ++idevice ) {                \
        magma_setdevice( idevice );                                        \
        if( CUBLAS_STATUS_SUCCESS != cublasInit() ) {                      \
            fprintf(stderr, "ERROR: gpu %d: cublasInit failed\n", idevice);\
            magma_finalize();                                              \
            exit(-1);                                                      \
        }                                                                  \
    }                                                                      \
    magma_setdevice(0);                                                    \
    magma_print_devices();                                                 \
}

#define MAGMA_FINALIZE_MGPU()                                              \
{                                                                          \
    magma_finalize();                                                      \
    int ndevices;                                                          \
    cudaGetDeviceCount( &ndevices );                                       \
    for( int idevice = 0; idevice < ndevices; ++idevice ) {                \
        magma_setdevice(idevice);                                          \
        cublasShutdown();                                                  \
    }                                                                      \
}

#define MAGMA_MALLOC( ptr, type, size )                                    \
    if ( MAGMA_SUCCESS !=                                                  \
            magma_malloc_cpu( (void**) &ptr, (size)*sizeof(type) )) {      \
        fprintf( stderr, "!!!! malloc failed for: %s\n", #ptr );           \
        magma_finalize();                                                  \
        exit(-1);                                                          \
    }


#define MAGMA_HOSTALLOC( ptr, type, size )                                    \
    if ( MAGMA_SUCCESS !=                                                     \
            magma_malloc_pinned( (void**) &ptr, (size)*sizeof(type) )) {      \
        fprintf( stderr, "!!!! magma_malloc_pinned failed for: %s\n", #ptr ); \
        magma_finalize();                                                     \
        exit(-1);                                                             \
    }

#define MAGMA_HOSTREALLOC( result, ptr, new_size, old_size )		\
	 if ( MAGMA_SUCCESS !=						\
            magma_malloc_pinned( (void**) &result, new_size)) {      	\
		result = 0;						\
	}								\
	if ( cudaSuccess != cudaMemcpy(result, ptr, old_size, cudaMemcpyHostToHost)) {	\
		result = 0;						\
	}								\
	magma_free_pinned( ptr );					\

#define MAGMA_DEVALLOC( ptr, type, size )                                \
    if ( MAGMA_SUCCESS !=                                                \
            magma_malloc( (void**) &ptr, (size)*sizeof(type) )) {        \
        fprintf( stderr, "!!!! magma_malloc failed for: %s\n", #ptr );   \
        magma_finalize();                                                \
        exit(-1);                                                        \
    }


#define MAGMA_FREE( ptr )                                                \
    magma_free_cpu( ptr )


#define MAGMA_HOSTFREE( ptr )                                         	 \
    magma_free_pinned( ptr )


#define MAGMA_DEVFREE( ptr )                                             \
    magma_free( ptr )

namespace Eigen {

typedef std::complex<double> dcomplex;
typedef std::complex<float>  scomplex;

namespace internal {

template<typename MAGMAType, typename EigenType>
static inline void assign_scalar_eig2magma(MAGMAType& magmaScalar, const EigenType& eigenScalar) {
  magmaScalar=eigenScalar;
}

template<typename MAGMAType, typename EigenType>
static inline void assign_conj_scalar_eig2magma(MAGMAType& magmaScalar, const EigenType& eigenScalar) {
  magmaScalar=eigenScalar;
}

template <>
inline void assign_scalar_eig2magma<magmaDoubleComplex,dcomplex>(magmaDoubleComplex& magmaScalar, const dcomplex& eigenScalar) {
  magmaScalar.x=eigenScalar.real();
  magmaScalar.y=eigenScalar.imag();
}

template <>
inline void assign_scalar_eig2magma<magmaFloatComplex,scomplex>(magmaFloatComplex& magmaScalar, const scomplex& eigenScalar) {
  magmaScalar.x=eigenScalar.real();
  magmaScalar.y=eigenScalar.imag();
}

template <>
inline void assign_conj_scalar_eig2magma<magmaDoubleComplex,dcomplex>(magmaDoubleComplex& magmaScalar, const dcomplex& eigenScalar) {
  magmaScalar.x=eigenScalar.real();
  magmaScalar.y=-eigenScalar.imag();
}

template <>
inline void assign_conj_scalar_eig2magma<magmaFloatComplex,scomplex>(magmaFloatComplex& magmaScalar, const scomplex& eigenScalar) {
  magmaScalar.x=eigenScalar.real();
  magmaScalar.y=-eigenScalar.imag();
}

} // end namespace internal

} // end namespace Eigen

#endif

#endif // EIGEN_MAGMA_SUPPORT_H
