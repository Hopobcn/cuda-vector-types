//
// Created by pfarre on 22/09/15.
//

#pragma once

#ifdef __CUDACC__
#define HOST               __host__
#define DEVICE             __device__
#define HOST_INLINE        __host__ __forceinline__
#define DEVICE_INLINE      __device__ __forceinline__
#define HOST_DEVICE        __host__ __device__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define HOST
#define DEVICE
#define HOST_INLINE
#define DEVICE_INLINE
#define HOST_DEVICE
#define HOST_DEVICE_INLINE
#endif
