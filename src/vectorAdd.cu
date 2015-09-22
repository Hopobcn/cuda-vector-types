//
// Created by pfarre on 22/09/15.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.hpp>
#include <iostream>

// All PTX & SASS from sm_52 - Gforce 970

template <typename T>
__global__ void
vectorAddKernel(const typename vec4<T>::VecType* __restrict__ a,
                const typename vec4<T>::VecType* __restrict__ b,
                      typename vec4<T>::VecType* __restrict__ c,
                unsigned numElements) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
                  i < numElements;
                  i += gridDim.x * blockDim.x) {
        // Expected Behaviour
        // 1.convert float4 to vec4<float>
        // 2.perform a vec4<float>+vec4<float>
        // 3.convert to float4 and store

        // Code:
        typename vec4<T>::Type x = a[i];
        typename vec4<T>::Type y = b[i];
        typename vec4<T>::Type z = x + y;
        c[i] = static_cast<typename vec4<T>::VecType>(z);

        // -----PTX [floats]-----
        // ld.global.nc.f32 	%f1, [%rd9];
        // ld.global.nc.f32 	%f2, [%rd9+4];
        // ld.global.nc.f32 	%f3, [%rd9+8];
        // ld.global.nc.f32 	%f4, [%rd9+12];
        // ld.global.nc.f32 	%f5, [%rd8];
        // ld.global.nc.f32 	%f6, [%rd8+4];
        // ld.global.nc.f32 	%f7, [%rd8+8];
        // ld.global.nc.f32 	%f8, [%rd8+12];
        // add.f32 	%f9, %f5, %f1;
        // add.f32 	%f10, %f6, %f2;
        // add.f32 	%f11, %f7, %f3;
        // add.f32 	%f12, %f8, %f4;
        // add.s64 	%rd10, %rd1, %rd7;
        // st.global.f32 	[%rd10], %f9;
        // st.global.f32 	[%rd10+4], %f10;
        // st.global.f32 	[%rd10+8], %f11;
        // st.global.f32 	[%rd10+12], %f12;


        // -----SASS [floats]-----
        // LDG.E.CI.128 R4, [R4];
        // ...
        // LDG.E.CI.128 R8, [R2];
        // ...
        // FADD R7, R11, R7;
        // FADD R6, R10, R6;
        // FADD R5, R9, R5;
        // FADD R4, R8, R4;
        // STG.E.128 [R2], R4;

        // -----PTX [doubles]-----
        // 	ld.global.f64 	%fd1, [%rd9+24];
        // ld.global.f64 	%fd2, [%rd9+16];
        // ld.global.f64 	%fd3, [%rd9+8];
        // ld.global.f64 	%fd4, [%rd9];
        // ld.global.f64 	%fd5, [%rd8+24];
        // ld.global.f64 	%fd6, [%rd8+16];
        // ld.global.f64 	%fd7, [%rd8+8];
        // ld.global.f64 	%fd8, [%rd8];
        // add.f64 	%fd9, %fd8, %fd4;
        // add.f64 	%fd10, %fd7, %fd3;
        // add.f64 	%fd11, %fd6, %fd2;
        // add.f64 	%fd12, %fd5, %fd1;
        // add.s64 	%rd10, %rd1, %rd7;
        // st.global.f64 	[%rd10], %fd9;
        // st.global.f64 	[%rd10+8], %fd10;
        // st.global.f64 	[%rd10+16], %fd11;
        // st.global.f64 	[%rd10+24], %fd12;


        // -----SASS [doubles]-----
        // LDG.E.128 R4, [R20];
        // ...
        // LDG.E.128 R16, [R20+0x10];
        // ...
        // LDG.E.128 R12, [R2];
        // ...
        // LDG.E.128 R8, [R2+0x10];
        // ...
        // DADD R6, R6, R14;
        // DADD R4, R4, R12;
        // DADD R10, R18, R10;
        // STG.E.128 [R2], R4;
        // DADD R8, R16, R8;
        // STG.E.128 [R2+0x10], R8;

    }
}

template <typename T>
__global__ void
vectorAddKernel(const typename vec4<T>::Type* __restrict__ a,
                const typename vec4<T>::Type* __restrict__ b,
                typename vec4<T>::Type* __restrict__ c,
                unsigned numElements) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numElements;
         i += gridDim.x * blockDim.x) {
        // Expected Behaviour
        // 1. load two vec4<float>
        // 2. perform a vec4<float>+vec4<float>
        // 3. store

        // Code:
        c[i] = a[i] + b[i];

        // -----PTX [floats]-----
        // ld.global.nc.v4.f32 {%f1, %f2,  %f3,  %f4},      -- 128bit
        // ld.global.nc.v4.f32 {%f9, %f10, %f11, %f12},     -- 128bit
        // add.f32 %f17, %f12, %f4;
        // add.f32 %f18, %f11, %f3;
        // add.f32 %f19, %f10, %f2;
        // add.f32 %f20, %f09, %f1;
        // st.global.v4.f32 {%f20, %f19, %f18, %f17         -- 128bit

        // -----SASS [floats]-----
        // LDG.E.CI R11, [R2];
        // ...
        // LDG.E.CI R10, [R2+0x4];
        // ...
        // LDG.E.CI R13, [R2+0x8];
        // ...
        // LDG.E.CI R6, [R4];
        // ...
        // LDG.E.CI R7, [R4+0x4];
        // ...
        // LDG.E.CI R8, [R4+0x8];
        // ...
        // LDG.E.CI R9, [R4+0xc];
        // ...
        // LDG.E.CI R12, [R2+0xc];
        // ...
        // FADD R11, R6, R11;
        // FADD R7, R7, R10;
        // STG.E [R2], R11;
        // ...
        // FADD R4, R8, R13;
        // ...
        // FADD R5, R9, R12;
        // STG.E [R2+0x8], R4;
        // STG.E [R2+0xc], R5;

        // -----PTX [doubles]-----
        // ld.global.v2.f64 	{%fd1, %fd2},               -- 128bit
        // ld.global.v2.f64 	{%fd5, %fd6},               -- 128bit
        // ld.global.v2.f64 	{%fd9, %fd10},              -- 128bit
        // ld.global.v2.f64 	{%fd13, %fd14},             -- 128bit
        // add.f64 	           %fd17, %fd14, %fd6;
        // add.f64 	           %fd18, %fd13, %fd5;
        // st.global.v2.f64 	{%fd18, %fd17};             -- 128bit
        // add.f64 	           %fd19, %fd10, %fd2;
        // add.f64 	           %fd20, %fd9, %fd1;
        // st.global.v2.f64 	{%fd20, %fd19};             -- 128bit

        // -----SASS [doubles]-----
        // LDG.E.64 R8, [R4+0x10];
        // ...
        // LDG.E.64 R6, [R4+0x18];
        // ...
        // LDG.E.64 R12, [R4];
        // ...
        // LDG.E.64 R14, [R2+0x10];
        // ...
        // LDG.E.64 R16, [R2+0x18];
        // ...
        // LDG.E.64 R18, [R2];
        // ...
        // LDG.E.64 R10, [R4+0x8];
        // ...
        // LDG.E.64 R20, [R2+0x8];
        // ...
        // DADD R2, R8, R14;
        // DADD R16, R6, R16;
        // ...
        // DADD R4, R12, R18;
        // DADD R6, R10, R20;
        // STG.E.64 [R8], R4;
        // STG.E.64 [R8+0x8], R6;
        // STG.E.64 [R8+0x10], R2;
        // STG.E.64 [R8+0x18], R16;

    }
}

template <typename T, int blockSize>
void
vectorAdd(const typename vec4<T>::VecType* a,
          const typename vec4<T>::VecType* b,
                typename vec4<T>::VecType* c,
          unsigned numElements) {

    std::cout << "--VecType numElements: " << numElements << std::endl;

    dim3 dimGrid {(numElements + blockSize - 1)/ blockSize, 1, 1};
    dim3 dimBlock{blockSize, 1, 1};

    vectorAddKernel<T><<<dimGrid,dimBlock>>>(a,b,c,numElements);
    cudaDeviceSynchronize();
}

template <typename T, int blockSize>
void
vectorAdd(const typename vec4<T>::Type* a,
          const typename vec4<T>::Type* b,
          typename vec4<T>::Type* c,
          unsigned numElements) {

    std::cout << "--Type numElements: " << numElements << std::endl;

    dim3 dimGrid {(numElements + blockSize - 1)/ blockSize, 1, 1};
    dim3 dimBlock{blockSize, 1, 1};

    vectorAddKernel<T><<<dimGrid,dimBlock>>>(a,b,c,numElements);
    cudaDeviceSynchronize();
}

template <typename T>
void launchTest(unsigned numElements, unsigned repetitions) {

    std::cout << "--launchTest numElements: " << numElements << std::endl;

    T* a_d;
    T* b_d;
    T* c_d;

    cudaMalloc(&a_d, numElements * sizeof(T));
    cudaMalloc(&b_d, numElements * sizeof(T));
    cudaMalloc(&c_d, numElements * sizeof(T));

    for (unsigned i = 0; i < repetitions; ++i)
        vectorAdd<T,256>((typename vec4<T>::VecType *)a_d,
                         (typename vec4<T>::VecType *)b_d,
                         (typename vec4<T>::VecType *)c_d,
                          numElements/4);

    for (unsigned i = 0; i < repetitions; ++i)
        vectorAdd<T,256>((typename vec4<T>::Type *)a_d,
                         (typename vec4<T>::Type *)b_d,
                         (typename vec4<T>::Type *)c_d,
                         numElements/4);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void launchAllTests(unsigned numElements, unsigned repetitions) {
    launchTest<float>(numElements, repetitions);
    launchTest<double>(numElements, repetitions);
}