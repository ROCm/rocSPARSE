/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <vector>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(x) ASSERT_EQ(x, hipSuccess)

template <class T>
__global__
void axpy_kernel(const T *x, T *y, T a, size_t size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size)
    {
        y[i] += a * x[i];
    }
}

TEST(Tests, Saxpy)
{
    size_t N = 100;

    float a = 100.0f;
    std::vector<float> x(N, 2.0f);
    std::vector<float> y(N, 1.0f);

    float *d_x;
    float *d_y;
    HIP_CHECK(hipMalloc(&d_x, N*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, x.data(),
                        N*sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, y.data(),
                        N*sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_kernel<float>),
                       dim3((N+255)/256), dim3(256), 0, 0,
                       d_x, d_y, a, N);
    HIP_CHECK(hipPeekAtLastError());

    HIP_CHECK(hipMemcpy(y.data(), d_y,
                        N*sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    for(size_t i=0; i<N; ++i)
    {
        EXPECT_NEAR(y[i], 201.0f, 0.1f);
    }
}

TEST(Tests, Daxpy)
{
    size_t N = 100;

    double a = 100.0f;
    std::vector<double> x(N, 2.0f);
    std::vector<double> y(N, 1.0f);

    double *d_x;
    double *d_y;
    HIP_CHECK(hipMalloc(&d_x, N*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_y, N*sizeof(double)));
    HIP_CHECK(hipMemcpy(d_x, x.data(),
                        N*sizeof(double),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, y.data(),
                        N*sizeof(double),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_kernel<double>),
                       dim3((N+255)/256), dim3(256), 0, 0,
                       d_x, d_y, a, N);
    HIP_CHECK(hipPeekAtLastError());

    HIP_CHECK(hipMemcpy(y.data(), d_y,
                        N*sizeof(double),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    for(size_t i=0; i<N; ++i)
    {
        EXPECT_NEAR(y[i], 201.0f, 0.1f);
    }
}
