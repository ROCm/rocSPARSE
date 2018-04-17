/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "test_utils.h"

#include <stdlib.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocsparse.h>

#define HIP_CHECK(x) ASSERT_EQ(x, hipSuccess)
#define ROCSPARSE_CHECK(x) ASSERT_EQ(x, ROCSPARSE_STATUS_SUCCESS)

TEST(Tests, rocsparseScsrmv)
{
    rocsparseHandle_t handle;
    ROCSPARSE_CHECK(rocsparseCreate(&handle));

    // Generate problem
    int *Aptr = NULL;
    int *Acol = NULL;
    float *Aval = NULL;
    int nrow = gen2DLaplacianUS(2000, &Aptr, &Acol, &Aval);
    int nnz = Aptr[nrow];

    // Sample some random data
    srand(12345ULL);

    float alpha = (float) rand() / RAND_MAX;
    float beta = (float) rand() / RAND_MAX;

    float *x = (float*) malloc(sizeof(float)*nrow);
    float *y = (float*) malloc(sizeof(float)*nrow);
    for (int i=0; i<nrow; ++i)
    {
        x[i] = (float) rand() / RAND_MAX;
        y[i] = 0.0f;
    }

    // Matrix descriptor
    rocsparseMatDescr_t descrA;
    ROCSPARSE_CHECK(rocsparseCreateMatDescr(&descrA));

    // Offload data to device
    int *dAptr = NULL;
    int *dAcol = NULL;
    float *dAval = NULL;
    float *dx = NULL;
    float *dy = NULL;

    HIP_CHECK(hipMalloc((void**) &dAptr, sizeof(int)*(nrow+1)));
    HIP_CHECK(hipMalloc((void**) &dAcol, sizeof(int)*nnz));
    HIP_CHECK(hipMalloc((void**) &dAval, sizeof(float)*nnz));
    HIP_CHECK(hipMalloc((void**) &dx, sizeof(float)*nrow));
    HIP_CHECK(hipMalloc((void**) &dy, sizeof(float)*nrow));

    HIP_CHECK(hipMemcpy(dAptr, Aptr, sizeof(int)*(nrow+1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, Acol, sizeof(int)*nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, Aval, sizeof(float)*nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x, sizeof(float)*nrow, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, y, sizeof(float)*nrow, hipMemcpyHostToDevice));

//TODO analyse step
    ROCSPARSE_CHECK(rocsparseScsrmv(handle, ROCSPARSE_OPERATION_NON_TRANSPOSE,
                                    nrow, nrow, nnz, &alpha, descrA, dAval,
                                    dAptr, dAcol, dx, &beta, dy));

    // Copy result to host
    float *result = (float*) malloc(sizeof(float)*nrow);
    HIP_CHECK(hipMemcpy(result, dy, sizeof(float)*nrow, hipMemcpyDeviceToHost));

    // Check if result is correct
    for (int i=0; i<nrow; ++i)
    {
        float sum = beta * y[i];
        for (int j=Aptr[i]; j<Aptr[i+1]; ++j)
        {
            sum += alpha * Aval[j] * x[Acol[j]];
        }
        float eps = std::max(fabs(sum)*1e-4f, 1e-7);
        ASSERT_NEAR(result[i], sum, eps);
    }


    ROCSPARSE_CHECK(rocsparseDestroy(handle));
}

TEST(Tests, rocsparseDcsrmv)
{
    rocsparseHandle_t handle;
    ROCSPARSE_CHECK(rocsparseCreate(&handle));

    // Generate problem
    int *Aptr = NULL;
    int *Acol = NULL;
    double *Aval = NULL;
    int nrow = gen2DLaplacianUS(2000, &Aptr, &Acol, &Aval);
    int nnz = Aptr[nrow];

    // Sample some random data
    srand(12345ULL);

    double alpha = (double) rand() / RAND_MAX;
    double beta = (double) rand() / RAND_MAX;

    double *x = (double*) malloc(sizeof(double)*nrow);
    double *y = (double*) malloc(sizeof(double)*nrow);
    for (int i=0; i<nrow; ++i)
    {
        x[i] = (double) rand() / RAND_MAX;
        y[i] = 0.0;
    }

    // Matrix descriptor
    rocsparseMatDescr_t descrA;
    ROCSPARSE_CHECK(rocsparseCreateMatDescr(&descrA));

    // Offload data to device
    int *dAptr = NULL;
    int *dAcol = NULL;
    double *dAval = NULL;
    double *dx = NULL;
    double *dy = NULL;

    HIP_CHECK(hipMalloc((void**) &dAptr, sizeof(int)*(nrow+1)));
    HIP_CHECK(hipMalloc((void**) &dAcol, sizeof(int)*nnz));
    HIP_CHECK(hipMalloc((void**) &dAval, sizeof(double)*nnz));
    HIP_CHECK(hipMalloc((void**) &dx, sizeof(double)*nrow));
    HIP_CHECK(hipMalloc((void**) &dy, sizeof(double)*nrow));

    HIP_CHECK(hipMemcpy(dAptr, Aptr, sizeof(int)*(nrow+1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, Acol, sizeof(int)*nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, Aval, sizeof(double)*nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x, sizeof(double)*nrow, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, y, sizeof(double)*nrow, hipMemcpyHostToDevice));


//TODO analyse step

    ROCSPARSE_CHECK(rocsparseDcsrmv(handle, ROCSPARSE_OPERATION_NON_TRANSPOSE,
                                    nrow, nrow, nnz, &alpha, descrA, dAval,
                                    dAptr, dAcol, dx, &beta, dy));

    // Copy result to host
    double *result = (double*) malloc(sizeof(double)*nrow);
    HIP_CHECK(hipMemcpy(result, dy, sizeof(double)*nrow, hipMemcpyDeviceToHost));

    // Check if result is correct
    for (int i=0; i<nrow; ++i)
    {
        double sum = beta * y[i];
        for (int j=Aptr[i]; j<Aptr[i+1]; ++j)
        {
            sum += alpha * Aval[j] * x[Acol[j]];
        }
        double eps = std::max(fabs(sum) * 1e-8, 1e-15);
        ASSERT_NEAR(result[i], sum, eps);
    }


    ROCSPARSE_CHECK(rocsparseDestroy(handle));
}
