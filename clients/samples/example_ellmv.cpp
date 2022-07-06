/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "utils.hpp"

#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
            return -1;                                                         \
        }                                                                      \
    }

#define ROCSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if(stat != rocsparse_status_success)                                         \
        {                                                                            \
            std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
            return -1;                                                               \
        }                                                                            \
    }

int main(int argc, char* argv[])
{
    // Parse command line
    if(argc < 2)
    {
        std::cerr << argv[0] << " <ndim> [<trials> <batch_size>]" << std::endl;
        return -1;
    }

    rocsparse_int ndim       = atoi(argv[1]);
    int           trials     = 200;
    int           batch_size = 1;

    if(argc > 2)
    {
        trials = atoi(argv[2]);
    }
    if(argc > 3)
    {
        batch_size = atoi(argv[3]);
    }

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    hipDeviceProp_t devProp;
    int             device_id = 0;

    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "Device: " << devProp.name << std::endl;

    // Generate problem in CSR format
    std::vector<rocsparse_int> hAptr;
    std::vector<rocsparse_int> hAcol;
    std::vector<double>        hAval;

    rocsparse_int m;
    rocsparse_int n;
    rocsparse_int nnz;

    utils_init_csr_laplace2d(hAptr, hAcol, hAval, ndim, ndim, m, n, nnz, rocsparse_index_base_zero);

    // Sample some random data
    utils_seedrand();

    double halpha = utils_random<double>();
    double hbeta  = 0.0;

    std::vector<double> hx(n);
    utils_init<double>(hx, 1, n, 1);

    // Matrix descriptors
    rocsparse_mat_descr descrA;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrA));

    rocsparse_mat_descr descrB;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrB));

    // Offload data to device
    rocsparse_int* dAptr = NULL;
    rocsparse_int* dAcol = NULL;
    double*        dAval = NULL;
    double*        dx    = NULL;
    double*        dy    = NULL;

    HIP_CHECK(hipMalloc((void**)&dAptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dAcol, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dAval, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * n));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * m));

    HIP_CHECK(
        hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice));

    // Convert CSR matrix to ELL format
    rocsparse_int* dBcol = NULL;
    double*        dBval = NULL;

    // Determine ELL width
    rocsparse_int ell_width;
    ROCSPARSE_CHECK(rocsparse_csr2ell_width(handle, m, descrA, dAptr, descrB, &ell_width));

    // Allocate memory for ELL storage format
    HIP_CHECK(hipMalloc((void**)&dBcol, sizeof(rocsparse_int) * ell_width * m));
    HIP_CHECK(hipMalloc((void**)&dBval, sizeof(double) * ell_width * m));

    // Convert matrix from CSR to ELL
    ROCSPARSE_CHECK(rocsparse_dcsr2ell(
        handle, m, descrA, dAval, dAptr, dAcol, descrB, ell_width, dBval, dBcol));

    // Clean up CSR structures
    HIP_CHECK(hipFree(dAptr));
    HIP_CHECK(hipFree(dAcol));
    HIP_CHECK(hipFree(dAval));

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse ellmv
        ROCSPARSE_CHECK(rocsparse_dellmv(handle,
                                         rocsparse_operation_none,
                                         m,
                                         n,
                                         &halpha,
                                         descrB,
                                         dBval,
                                         dBcol,
                                         ell_width,
                                         dx,
                                         &hbeta,
                                         dy));
    }

    // Device synchronization
    HIP_CHECK(hipDeviceSynchronize());

    // Start time measurement
    double time = utils_time_us();

    // ELL matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int j = 0; j < batch_size; ++j)
        {
            // Call rocsparse ellmv
            ROCSPARSE_CHECK(rocsparse_dellmv(handle,
                                             rocsparse_operation_none,
                                             m,
                                             n,
                                             &halpha,
                                             descrB,
                                             dBval,
                                             dBcol,
                                             ell_width,
                                             dx,
                                             &hbeta,
                                             dy));
        }

        // Device synchronization
        HIP_CHECK(hipDeviceSynchronize());
    }

    time = (utils_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth
        = static_cast<double>(sizeof(double) * (2 * m + nnz) + sizeof(rocsparse_int) * (nnz)) / time
          / 1e6;
    double gflops = static_cast<double>(2 * nnz) / time / 1e6;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
              << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
              << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << ell_width * m
              << std::setw(12) << halpha << std::setw(12) << hbeta << std::setw(12) << gflops
              << std::setw(12) << bandwidth << std::setw(12) << time << std::endl;

    // Clear up on device
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrA));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrB));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    HIP_CHECK(hipFree(dBcol));
    HIP_CHECK(hipFree(dBval));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));

    return 0;
}
