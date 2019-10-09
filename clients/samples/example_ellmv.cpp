/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include "rocsparse_init.hpp"
#include "rocsparse_random.hpp"
#include "utility.hpp"

#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int             device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Generate problem in CSR format
    std::vector<rocsparse_int> hAptr;
    std::vector<rocsparse_int> hAcol;
    std::vector<double>        hAval;

    rocsparse_int m;
    rocsparse_int n;
    rocsparse_int nnz;

    rocsparse_init_csr_laplace2d(
        hAptr, hAcol, hAval, ndim, ndim, m, n, nnz, rocsparse_index_base_zero);

    // Sample some random data
    rocsparse_seedrand();

    double halpha = random_generator<double>();
    double hbeta  = 0.0;

    std::vector<double> hx(n);
    rocsparse_init<double>(hx, 1, n, 1);

    // Matrix descriptors
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);

    rocsparse_mat_descr descrB;
    rocsparse_create_mat_descr(&descrB);

    // Offload data to device
    rocsparse_int* dAptr = NULL;
    rocsparse_int* dAcol = NULL;
    double*        dAval = NULL;
    double*        dx    = NULL;
    double*        dy    = NULL;

    hipMalloc((void**)&dAptr, sizeof(rocsparse_int) * (m + 1));
    hipMalloc((void**)&dAcol, sizeof(rocsparse_int) * nnz);
    hipMalloc((void**)&dAval, sizeof(double) * nnz);
    hipMalloc((void**)&dx, sizeof(double) * n);
    hipMalloc((void**)&dy, sizeof(double) * m);

    hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice);

    // Convert CSR matrix to ELL format
    rocsparse_int* dBcol = NULL;
    double*        dBval = NULL;

    // Determine ELL width
    rocsparse_int ell_width;
    rocsparse_csr2ell_width(handle, m, descrA, dAptr, descrB, &ell_width);

    // Allocate memory for ELL storage format
    hipMalloc((void**)&dBcol, sizeof(rocsparse_int) * ell_width * m);
    hipMalloc((void**)&dBval, sizeof(double) * ell_width * m);

    // Convert matrix from CSR to ELL
    rocsparse_dcsr2ell(handle, m, descrA, dAval, dAptr, dAcol, descrB, ell_width, dBval, dBcol);

    // Clean up CSR structures
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse ellmv
        rocsparse_dellmv(handle,
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
                         dy);
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // ELL matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // Call rocsparse ellmv
            rocsparse_dellmv(handle,
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
                             dy);
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
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
    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_mat_descr(descrB);
    rocsparse_destroy_handle(handle);

    hipFree(dBcol);
    hipFree(dBval);
    hipFree(dx);
    hipFree(dy);

    return 0;
}
