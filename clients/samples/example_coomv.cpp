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

    // Generate problem
    std::vector<rocsparse_int> hArow;
    std::vector<rocsparse_int> hAcol;
    std::vector<double>        hAval;

    rocsparse_int m;
    rocsparse_int n;
    rocsparse_int nnz;

    rocsparse_init_coo_laplace2d(
        hArow, hAcol, hAval, ndim, ndim, m, n, nnz, rocsparse_index_base_zero);

    // Sample some random data
    rocsparse_seedrand();

    double halpha = random_generator<double>();
    double hbeta  = 0.0;

    std::vector<double> hx(n);
    rocsparse_init<double>(hx, 1, n, 1);

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);

    // Offload data to device
    rocsparse_int* dArow = NULL;
    rocsparse_int* dAcol = NULL;
    double*        dAval = NULL;
    double*        dx    = NULL;
    double*        dy    = NULL;

    hipMalloc((void**)&dArow, sizeof(rocsparse_int) * nnz);
    hipMalloc((void**)&dAcol, sizeof(rocsparse_int) * nnz);
    hipMalloc((void**)&dAval, sizeof(double) * nnz);
    hipMalloc((void**)&dx, sizeof(double) * n);
    hipMalloc((void**)&dy, sizeof(double) * m);

    hipMemcpy(dArow, hArow.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice);

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse coomv
        rocsparse_dcoomv(handle,
                         rocsparse_operation_none,
                         m,
                         n,
                         nnz,
                         &halpha,
                         descrA,
                         dAval,
                         dArow,
                         dAcol,
                         dx,
                         &hbeta,
                         dy);
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // COO matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // Call rocsparse coomv
            rocsparse_dcoomv(handle,
                             rocsparse_operation_none,
                             m,
                             n,
                             nnz,
                             &halpha,
                             descrA,
                             dAval,
                             dArow,
                             dAcol,
                             dx,
                             &hbeta,
                             dy);
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth
        = static_cast<double>(sizeof(double) * (4 * m + nnz) + sizeof(rocsparse_int) * (2 * nnz))
          / time / 1e6;
    double gflops = static_cast<double>(3 * nnz) / time / 1e6;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
              << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
              << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz << std::setw(12)
              << halpha << std::setw(12) << hbeta << std::setw(12) << gflops << std::setw(12)
              << bandwidth << std::setw(12) << time << std::endl;

    // Clear up on device
    hipFree(dArow);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dx);
    hipFree(dy);

    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(handle);

    return 0;
}
