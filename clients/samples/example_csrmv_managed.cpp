/*! \file */
/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <rocsparse/rocsparse.h>
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
    std::vector<rocsparse_int> Aptr_temp;
    std::vector<rocsparse_int> Acol_temp;
    std::vector<double>        Aval_temp;

    rocsparse_int m;
    rocsparse_int n;
    rocsparse_int nnz;

    rocsparse_init_csr_laplace2d(
        Aptr_temp, Acol_temp, Aval_temp, ndim, ndim, m, n, nnz, rocsparse_index_base_zero);

    std::vector<double> x_temp(n);
    rocsparse_init<double>(x_temp, 1, n, 1);

    rocsparse_int* Aptr = NULL;
    rocsparse_int* Acol = NULL;
    double*        Aval = NULL;
    double*        x    = NULL;
    double*        y    = NULL;

    hipMallocManaged((void**)&Aptr, sizeof(rocsparse_int) * (m + 1));
    hipMallocManaged((void**)&Acol, sizeof(rocsparse_int) * nnz);
    hipMallocManaged((void**)&Aval, sizeof(double) * nnz);
    hipMallocManaged((void**)&x, sizeof(double) * n);
    hipMallocManaged((void**)&y, sizeof(double) * m);

    // Copy data
    for(int i = 0; i < m + 1; i++)
    {
        Aptr[i] = Aptr_temp[i];
    }

    for(int i = 0; i < nnz; i++)
    {
        Acol[i] = Acol_temp[i];
        Aval[i] = Aval_temp[i];
    }

    for(int i = 0; i < n; i++)
    {
        x[i] = x_temp[i];
    }

    double alpha = 1.0f;
    double beta  = 0.0;

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse csrmv
        rocsparse_dcsrmv(handle,
                         rocsparse_operation_none,
                         m,
                         n,
                         nnz,
                         &alpha,
                         descrA,
                         Aval,
                         Aptr,
                         Acol,
                         nullptr,
                         x,
                         &beta,
                         y);
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // Call rocsparse csrmv
            rocsparse_dcsrmv(handle,
                             rocsparse_operation_none,
                             m,
                             n,
                             nnz,
                             &alpha,
                             descrA,
                             Aval,
                             Aptr,
                             Acol,
                             nullptr,
                             x,
                             &beta,
                             y);
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time             = (get_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth = static_cast<double>(sizeof(double) * (2 * m + nnz)
                                           + sizeof(rocsparse_int) * (m + 1 + nnz))
                       / time / 1e6;
    double gflops = static_cast<double>(2 * nnz) / time / 1e6;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::endl << "### rocsparse_dcsrmv WITHOUT meta data ###" << std::endl;
    std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
              << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
              << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz << std::setw(12)
              << alpha << std::setw(12) << beta << std::setw(12) << gflops << std::setw(12)
              << bandwidth << std::setw(12) << time << std::endl;

    // Create meta data
    rocsparse_mat_info info;
    rocsparse_create_mat_info(&info);

    // Analyse CSR matrix
    rocsparse_dcsrmv_analysis(
        handle, rocsparse_operation_none, m, n, nnz, descrA, Aval, Aptr, Acol, info);

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse csrmv
        rocsparse_dcsrmv(handle,
                         rocsparse_operation_none,
                         m,
                         n,
                         nnz,
                         &alpha,
                         descrA,
                         Aval,
                         Aptr,
                         Acol,
                         info,
                         x,
                         &beta,
                         y);
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    time = get_time_us();

    // CSR matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // Call rocsparse csrmv
            rocsparse_dcsrmv(handle,
                             rocsparse_operation_none,
                             m,
                             n,
                             nnz,
                             &alpha,
                             descrA,
                             Aval,
                             Aptr,
                             Acol,
                             info,
                             x,
                             &beta,
                             y);
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time      = (get_time_us() - time) / (trials * batch_size * 1e3);
    bandwidth = static_cast<double>(sizeof(double) * (2 * m + nnz)
                                    + sizeof(rocsparse_int) * (m + 1 + nnz))
                / time / 1e6;
    gflops = static_cast<double>(2 * nnz) / time / 1e6;

    std::cout << "y" << std::endl;
    for(int i = 0; i < std::min(20, m); i++)
    {
        std::cout << y[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::endl << "### rocsparse_dcsrmv WITH meta data ###" << std::endl;
    std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
              << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
              << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz << std::setw(12)
              << alpha << std::setw(12) << beta << std::setw(12) << gflops << std::setw(12)
              << bandwidth << std::setw(12) << time << std::endl;

    // Clear up on device
    hipFree(Aptr);
    hipFree(Acol);
    hipFree(Aval);
    hipFree(x);
    hipFree(y);

    rocsparse_destroy_mat_info(info);
    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(handle);

    return 0;
}
