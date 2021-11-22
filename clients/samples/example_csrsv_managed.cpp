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
    rocsparse_create_handle(&handle);

    // Transposition of the matrix
    rocsparse_operation trans = rocsparse_operation_none;

    // Analysis policy
    rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;

    // Solve policy
    rocsparse_solve_policy solve_policy = rocsparse_solve_policy_auto;

    hipDeviceProp_t devProp;
    int             device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Generate problem
    std::vector<rocsparse_int> csr_row_ptr_temp;
    std::vector<rocsparse_int> csr_col_ind_temp;
    std::vector<double>        csr_val_temp;

    rocsparse_int m;
    rocsparse_int nnz;
    double        alpha = 1.0f;

    rocsparse_init_csr_laplace2d(csr_row_ptr_temp,
                                 csr_col_ind_temp,
                                 csr_val_temp,
                                 ndim,
                                 ndim,
                                 m,
                                 m,
                                 nnz,
                                 rocsparse_index_base_zero);

    std::vector<double> x_temp(m);
    rocsparse_init<double>(x_temp, 1, m, 1);

    rocsparse_int* csr_row_ptr = NULL;
    rocsparse_int* csr_col_ind = NULL;
    double*        csr_val     = NULL;
    double*        x           = NULL;
    double*        y           = NULL;

    hipMallocManaged((void**)&csr_row_ptr, sizeof(rocsparse_int) * (m + 1));
    hipMallocManaged((void**)&csr_col_ind, sizeof(rocsparse_int) * nnz);
    hipMallocManaged((void**)&csr_val, sizeof(double) * nnz);
    hipMallocManaged((void**)&x, sizeof(double) * m);
    hipMallocManaged((void**)&y, sizeof(double) * m);

    // Copy data
    for(int i = 0; i < m + 1; i++)
    {
        csr_row_ptr[i] = csr_row_ptr_temp[i];
    }

    for(int i = 0; i < nnz; i++)
    {
        csr_col_ind[i] = csr_col_ind_temp[i];
        csr_val[i]     = csr_val_temp[i];
    }

    for(int i = 0; i < m; i++)
    {
        x[i] = x_temp[i];
    }

    // Matrix descriptor
    rocsparse_mat_descr descr;
    rocsparse_create_mat_descr(&descr);

    // Matrix fill mode
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower));

    // Matrix diagonal type
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit));

    // Matrix info structure
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Obtain required buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, &buffer_size));

    // Allocate temporary buffer
    std::cout << "Allocating " << (buffer_size >> 10) << "kB temporary storage buffer" << std::endl;

    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Perform analysis step
    ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle,
                                              trans,
                                              m,
                                              nnz,
                                              descr,
                                              csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              analysis_policy,
                                              solve_policy,
                                              temp_buffer));

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle,
                                               trans,
                                               m,
                                               nnz,
                                               &alpha,
                                               descr,
                                               csr_val,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               info,
                                               x,
                                               y,
                                               solve_policy,
                                               temp_buffer));
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // Call dcsrsv to perform lower triangular solve Ly = x
    for(int i = 0; i < trials; ++i)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle,
                                                   trans,
                                                   m,
                                                   nnz,
                                                   &alpha,
                                                   descr,
                                                   csr_val,
                                                   csr_row_ptr,
                                                   csr_col_ind,
                                                   info,
                                                   x,
                                                   y,
                                                   solve_policy,
                                                   temp_buffer));

            // Device synchronization
            hipDeviceSynchronize();
        }
    }

    double solve_time = (get_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth  = ((m + 1 + nnz) * sizeof(rocsparse_int) + (m + m + nnz) * sizeof(double))
                       / solve_time / 1e6;

    // Check for zero pivots
    rocsparse_int    pivot;
    rocsparse_status status = rocsparse_csrsv_zero_pivot(handle, descr, info, &pivot);

    if(status == rocsparse_status_zero_pivot)
    {
        std::cout << "WARNING: Found zero pivot in matrix row " << pivot << std::endl;
    }

    // Device synchronization
    hipDeviceSynchronize();

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::endl << "### rocsparse_dcsrsv ###" << std::endl;
    std::cout << std::setw(12) << "m" << std::setw(12) << "nnz" << std::setw(12) << "alpha"
              << std::setw(12) << "GB/s" << std::setw(12) << "solve msec" << std::setw(12)
              << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << nnz << std::setw(12) << alpha
              << std::setw(12) << bandwidth << std::setw(12) << solve_time << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(csr_row_ptr));
    HIP_CHECK(hipFree(csr_col_ind));
    HIP_CHECK(hipFree(csr_val));
    HIP_CHECK(hipFree(x));
    HIP_CHECK(hipFree(y));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
