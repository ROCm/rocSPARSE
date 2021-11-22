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

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocsparse/rocsparse.h>
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
    // Query device
    int ndev;
    HIP_CHECK(hipGetDeviceCount(&ndev));

    if(ndev < 1)
    {
        std::cerr << "No HIP device found" << std::endl;
        return -1;
    }

    // Query device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "Device: " << prop.name << std::endl;

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Print rocSPARSE version and revision
    int  ver;
    char rev[64];

    ROCSPARSE_CHECK(rocsparse_get_version(handle, &ver));
    ROCSPARSE_CHECK(rocsparse_get_git_rev(handle, rev));

    std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << "-" << rev << std::endl;

    // Input data

    // Matrix A (m x k)
    // ( 1.0  2.0  0.0  3.0  0.0 )
    // ( 0.0  4.0  5.0  0.0  0.0 )
    // ( 6.0  0.0  0.0  7.0  8.0 )

    // Number of rows and columns
    rocsparse_int m = 3;
    rocsparse_int n = 2;
    rocsparse_int k = 5;

    // Number of non-zero entries
    rocsparse_int nnz_A = 8;

    // CSR row pointers
    rocsparse_int hcsr_row_ptr_A[4] = {0, 3, 5, 8};

    // CSR column indices
    rocsparse_int hcsr_col_ind_A[8] = {0, 1, 3, 1, 2, 0, 3, 4};

    // CSR values
    double hcsr_val_A[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    // Transposition of the matrix
    rocsparse_operation trans_A = rocsparse_operation_none;

    // Matrix B (k x n)
    // (  9.0  10.0 )
    // ( 11.0   0.0 )
    // (  0.0   0.0 )
    // ( 12.0  13.0 )
    // (  0.0  14.0 )

    // Number of non-zero entries
    rocsparse_int nnz_B = 6;

    // CSR row pointers
    rocsparse_int hcsr_row_ptr_B[6] = {0, 2, 3, 3, 5, 6};

    // CSR column indices
    rocsparse_int hcsr_col_ind_B[6] = {0, 1, 0, 0, 1, 1};

    // CSR values
    double hcsr_val_B[6] = {9.0, 10.0, 11.0, 12.0, 13.0, 14.0};

    // Transposition of the matrix
    rocsparse_operation trans_B = rocsparse_operation_none;

    // Scalar alpha
    double alpha = 3.7;

    // Matrix descriptor
    rocsparse_mat_descr descr_A;
    rocsparse_mat_descr descr_B;
    rocsparse_mat_descr descr_C;

    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_B));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_C));

    // Matrix C info structure
    rocsparse_mat_info info_C;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info_C));

    // Offload data to device
    rocsparse_int* dcsr_row_ptr_A;
    rocsparse_int* dcsr_col_ind_A;
    double*        dcsr_val_A;
    rocsparse_int* dcsr_row_ptr_B;
    rocsparse_int* dcsr_col_ind_B;
    double*        dcsr_val_B;
    rocsparse_int* dcsr_row_ptr_C;
    rocsparse_int* dcsr_col_ind_C;
    double*        dcsr_val_C;

    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_A, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_A, sizeof(rocsparse_int) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_A, sizeof(double) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_B, sizeof(rocsparse_int) * (k + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_B, sizeof(rocsparse_int) * nnz_B));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_B, sizeof(double) * nnz_B));
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_C, sizeof(rocsparse_int) * (m + 1)));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A, sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(double) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_B, hcsr_row_ptr_B, sizeof(rocsparse_int) * (k + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind_B, hcsr_col_ind_B, sizeof(rocsparse_int) * nnz_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val_B, hcsr_val_B, sizeof(double) * nnz_B, hipMemcpyHostToDevice));

    // Obtain required buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dcsrgemm_buffer_size(handle,
                                                   trans_A,
                                                   trans_B,
                                                   m,
                                                   n,
                                                   k,
                                                   &alpha,
                                                   descr_A,
                                                   nnz_A,
                                                   dcsr_row_ptr_A,
                                                   dcsr_col_ind_A,
                                                   descr_B,
                                                   nnz_B,
                                                   dcsr_row_ptr_B,
                                                   dcsr_col_ind_B,
                                                   NULL,
                                                   NULL,
                                                   0,
                                                   NULL,
                                                   NULL,
                                                   info_C,
                                                   &buffer_size));

    // Allocate temporary buffer
    std::cout << "Allocating " << (buffer_size >> 10) << "kB temporary storage buffer" << std::endl;

    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Obtain number of total non-zero entries in C and row pointers of C
    rocsparse_int nnz_C;

    ROCSPARSE_CHECK(rocsparse_csrgemm_nnz(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          descr_A,
                                          nnz_A,
                                          dcsr_row_ptr_A,
                                          dcsr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          dcsr_row_ptr_B,
                                          dcsr_col_ind_B,
                                          NULL,
                                          0,
                                          NULL,
                                          NULL,
                                          descr_C,
                                          dcsr_row_ptr_C,
                                          &nnz_C,
                                          info_C,
                                          temp_buffer));

    std::cout << "Matrix C contains " << nnz_C << " non-zero elements" << std::endl;

    // Compute column indices and values of C
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_C, sizeof(rocsparse_int) * nnz_C));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_C, sizeof(double) * nnz_C));

    ROCSPARSE_CHECK(rocsparse_dcsrgemm(handle,
                                       trans_A,
                                       trans_B,
                                       m,
                                       n,
                                       k,
                                       &alpha,
                                       descr_A,
                                       nnz_A,
                                       dcsr_val_A,
                                       dcsr_row_ptr_A,
                                       dcsr_col_ind_A,
                                       descr_B,
                                       nnz_B,
                                       dcsr_val_B,
                                       dcsr_row_ptr_B,
                                       dcsr_col_ind_B,
                                       NULL,
                                       NULL,
                                       0,
                                       NULL,
                                       NULL,
                                       NULL,
                                       descr_C,
                                       dcsr_val_C,
                                       dcsr_row_ptr_C,
                                       dcsr_col_ind_C,
                                       info_C,
                                       temp_buffer));

    // Print result
    std::vector<rocsparse_int> hcsr_row_ptr_C(m + 1);
    std::vector<rocsparse_int> hcsr_col_ind_C(nnz_C);
    std::vector<double>        hcsr_val_C(nnz_C);

    HIP_CHECK(hipMemcpy(hcsr_row_ptr_C.data(),
                        dcsr_row_ptr_C,
                        sizeof(rocsparse_int) * (m + 1),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsr_col_ind_C.data(),
                        dcsr_col_ind_C,
                        sizeof(rocsparse_int) * nnz_C,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(double) * nnz_C, hipMemcpyDeviceToHost));

    std::cout << "C row pointer:";

    for(int i = 0; i < m + 1; ++i)
    {
        std::cout << " " << hcsr_row_ptr_C[i];
    }

    std::cout << std::endl << "C column indices:";

    for(int i = 0; i < nnz_C; ++i)
    {
        std::cout << " " << hcsr_col_ind_C[i];
    }

    std::cout << std::endl << "C values:";

    for(int i = 0; i < nnz_C; ++i)
    {
        std::cout << " " << hcsr_val_C[i];
    }

    std::cout << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info_C));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_B));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_C));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr_A));
    HIP_CHECK(hipFree(dcsr_col_ind_A));
    HIP_CHECK(hipFree(dcsr_val_A));
    HIP_CHECK(hipFree(dcsr_row_ptr_B));
    HIP_CHECK(hipFree(dcsr_col_ind_B));
    HIP_CHECK(hipFree(dcsr_val_B));
    HIP_CHECK(hipFree(dcsr_row_ptr_C));
    HIP_CHECK(hipFree(dcsr_col_ind_C));
    HIP_CHECK(hipFree(dcsr_val_C));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
