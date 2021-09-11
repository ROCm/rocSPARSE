/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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

#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

template <typename T>
void testing_csrgemm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    T h_alpha = 0.6;
    T h_beta  = 0.2;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptors
    rocsparse_local_mat_descr local_descr_A;
    rocsparse_local_mat_descr local_descr_B;
    rocsparse_local_mat_descr local_descr_C;
    rocsparse_local_mat_descr local_descr_D;

    // Create info desciptor
    rocsparse_local_mat_info local_info_C;

    rocsparse_handle    handle      = local_handle;
    rocsparse_operation trans_A     = rocsparse_operation_none;
    rocsparse_operation trans_B     = rocsparse_operation_none;
    rocsparse_int       m           = safe_size;
    rocsparse_int       n           = safe_size;
    rocsparse_int       k           = safe_size;
    rocsparse_mat_info  info_C      = local_info_C;
    size_t*             buffer_size = (size_t*)0x4;
    void*               temp_buffer = (void*)0x4;

    // C matrix
    const rocsparse_mat_descr descr_C       = local_descr_C;
    T*                        csr_val_C     = (T*)0x4;
    rocsparse_int*            csr_row_ptr_C = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind_C = (rocsparse_int*)0x4;
    rocsparse_int*            nnz_C         = (rocsparse_int*)0x4;

#define PARAMS_BUFFER_SIZE                                                                  \
    handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, \
        descr_B, nnz_B, csr_row_ptr_B, csr_col_ind_B, beta, descr_D, nnz_D, csr_row_ptr_D,  \
        csr_col_ind_D, info_C, buffer_size

#define PARAMS_NNZ                                                                            \
    handle, trans_A, trans_B, m, n, k, descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, descr_B, \
        nnz_B, csr_row_ptr_B, csr_col_ind_B, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D,    \
        descr_C, csr_row_ptr_C, nnz_C, info_C, temp_buffer

#define PARAMS                                                                                 \
    handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_val_A, csr_row_ptr_A,        \
        csr_col_ind_A, descr_B, nnz_B, csr_val_B, csr_row_ptr_B, csr_col_ind_B, beta, descr_D, \
        nnz_D, csr_val_D, csr_row_ptr_D, csr_col_ind_D, descr_C, csr_val_C, csr_row_ptr_C,     \
        csr_col_ind_C, info_C, temp_buffer

    // 4 Scenarios need to be tested:

    // Scenario 1: alpha == nullptr && beta == nullptr
    // Scenario 2: alpha != nullptr && beta == nullptr
    // Scenario 3: alpha == nullptr && beta != nullptr
    // Scenario 4: alpha != nullptr && beta != nullptr

    // ###############################################
    // Scenario 1: alpha == nullptr && beta == nullptr
    // ###############################################
    {
        // In this scenario matrices A == B == D == nullptr
        int nargs_to_exclude_buffer_size = 14;
        int nargs_to_exclude_nnz         = 12;
        int nargs_to_exclude             = 17;

        const int args_to_exclude_buffer_size[14]
            = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        const int args_to_exclude_nnz[12] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        const int args_to_exclude[17]
            = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

        const T* alpha = (const T*)nullptr;
        const T* beta  = (const T*)nullptr;

        // A matrix
        const rocsparse_mat_descr descr_A       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_A         = 0;
        const T*                  csr_val_A     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)nullptr;

        // B matrix
        const rocsparse_mat_descr descr_B       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_B         = 0;
        const T*                  csr_val_B     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)nullptr;

        // D matrix
        const rocsparse_mat_descr descr_D       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_D         = 0;
        const T*                  csr_val_D     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_D = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_D = (const rocsparse_int*)nullptr;

        auto_testing_bad_arg(rocsparse_csrgemm_buffer_size<T>,
                             nargs_to_exclude_buffer_size,
                             args_to_exclude_buffer_size,
                             PARAMS_BUFFER_SIZE);
        auto_testing_bad_arg(
            rocsparse_csrgemm_nnz, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);
        auto_testing_bad_arg(rocsparse_csrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 2: alpha != nullptr && beta == nullptr
    // ###############################################
    {
        // In this scenario matrices A != B != nullptr and D == nullptr
        int nargs_to_exclude_buffer_size = 5;
        int nargs_to_exclude_nnz         = 4;
        int nargs_to_exclude             = 6;

        const int args_to_exclude_buffer_size[5] = {15, 16, 17, 18, 19};
        const int args_to_exclude_nnz[4]         = {14, 15, 16, 17};
        const int args_to_exclude[6]             = {17, 18, 19, 20, 21, 22};

        const T* alpha = &h_alpha;
        const T* beta  = (const T*)nullptr;

        // A matrix
        const rocsparse_mat_descr descr_A       = local_descr_A;
        rocsparse_int             nnz_A         = safe_size;
        const T*                  csr_val_A     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)0x4;

        // B matrix
        const rocsparse_mat_descr descr_B       = local_descr_B;
        rocsparse_int             nnz_B         = safe_size;
        const T*                  csr_val_B     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)0x4;

        // D matrix
        const rocsparse_mat_descr descr_D       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_D         = 0;
        const T*                  csr_val_D     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_D = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_D = (const rocsparse_int*)nullptr;

        auto_testing_bad_arg(rocsparse_csrgemm_buffer_size<T>,
                             nargs_to_exclude_buffer_size,
                             args_to_exclude_buffer_size,
                             PARAMS_BUFFER_SIZE);
        auto_testing_bad_arg(
            rocsparse_csrgemm_nnz, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);
        auto_testing_bad_arg(rocsparse_csrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 3: alpha == nullptr && beta != nullptr
    // ###############################################
    {
        // In this scenario matrices A == B == nullptr and D != nullptr
        int nargs_to_exclude_buffer_size = 9;
        int nargs_to_exclude_nnz         = 8;
        int nargs_to_exclude             = 11;

        const int args_to_exclude_buffer_size[9] = {6, 7, 8, 9, 10, 11, 12, 13, 14};
        const int args_to_exclude_nnz[8]         = {6, 7, 8, 9, 10, 11, 12, 13};
        const int args_to_exclude[11]            = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

        const T* alpha = (const T*)nullptr;
        const T* beta  = &h_beta;

        // A matrix
        const rocsparse_mat_descr descr_A       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_A         = 0;
        const T*                  csr_val_A     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)nullptr;

        // B matrix
        const rocsparse_mat_descr descr_B       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_B         = 0;
        const T*                  csr_val_B     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)nullptr;

        // D matrix
        const rocsparse_mat_descr descr_D       = local_descr_D;
        rocsparse_int             nnz_D         = safe_size;
        const T*                  csr_val_D     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_D = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_D = (const rocsparse_int*)0x4;

        auto_testing_bad_arg(rocsparse_csrgemm_buffer_size<T>,
                             nargs_to_exclude_buffer_size,
                             args_to_exclude_buffer_size,
                             PARAMS_BUFFER_SIZE);
        auto_testing_bad_arg(
            rocsparse_csrgemm_nnz, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);
        auto_testing_bad_arg(rocsparse_csrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 4: alpha != nullptr && beta != nullptr
    // ###############################################
    {
        // In this scenario matrices A != B != D != nullptr
        int nargs_to_exclude_buffer_size = 2;
        int nargs_to_exclude             = 2;

        const int args_to_exclude_buffer_size[2] = {6, 15};
        const int args_to_exclude[2]             = {6, 17};

        const T* alpha = &h_alpha;
        const T* beta  = &h_beta;

        // A matrix
        const rocsparse_mat_descr descr_A       = local_descr_A;
        rocsparse_int             nnz_A         = safe_size;
        const T*                  csr_val_A     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)0x4;

        // B matrix
        const rocsparse_mat_descr descr_B       = local_descr_B;
        rocsparse_int             nnz_B         = safe_size;
        const T*                  csr_val_B     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)0x4;

        // D matrix
        const rocsparse_mat_descr descr_D       = local_descr_D;
        rocsparse_int             nnz_D         = safe_size;
        const T*                  csr_val_D     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_D = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_D = (const rocsparse_int*)0x4;

        auto_testing_bad_arg(rocsparse_csrgemm_buffer_size<T>,
                             nargs_to_exclude_buffer_size,
                             args_to_exclude_buffer_size,
                             PARAMS_BUFFER_SIZE);
        auto_testing_bad_arg(rocsparse_csrgemm_nnz, PARAMS_NNZ);
        auto_testing_bad_arg(rocsparse_csrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

#undef PARAMS
#undef PARAMS_NNZ
#undef PARAMS_BUFFER_SIZE
    
    
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_transpose,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             &h_beta,
                                                             descrD,
                                                             safe_size,
                                                             dcsr_row_ptr_D,
                                                             dcsr_col_ind_D,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_transpose,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             &h_beta,
                                                             descrD,
                                                             safe_size,
                                                             dcsr_row_ptr_D,
                                                             dcsr_col_ind_D,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_transpose,
                                                             rocsparse_operation_transpose,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             &h_beta,
                                                             descrD,
                                                             safe_size,
                                                             dcsr_row_ptr_D,
                                                             dcsr_col_ind_D,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_transpose,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_transpose,
                                                  rocsparse_operation_transpose,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_transpose,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_transpose,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_transpose,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_transpose,
                                                 rocsparse_operation_transpose,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_not_implemented);
}

template <typename T>
void testing_csrgemm(const Arguments& arg)
{
    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_operation   transA    = arg.transA;
    rocsparse_operation   transB    = arg.transB;
    rocsparse_index_base  baseA     = arg.baseA;
    rocsparse_index_base  baseB     = arg.baseB;
    rocsparse_index_base  baseC     = arg.baseC;
    rocsparse_index_base  baseD     = arg.baseD;
    static constexpr bool full_rank = false;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    T* halpha_ptr = nullptr;
    T* hbeta_ptr  = nullptr;
    T* dalpha_ptr = nullptr;
    T* dbeta_ptr  = nullptr;

    // 4 Scenarios need to be tested:

    // Scenario 1: alpha == nullptr && beta == nullptr
    // Scenario 2: alpha != nullptr && beta == nullptr
    // Scenario 3: alpha == nullptr && beta != nullptr
    // Scenario 4: alpha != nullptr && beta != nullptr

    // alpha == -99 means test for alpha == nullptr
    // beta  == -99 means test for beta == nullptr
    int scenario;
    if(h_alpha == static_cast<T>(-99) && h_beta == static_cast<T>(-99))
    {
        scenario = 1;
    }
    else if(h_alpha != static_cast<T>(-99) && h_beta == static_cast<T>(-99))
    {
        scenario   = 2;
        halpha_ptr = &h_alpha;
    }
    else if(h_alpha == static_cast<T>(-99) && h_beta != static_cast<T>(-99))
    {
        scenario  = 3;
        hbeta_ptr = &h_beta;
    }
    else if(h_alpha != static_cast<T>(-99) && h_beta != static_cast<T>(-99))
    {
        scenario   = 4;
        halpha_ptr = &h_alpha;
        hbeta_ptr  = &h_beta;
    }
    else
    {
        return;
    }

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descrA;
    rocsparse_local_mat_descr descrB;
    rocsparse_local_mat_descr descrC;
    rocsparse_local_mat_descr descrD;

    // Create matrix info for C
    rocsparse_local_mat_info info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrC, baseC));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrD, baseD));

    // Argument sanity check before allocating invalid memory
    if((M <= 0 || N <= 0 || K <= 0) || scenario == 1)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr_A(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_A(safe_size);
        device_vector<T>             dcsr_val_A(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_B(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_B(safe_size);
        device_vector<T>             dcsr_val_B(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_C(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_C(safe_size);
        device_vector<T>             dcsr_val_C(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_D(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_D(safe_size);
        device_vector<T>             dcsr_val_D(safe_size);
        device_vector<T>             dbuffer(safe_size);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        size_t        buffer_size;
        rocsparse_int nnz_C;

        rocsparse_status status_1 = rocsparse_csrgemm_buffer_size<T>(handle,
                                                                     transA,
                                                                     transB,
                                                                     M,
                                                                     N,
                                                                     K,
                                                                     halpha_ptr,
                                                                     descrA,
                                                                     safe_size,
                                                                     dcsr_row_ptr_A,
                                                                     dcsr_col_ind_A,
                                                                     descrB,
                                                                     safe_size,
                                                                     dcsr_row_ptr_B,
                                                                     dcsr_col_ind_B,
                                                                     hbeta_ptr,
                                                                     descrD,
                                                                     safe_size,
                                                                     dcsr_row_ptr_D,
                                                                     dcsr_col_ind_D,
                                                                     info,
                                                                     &buffer_size);
        rocsparse_status status_2 = rocsparse_csrgemm_nnz(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          descrA,
                                                          safe_size,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          descrB,
                                                          safe_size,
                                                          dcsr_row_ptr_B,
                                                          dcsr_col_ind_B,
                                                          descrD,
                                                          safe_size,
                                                          dcsr_row_ptr_D,
                                                          dcsr_col_ind_D,
                                                          descrC,
                                                          dcsr_row_ptr_C,
                                                          &nnz_C,
                                                          info,
                                                          dbuffer);
        rocsparse_status status_3 = rocsparse_csrgemm<T>(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         halpha_ptr,
                                                         descrA,
                                                         safe_size,
                                                         dcsr_val_A,
                                                         dcsr_row_ptr_A,
                                                         dcsr_col_ind_A,
                                                         descrB,
                                                         safe_size,
                                                         dcsr_val_B,
                                                         dcsr_row_ptr_B,
                                                         dcsr_col_ind_B,
                                                         hbeta_ptr,
                                                         descrD,
                                                         safe_size,
                                                         dcsr_val_D,
                                                         dcsr_row_ptr_D,
                                                         dcsr_col_ind_D,
                                                         descrC,
                                                         dcsr_val_C,
                                                         dcsr_row_ptr_C,
                                                         dcsr_col_ind_C,
                                                         info,
                                                         dbuffer);

        EXPECT_ROCSPARSE_STATUS(status_1,
                                (M < 0 || N < 0 || K < 0) ? rocsparse_status_invalid_size
                                : scenario == 1           ? rocsparse_status_invalid_pointer
                                                          : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(status_2,
                                (M < 0 || N < 0 || K < 0) ? rocsparse_status_invalid_size
                                : scenario == 1           ? rocsparse_status_invalid_pointer
                                                          : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(status_3,
                                (M < 0 || N < 0 || K < 0) ? rocsparse_status_invalid_size
                                : scenario == 1           ? rocsparse_status_invalid_pointer
                                                          : rocsparse_status_success);

        return;
    }

    // Allocate host memory for matrices
    host_vector<rocsparse_int> hcsr_row_ptr_A;
    host_vector<rocsparse_int> hcsr_col_ind_A;
    host_vector<T>             hcsr_val_A;
    host_vector<rocsparse_int> hcsr_row_ptr_B;
    host_vector<rocsparse_int> hcsr_col_ind_B;
    host_vector<T>             hcsr_val_B;
    host_vector<rocsparse_int> hcsr_row_ptr_D;
    host_vector<rocsparse_int> hcsr_col_ind_D;
    host_vector<T>             hcsr_val_D;

    // Sample matrix
    rocsparse_int nnz_A = 4;
    rocsparse_int nnz_B = 4;
    rocsparse_int hnnz_C_gold;
    rocsparse_int hnnz_C_1;
    rocsparse_int hnnz_C_2;
    rocsparse_int nnz_D = 4;

    rocsparse_matrix_factory<T> matrix_factory(arg, arg.timing ? false : true, full_rank);
    if(scenario == 2)
    {
        // alpha != nullptr && beta == nullptr
        matrix_factory.init_csr(hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, M, K, nnz_A, baseA);
        rocsparse_matrix_factory_random<T> matrix_factory_random(full_rank);
        matrix_factory_random.init_csr(
            hcsr_row_ptr_B, hcsr_col_ind_B, hcsr_val_B, K, N, nnz_B, baseB);
    }
    else if(scenario == 3)
    {
        // alpha == nullptr && beta != nullptr
        matrix_factory.init_csr(hcsr_row_ptr_D, hcsr_col_ind_D, hcsr_val_D, M, N, nnz_D, baseD);
    }
    else if(scenario == 4)
    {
        // alpha != nullptr && beta == nullptr
        matrix_factory.init_csr(hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, M, K, nnz_A, baseA);
        rocsparse_matrix_factory_random<T> matrix_factory_random(full_rank);
        matrix_factory_random.init_csr(
            hcsr_row_ptr_B, hcsr_col_ind_B, hcsr_val_B, K, N, nnz_B, baseB);
        matrix_factory.init_csr(hcsr_row_ptr_D, hcsr_col_ind_D, hcsr_val_D, M, N, nnz_D, baseD);
    }

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr_A(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_A(nnz_A);
    device_vector<T>             dcsr_val_A(nnz_A);
    device_vector<rocsparse_int> dcsr_row_ptr_B(K + 1);
    device_vector<rocsparse_int> dcsr_col_ind_B(nnz_B);
    device_vector<T>             dcsr_val_B(nnz_B);
    device_vector<rocsparse_int> dcsr_row_ptr_D(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_D(nnz_D);
    device_vector<T>             dcsr_val_D(nnz_D);
    device_vector<T>             d_alpha(1);
    device_vector<T>             d_beta(1);
    device_vector<rocsparse_int> dcsr_row_ptr_C_1(M + 1);
    device_vector<rocsparse_int> dcsr_row_ptr_C_2(M + 1);
    device_vector<rocsparse_int> dnnz_C_2(1);

    // Copy data from CPU to device
    if(scenario == 2)
    {
        CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_A,
                                  hcsr_row_ptr_A,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dcsr_col_ind_A, hcsr_col_ind_A, sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(T) * nnz_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_B,
                                  hcsr_row_ptr_B,
                                  sizeof(rocsparse_int) * (K + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dcsr_col_ind_B, hcsr_col_ind_B, sizeof(rocsparse_int) * nnz_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_val_B, hcsr_val_B, sizeof(T) * nnz_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        dalpha_ptr = d_alpha;
    }
    else if(scenario == 3)
    {
        CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_D,
                                  hcsr_row_ptr_D,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dcsr_col_ind_D, hcsr_col_ind_D, sizeof(rocsparse_int) * nnz_D, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_val_D, hcsr_val_D, sizeof(T) * nnz_D, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
        dbeta_ptr = d_beta;
    }
    else if(scenario == 4)
    {
        CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_A,
                                  hcsr_row_ptr_A,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dcsr_col_ind_A, hcsr_col_ind_A, sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(T) * nnz_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_B,
                                  hcsr_row_ptr_B,
                                  sizeof(rocsparse_int) * (K + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dcsr_col_ind_B, hcsr_col_ind_B, sizeof(rocsparse_int) * nnz_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_val_B, hcsr_val_B, sizeof(T) * nnz_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        dalpha_ptr = d_alpha;

        CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_D,
                                  hcsr_row_ptr_D,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dcsr_col_ind_D, hcsr_col_ind_D, sizeof(rocsparse_int) * nnz_D, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_val_D, hcsr_val_D, sizeof(T) * nnz_D, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
        dbeta_ptr = d_beta;
    }

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_buffer_size<T>(handle,
                                                           transA,
                                                           transB,
                                                           M,
                                                           N,
                                                           K,
                                                           halpha_ptr,
                                                           descrA,
                                                           nnz_A,
                                                           dcsr_row_ptr_A,
                                                           dcsr_col_ind_A,
                                                           descrB,
                                                           nnz_B,
                                                           dcsr_row_ptr_B,
                                                           dcsr_col_ind_B,
                                                           hbeta_ptr,
                                                           descrD,
                                                           nnz_D,
                                                           dcsr_row_ptr_D,
                                                           dcsr_col_ind_D,
                                                           info,
                                                           &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Obtain nnz of C

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrD,
                                                    nnz_D,
                                                    dcsr_row_ptr_D,
                                                    dcsr_col_ind_D,
                                                    descrC,
                                                    dcsr_row_ptr_C_1,
                                                    &hnnz_C_1,
                                                    info,
                                                    dbuffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrD,
                                                    nnz_D,
                                                    dcsr_row_ptr_D,
                                                    dcsr_col_ind_D,
                                                    descrC,
                                                    dcsr_row_ptr_C_2,
                                                    dnnz_C_2,
                                                    info,
                                                    dbuffer));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_row_ptr_C_1(M + 1);
        host_vector<rocsparse_int> hcsr_row_ptr_C_2(M + 1);
        CHECK_HIP_ERROR(
            hipMemcpy(&hnnz_C_2, dnnz_C_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C_1,
                                  dcsr_row_ptr_C_1,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C_2,
                                  dcsr_row_ptr_C_2,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));

        // CPU csrgemm_nnz
        host_vector<rocsparse_int> hcsr_row_ptr_C_gold(M + 1);
        host_csrgemm_nnz(M,
                         N,
                         K,
                         halpha_ptr,
                         hcsr_row_ptr_A,
                         hcsr_col_ind_A,
                         hcsr_row_ptr_B,
                         hcsr_col_ind_B,
                         hbeta_ptr,
                         hcsr_row_ptr_D,
                         hcsr_col_ind_D,
                         hcsr_row_ptr_C_gold,
                         &hnnz_C_gold,
                         baseA,
                         baseB,
                         baseC,
                         baseD);

        // Check nnz of C
        unit_check_scalar(hnnz_C_gold, hnnz_C_1);
        unit_check_scalar(hnnz_C_gold, hnnz_C_2);

        // Check row pointers of C
        unit_check_segments<rocsparse_int>(M + 1, hcsr_row_ptr_C_gold, hcsr_row_ptr_C_1);
        unit_check_segments<rocsparse_int>(M + 1, hcsr_row_ptr_C_gold, hcsr_row_ptr_C_2);

        // Allocate device memory for C
        device_vector<rocsparse_int> dcsr_col_ind_C_1(hnnz_C_1);
        device_vector<rocsparse_int> dcsr_col_ind_C_2(hnnz_C_2);
        device_vector<T>             dcsr_val_C_1(hnnz_C_1);
        device_vector<T>             dcsr_val_C_2(hnnz_C_2);

        // Perform matrix matrix multiplication

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm<T>(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   halpha_ptr,
                                                   descrA,
                                                   nnz_A,
                                                   dcsr_val_A,
                                                   dcsr_row_ptr_A,
                                                   dcsr_col_ind_A,
                                                   descrB,
                                                   nnz_B,
                                                   dcsr_val_B,
                                                   dcsr_row_ptr_B,
                                                   dcsr_col_ind_B,
                                                   hbeta_ptr,
                                                   descrD,
                                                   nnz_D,
                                                   dcsr_val_D,
                                                   dcsr_row_ptr_D,
                                                   dcsr_col_ind_D,
                                                   descrC,
                                                   dcsr_val_C_1,
                                                   dcsr_row_ptr_C_1,
                                                   dcsr_col_ind_C_1,
                                                   info,
                                                   dbuffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm<T>(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   dalpha_ptr,
                                                   descrA,
                                                   nnz_A,
                                                   dcsr_val_A,
                                                   dcsr_row_ptr_A,
                                                   dcsr_col_ind_A,
                                                   descrB,
                                                   nnz_B,
                                                   dcsr_val_B,
                                                   dcsr_row_ptr_B,
                                                   dcsr_col_ind_B,
                                                   dbeta_ptr,
                                                   descrD,
                                                   nnz_D,
                                                   dcsr_val_D,
                                                   dcsr_row_ptr_D,
                                                   dcsr_col_ind_D,
                                                   descrC,
                                                   dcsr_val_C_2,
                                                   dcsr_row_ptr_C_2,
                                                   dcsr_col_ind_C_2,
                                                   info,
                                                   dbuffer));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_col_ind_C_1(hnnz_C_1);
        host_vector<rocsparse_int> hcsr_col_ind_C_2(hnnz_C_2);
        host_vector<T>             hcsr_val_C_1(hnnz_C_1);
        host_vector<T>             hcsr_val_C_2(hnnz_C_2);

        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C_1,
                                  dcsr_col_ind_C_1,
                                  sizeof(rocsparse_int) * hnnz_C_1,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C_2,
                                  dcsr_col_ind_C_2,
                                  sizeof(rocsparse_int) * hnnz_C_2,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_1, dcsr_val_C_1, sizeof(T) * hnnz_C_1, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_2, dcsr_val_C_2, sizeof(T) * hnnz_C_2, hipMemcpyDeviceToHost));

        // CPU csrgemm
        host_vector<rocsparse_int> hcsr_col_ind_C_gold(hnnz_C_gold);
        host_vector<T>             hcsr_val_C_gold(hnnz_C_gold);
        host_csrgemm(M,
                     N,
                     K,
                     halpha_ptr,
                     hcsr_row_ptr_A,
                     hcsr_col_ind_A,
                     hcsr_val_A,
                     hcsr_row_ptr_B,
                     hcsr_col_ind_B,
                     hcsr_val_B,
                     hbeta_ptr,
                     hcsr_row_ptr_D,
                     hcsr_col_ind_D,
                     hcsr_val_D,
                     hcsr_row_ptr_C_gold,
                     hcsr_col_ind_C_gold,
                     hcsr_val_C_gold,
                     baseA,
                     baseB,
                     baseC,
                     baseD);

        // Check C
        unit_check_segments<rocsparse_int>(hnnz_C_gold, hcsr_col_ind_C_gold, hcsr_col_ind_C_1);
        unit_check_segments<rocsparse_int>(hnnz_C_gold, hcsr_col_ind_C_gold, hcsr_col_ind_C_2);
        near_check_segments<T>(hnnz_C_gold, hcsr_val_C_gold, hcsr_val_C_1);
        near_check_segments<T>(hnnz_C_gold, hcsr_val_C_gold, hcsr_val_C_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        K,
                                                        descrA,
                                                        nnz_A,
                                                        dcsr_row_ptr_A,
                                                        dcsr_col_ind_A,
                                                        descrB,
                                                        nnz_B,
                                                        dcsr_row_ptr_B,
                                                        dcsr_col_ind_B,
                                                        descrD,
                                                        nnz_D,
                                                        dcsr_row_ptr_D,
                                                        dcsr_col_ind_D,
                                                        descrC,
                                                        dcsr_row_ptr_C_1,
                                                        &hnnz_C_1,
                                                        info,
                                                        dbuffer));

            device_vector<rocsparse_int> dcsr_col_ind_C(hnnz_C_1);
            device_vector<T>             dcsr_val_C(hnnz_C_1);

            CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm<T>(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       halpha_ptr,
                                                       descrA,
                                                       nnz_A,
                                                       dcsr_val_A,
                                                       dcsr_row_ptr_A,
                                                       dcsr_col_ind_A,
                                                       descrB,
                                                       nnz_B,
                                                       dcsr_val_B,
                                                       dcsr_row_ptr_B,
                                                       dcsr_col_ind_B,
                                                       hbeta_ptr,
                                                       descrD,
                                                       nnz_D,
                                                       dcsr_val_D,
                                                       dcsr_row_ptr_D,
                                                       dcsr_col_ind_D,
                                                       descrC,
                                                       dcsr_val_C,
                                                       dcsr_row_ptr_C_1,
                                                       dcsr_col_ind_C,
                                                       info,
                                                       dbuffer));
        }

        double gpu_analysis_time_used = get_time_us();

        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrD,
                                                    nnz_D,
                                                    dcsr_row_ptr_D,
                                                    dcsr_col_ind_D,
                                                    descrC,
                                                    dcsr_row_ptr_C_1,
                                                    &hnnz_C_1,
                                                    info,
                                                    dbuffer));

        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        device_vector<rocsparse_int> dcsr_col_ind_C(hnnz_C_1);
        device_vector<T>             dcsr_val_C(hnnz_C_1);

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm<T>(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       halpha_ptr,
                                                       descrA,
                                                       nnz_A,
                                                       dcsr_val_A,
                                                       dcsr_row_ptr_A,
                                                       dcsr_col_ind_A,
                                                       descrB,
                                                       nnz_B,
                                                       dcsr_val_B,
                                                       dcsr_row_ptr_B,
                                                       dcsr_col_ind_B,
                                                       hbeta_ptr,
                                                       descrD,
                                                       nnz_D,
                                                       dcsr_val_D,
                                                       dcsr_row_ptr_D,
                                                       dcsr_col_ind_D,
                                                       descrC,
                                                       dcsr_val_C,
                                                       dcsr_row_ptr_C_1,
                                                       dcsr_col_ind_C,
                                                       info,
                                                       dbuffer));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gpu_gflops = csrgemm_gflop_count<T, rocsparse_int, rocsparse_int>(M,
                                                                                 halpha_ptr,
                                                                                 hcsr_row_ptr_A,
                                                                                 hcsr_col_ind_A,
                                                                                 hcsr_row_ptr_B,
                                                                                 hbeta_ptr,
                                                                                 hcsr_row_ptr_D,
                                                                                 baseA)
                            / gpu_solve_time_used * 1e6;
        double gpu_gbyte
            = csrgemm_gbyte_count(M, N, K, nnz_A, nnz_B, hnnz_C_1, nnz_D, halpha_ptr, hbeta_ptr)
              / gpu_solve_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "opA" << std::setw(12) << "opB" << std::setw(12) << "M"
                  << std::setw(12) << "N" << std::setw(12) << "K" << std::setw(12) << "nnz_A"
                  << std::setw(12) << "nnz_B" << std::setw(12) << "nnz_C" << std::setw(12)
                  << "nnz_D" << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(16) << "nnz msec"
                  << std::setw(16) << "gemm msec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << rocsparse_operation2string(transA) << std::setw(12)
                  << rocsparse_operation2string(transB) << std::setw(12) << M << std::setw(12) << N
                  << std::setw(12) << K << std::setw(12) << nnz_A << std::setw(12) << nnz_B
                  << std::setw(12) << hnnz_C_1 << std::setw(12) << nnz_D;
        if(scenario == 2 || scenario == 4)
        {
            std::cout << std::setw(12) << h_alpha;
        }
        else
        {
            std::cout << std::setw(12) << "null";
        }
        if(scenario == 3 || scenario == 4)
        {
            std::cout << std::setw(12) << h_beta;
        }
        else
        {
            std::cout << std::setw(12) << "null";
        }
        std::cout << std::setw(12) << gpu_gflops << std::setw(12) << gpu_gbyte << std::setw(16)
                  << gpu_analysis_time_used / 1e3 << std::setw(16) << gpu_solve_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // Free buffer
    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csrgemm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrgemm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
