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

#include "testing.hpp"

template <typename I, typename J, typename T>
void testing_spgemm_csr_bad_arg(const Arguments& arg)
{
    J m     = 100;
    J n     = 100;
    J k     = 100;
    I nnz_A = 100;
    I nnz_B = 100;
    I nnz_C = 100;
    I nnz_D = 100;

    I safe_size = 100;

    T alpha = 0.6;
    T beta  = 0.1;

    rocsparse_operation    trans = rocsparse_operation_none;
    rocsparse_index_base   base  = rocsparse_index_base_zero;
    rocsparse_spgemm_alg   alg   = rocsparse_spgemm_alg_default;
    rocsparse_spgemm_stage stage = rocsparse_spgemm_stage_auto;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<I> dcsr_row_ptr_A(m + 1);
    device_vector<I> dcsr_col_ind_A(nnz_A);
    device_vector<T> dcsr_val_A(nnz_A);
    device_vector<I> dcsr_row_ptr_B(k + 1);
    device_vector<I> dcsr_col_ind_B(nnz_B);
    device_vector<T> dcsr_val_B(nnz_B);
    device_vector<I> dcsr_row_ptr_D(m + 1);
    device_vector<I> dcsr_row_ptr_C(m + 1);
    device_vector<I> dcsr_col_ind_C(safe_size);
    device_vector<T> dcsr_val_C(safe_size);
    device_vector<I> dcsr_col_ind_D(nnz_D);
    device_vector<T> dcsr_val_D(nnz_D);

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
       || !dcsr_val_B || !dcsr_row_ptr_C || !dcsr_col_ind_C || !dcsr_val_C || !dcsr_row_ptr_D
       || !dcsr_col_ind_D || !dcsr_val_D)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // SpGEMM structures
    rocsparse_local_spmat A(
        m, k, nnz_A, dcsr_row_ptr_A, dcsr_col_ind_A, dcsr_val_A, itype, jtype, base, ttype);
    rocsparse_local_spmat B(
        k, n, nnz_B, dcsr_row_ptr_B, dcsr_col_ind_B, dcsr_val_B, itype, jtype, base, ttype);
    rocsparse_local_spmat C(
        m, n, nnz_C, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C, itype, jtype, base, ttype);
    rocsparse_local_spmat D(
        m, n, nnz_D, dcsr_row_ptr_D, dcsr_col_ind_D, dcsr_val_D, itype, jtype, base, ttype);

    // Test SpMV with invalid buffer
    size_t buffer_size;

    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(nullptr,
                                             trans,
                                             trans,
                                             &alpha,
                                             A,
                                             B,
                                             &beta,
                                             D,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             nullptr),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             &alpha,
                                             nullptr,
                                             B,
                                             &beta,
                                             D,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             &alpha,
                                             A,
                                             nullptr,
                                             &beta,
                                             D,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             &alpha,
                                             A,
                                             B,
                                             &beta,
                                             nullptr,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             &alpha,
                                             A,
                                             B,
                                             &beta,
                                             D,
                                             nullptr,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spgemm(
            handle, trans, trans, &alpha, A, B, &beta, D, C, ttype, alg, stage, nullptr, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             nullptr,
                                             A,
                                             B,
                                             nullptr,
                                             D,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             nullptr),
                            rocsparse_status_invalid_pointer);

    // Test SpGEMM with valid buffer
    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, 100));

    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(nullptr,
                                             trans,
                                             trans,
                                             &alpha,
                                             A,
                                             B,
                                             &beta,
                                             D,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             &alpha,
                                             nullptr,
                                             B,
                                             &beta,
                                             D,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             &alpha,
                                             A,
                                             nullptr,
                                             &beta,
                                             D,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             &alpha,
                                             A,
                                             B,
                                             &beta,
                                             nullptr,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             &alpha,
                                             A,
                                             B,
                                             &beta,
                                             D,
                                             nullptr,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans,
                                             trans,
                                             nullptr,
                                             A,
                                             B,
                                             nullptr,
                                             D,
                                             C,
                                             ttype,
                                             alg,
                                             stage,
                                             &buffer_size,
                                             dbuffer),
                            rocsparse_status_invalid_pointer);

    CHECK_HIP_ERROR(hipFree(dbuffer));
}

template <typename I, typename J, typename T>
void testing_spgemm_csr(const Arguments& arg)
{
    J                     M         = arg.M;
    J                     N         = arg.N;
    J                     K         = arg.K;
    int32_t               dim_x     = arg.dimx;
    int32_t               dim_y     = arg.dimy;
    int32_t               dim_z     = arg.dimz;
    rocsparse_operation   trans_A   = arg.transA;
    rocsparse_operation   trans_B   = arg.transA;
    rocsparse_index_base  base_A    = arg.baseA;
    rocsparse_index_base  base_B    = arg.baseB;
    rocsparse_index_base  base_C    = arg.baseC;
    rocsparse_index_base  base_D    = arg.baseD;
    rocsparse_spgemm_alg  alg       = arg.spgemm_alg;
    rocsparse_matrix_init mat       = arg.matrix;
    bool                  full_rank = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // -99 means nullptr
    T* h_alpha_ptr = (h_alpha == (T)-99) ? nullptr : &h_alpha;
    T* h_beta_ptr  = (h_beta == (T)-99) ? nullptr : &h_beta;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // SpGEMM stage
    rocsparse_spgemm_stage stage = rocsparse_spgemm_stage_auto;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if((M <= 0 || N <= 0 || K <= 0) && (M <= 0 || N <= 0 || K != 0 || h_beta_ptr == nullptr))
    {
        static const I safe_size = 1;

        // Allocate memory on device
        device_vector<I> dcsr_row_ptr_A(safe_size);
        device_vector<J> dcsr_col_ind_A(safe_size);
        device_vector<T> dcsr_val_A(safe_size);
        device_vector<I> dcsr_row_ptr_B(safe_size);
        device_vector<J> dcsr_col_ind_B(safe_size);
        device_vector<T> dcsr_val_B(safe_size);
        device_vector<I> dcsr_row_ptr_D(safe_size);
        device_vector<J> dcsr_col_ind_D(safe_size);
        device_vector<T> dcsr_val_D(safe_size);
        device_vector<I> dcsr_row_ptr_C(safe_size);
        device_vector<J> dcsr_col_ind_C(safe_size);
        device_vector<T> dcsr_val_C(safe_size);

        if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
           || !dcsr_val_B || !dcsr_row_ptr_D || !dcsr_col_ind_D || !dcsr_val_D || !dcsr_row_ptr_C
           || !dcsr_col_ind_C || !dcsr_val_C)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Non-zero entries
        I nnz_A = (M > 0 && K > 0) ? safe_size : 0;
        I nnz_B = (K > 0 && N > 0) ? safe_size : 0;
        I nnz_D = (M > 0 && N > 0) ? safe_size : 0;

        // Check structures
        rocsparse_local_spmat A(
            M, K, nnz_A, dcsr_row_ptr_A, dcsr_col_ind_A, dcsr_val_A, itype, jtype, base_A, ttype);
        rocsparse_local_spmat B(
            K, N, nnz_B, dcsr_row_ptr_B, dcsr_col_ind_B, dcsr_val_B, itype, jtype, base_B, ttype);
        rocsparse_local_spmat D(
            M, N, nnz_D, dcsr_row_ptr_D, dcsr_col_ind_D, dcsr_val_D, itype, jtype, base_D, ttype);
        rocsparse_local_spmat C(
            M, N, 0, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C, itype, jtype, base_C, ttype);

        // Pointer mode
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Query SpGEMM buffer
        size_t buffer_size;
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                                 trans_A,
                                                 trans_B,
                                                 h_alpha_ptr,
                                                 A,
                                                 B,
                                                 h_beta_ptr,
                                                 D,
                                                 C,
                                                 ttype,
                                                 alg,
                                                 stage,
                                                 &buffer_size,
                                                 nullptr),
                                rocsparse_status_success);

        void* dbuffer;
        CHECK_HIP_ERROR(hipMalloc(&dbuffer, safe_size));

        // Count non-zeros of C
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                                 trans_A,
                                                 trans_B,
                                                 h_alpha_ptr,
                                                 A,
                                                 B,
                                                 h_beta_ptr,
                                                 D,
                                                 C,
                                                 ttype,
                                                 alg,
                                                 stage,
                                                 &buffer_size,
                                                 dbuffer),
                                rocsparse_status_success);

        // Verify that nnz_C is equal to zero
        int64_t rows_C;
        int64_t cols_C;
        int64_t nnz_C;
        int64_t zero = 0;

        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_size(C, &rows_C, &cols_C, &nnz_C));
        unit_check_general(1, 1, 1, &zero, &nnz_C);

        CHECK_HIP_ERROR(hipFree(dbuffer));

        return;
    }

    // Allocate host memory for matrix
    host_vector<I> hcsr_row_ptr_A;
    host_vector<J> hcsr_col_ind_A;
    host_vector<T> hcsr_val_A;
    host_vector<I> hcsr_row_ptr_B;
    host_vector<J> hcsr_col_ind_B;
    host_vector<T> hcsr_val_B;
    host_vector<I> hcsr_row_ptr_D;
    host_vector<J> hcsr_col_ind_D;
    host_vector<T> hcsr_val_D;

    rocsparse_seedrand();

    // Sample matrices
    I nnz_A;
    I nnz_B;
    I nnz_D;

    // Matrix A
    rocsparse_init_csr_matrix(hcsr_row_ptr_A,
                              hcsr_col_ind_A,
                              hcsr_val_A,
                              M,
                              K,
                              N,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz_A,
                              base_A,
                              mat,
                              filename.c_str(),
                              arg.timing ? false : true,
                              full_rank);

    // Matrix B
    rocsparse_init_csr_matrix(hcsr_row_ptr_B,
                              hcsr_col_ind_B,
                              hcsr_val_B,
                              K,
                              N,
                              M,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz_B,
                              base_B,
                              rocsparse_matrix_random,
                              filename.c_str(),
                              arg.timing ? false : true,
                              full_rank);

    // Matrix D
    rocsparse_init_csr_matrix(hcsr_row_ptr_D,
                              hcsr_col_ind_D,
                              hcsr_val_D,
                              M,
                              N,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz_D,
                              base_D,
                              rocsparse_matrix_random,
                              filename.c_str(),
                              arg.timing ? false : true,
                              full_rank);

    // Allocate device memory
    device_vector<I> dcsr_row_ptr_A(M + 1);
    device_vector<J> dcsr_col_ind_A(nnz_A);
    device_vector<T> dcsr_val_A(nnz_A);
    device_vector<I> dcsr_row_ptr_B(K + 1);
    device_vector<J> dcsr_col_ind_B(nnz_B);
    device_vector<T> dcsr_val_B(nnz_B);
    device_vector<I> dcsr_row_ptr_D(M + 1);
    device_vector<J> dcsr_col_ind_D(nnz_D);
    device_vector<T> dcsr_val_D(nnz_D);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    device_vector<I> dcsr_row_ptr_C_1(M + 1);
    device_vector<I> dcsr_row_ptr_C_2(M + 1);

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
       || !dcsr_val_B || !dcsr_row_ptr_D || !dcsr_col_ind_D || !dcsr_val_D || !d_alpha || !d_beta
       || !dcsr_row_ptr_C_1 || !dcsr_row_ptr_C_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr_A, hcsr_row_ptr_A, sizeof(I) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind_A, hcsr_col_ind_A, sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr_B, hcsr_row_ptr_B, sizeof(I) * (K + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind_B, hcsr_col_ind_B, sizeof(J) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val_B, hcsr_val_B, sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr_D, hcsr_row_ptr_D, sizeof(I) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind_D, hcsr_col_ind_D, sizeof(J) * nnz_D, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val_D, hcsr_val_D, sizeof(T) * nnz_D, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    T* d_alpha_ptr = (h_alpha == (T)-99) ? nullptr : d_alpha;
    T* d_beta_ptr  = (h_beta == (T)-99) ? nullptr : d_beta;

    // Create descriptors
    rocsparse_local_spmat A(
        M, K, nnz_A, dcsr_row_ptr_A, dcsr_col_ind_A, dcsr_val_A, itype, jtype, base_A, ttype);
    rocsparse_local_spmat B(
        K, N, nnz_B, dcsr_row_ptr_B, dcsr_col_ind_B, dcsr_val_B, itype, jtype, base_B, ttype);
    rocsparse_local_spmat D(
        M, N, nnz_D, dcsr_row_ptr_D, dcsr_col_ind_D, dcsr_val_D, itype, jtype, base_D, ttype);
    rocsparse_local_spmat C1(
        M, N, 0, dcsr_row_ptr_C_1, nullptr, nullptr, itype, jtype, base_C, ttype);
    rocsparse_local_spmat C2(
        M, N, 0, dcsr_row_ptr_C_2, nullptr, nullptr, itype, jtype, base_C, ttype);

    // Query SpGEMM buffer
    size_t buffer_size_1;
    size_t buffer_size_2;
    CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                           trans_A,
                                           trans_B,
                                           h_alpha_ptr,
                                           A,
                                           B,
                                           h_beta_ptr,
                                           D,
                                           C1,
                                           ttype,
                                           alg,
                                           stage,
                                           &buffer_size_1,
                                           nullptr));
    CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                           trans_A,
                                           trans_B,
                                           d_alpha_ptr,
                                           A,
                                           B,
                                           d_beta_ptr,
                                           D,
                                           C2,
                                           ttype,
                                           alg,
                                           stage,
                                           &buffer_size_2,
                                           nullptr));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, std::max(buffer_size_1, buffer_size_2)));

    if(arg.unit_check)
    {
        // SpGEMM - count non-zeros of C

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha_ptr,
                                               A,
                                               B,
                                               h_beta_ptr,
                                               D,
                                               C1,
                                               ttype,
                                               alg,
                                               stage,
                                               &buffer_size_1,
                                               dbuffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                               trans_A,
                                               trans_B,
                                               d_alpha_ptr,
                                               A,
                                               B,
                                               d_beta_ptr,
                                               D,
                                               C2,
                                               ttype,
                                               alg,
                                               stage,
                                               &buffer_size_2,
                                               dbuffer));

        // Copy output to host
        host_vector<I> hcsr_row_ptr_C_1(M + 1);
        host_vector<I> hcsr_row_ptr_C_2(M + 1);

        int64_t rows_C;
        int64_t cols_C;
        int64_t nnz_C_1;
        int64_t nnz_C_2;

        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_size(C1, &rows_C, &cols_C, &nnz_C_1));
        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_size(C2, &rows_C, &cols_C, &nnz_C_2));

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr_C_1.data(), dcsr_row_ptr_C_1, sizeof(I) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr_C_2.data(), dcsr_row_ptr_C_2, sizeof(I) * (M + 1), hipMemcpyDeviceToHost));

        // CPU SpGEMM - count non-zeros of C
        I              nnz_C_gold;
        host_vector<I> hcsr_row_ptr_C_gold(M + 1);

        host_csrgemm_nnz(M,
                         N,
                         K,
                         h_alpha_ptr,
                         hcsr_row_ptr_A,
                         hcsr_col_ind_A,
                         hcsr_row_ptr_B,
                         hcsr_col_ind_B,
                         h_beta_ptr,
                         hcsr_row_ptr_D,
                         hcsr_col_ind_D,
                         hcsr_row_ptr_C_gold,
                         &nnz_C_gold,
                         base_A,
                         base_B,
                         base_C,
                         base_D);

        // Check nnz of C
        int64_t nnz_C = nnz_C_gold;
        unit_check_general(1, 1, 1, &nnz_C, &nnz_C_1);
        unit_check_general(1, 1, 1, &nnz_C, &nnz_C_2);

        // Check row pointers of C
        unit_check_general<I>(1, M + 1, 1, hcsr_row_ptr_C_gold, hcsr_row_ptr_C_1);
        unit_check_general<I>(1, M + 1, 1, hcsr_row_ptr_C_gold, hcsr_row_ptr_C_2);

        // Allocate device memory for C
        device_vector<J> dcsr_col_ind_C_1(nnz_C);
        device_vector<J> dcsr_col_ind_C_2(nnz_C);
        device_vector<T> dcsr_val_C_1(nnz_C);
        device_vector<T> dcsr_val_C_2(nnz_C);

        CHECK_ROCSPARSE_ERROR(
            rocsparse_csr_set_pointers(C1, dcsr_row_ptr_C_1, dcsr_col_ind_C_1, dcsr_val_C_1));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_csr_set_pointers(C2, dcsr_row_ptr_C_2, dcsr_col_ind_C_2, dcsr_val_C_2));

        // Compute sparse matrix sparse matrix product

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha_ptr,
                                               A,
                                               B,
                                               h_beta_ptr,
                                               D,
                                               C1,
                                               ttype,
                                               alg,
                                               stage,
                                               &buffer_size_1,
                                               dbuffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                               trans_A,
                                               trans_B,
                                               d_alpha_ptr,
                                               A,
                                               B,
                                               d_beta_ptr,
                                               D,
                                               C2,
                                               ttype,
                                               alg,
                                               stage,
                                               &buffer_size_2,
                                               dbuffer));

        // Copy output to host
        host_vector<J> hcsr_col_ind_C_1(nnz_C);
        host_vector<J> hcsr_col_ind_C_2(nnz_C);
        host_vector<T> hcsr_val_C_1(nnz_C);
        host_vector<T> hcsr_val_C_2(nnz_C);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C_1, dcsr_col_ind_C_1, sizeof(J) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C_2, dcsr_col_ind_C_2, sizeof(J) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_1, dcsr_val_C_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_2, dcsr_val_C_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

        // CPU SpGEMM
        host_vector<J> hcsr_col_ind_C_gold(nnz_C);
        host_vector<T> hcsr_val_C_gold(nnz_C);
        host_csrgemm(M,
                     N,
                     K,
                     h_alpha_ptr,
                     hcsr_row_ptr_A,
                     hcsr_col_ind_A,
                     hcsr_val_A,
                     hcsr_row_ptr_B,
                     hcsr_col_ind_B,
                     hcsr_val_B,
                     h_beta_ptr,
                     hcsr_row_ptr_D,
                     hcsr_col_ind_D,
                     hcsr_val_D,
                     hcsr_row_ptr_C_gold,
                     hcsr_col_ind_C_gold,
                     hcsr_val_C_gold,
                     base_A,
                     base_B,
                     base_C,
                     base_D);

        // Check C
        unit_check_general<J>(1, nnz_C, 1, hcsr_col_ind_C_gold, hcsr_col_ind_C_1);
        unit_check_general<J>(1, nnz_C, 1, hcsr_col_ind_C_gold, hcsr_col_ind_C_2);
        near_check_general<T>(1, nnz_C, 1, hcsr_val_C_gold, hcsr_val_C_1);
        near_check_general<T>(1, nnz_C, 1, hcsr_val_C_gold, hcsr_val_C_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        int64_t rows_C;
        int64_t cols_C;
        int64_t nnz_C;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            // Sparse matrix descriptor C
            rocsparse_local_spmat C(
                M, N, 0, dcsr_row_ptr_C_1, nullptr, nullptr, itype, jtype, base_C, ttype);

            // Query for buffer size
            size_t buffer_size;
            CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                                   trans_A,
                                                   trans_B,
                                                   h_alpha_ptr,
                                                   A,
                                                   B,
                                                   h_beta_ptr,
                                                   D,
                                                   C,
                                                   ttype,
                                                   alg,
                                                   stage,
                                                   &buffer_size,
                                                   nullptr));

            // Query non-zeros of C
            CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                                   trans_A,
                                                   trans_B,
                                                   h_alpha_ptr,
                                                   A,
                                                   B,
                                                   h_beta_ptr,
                                                   D,
                                                   C,
                                                   ttype,
                                                   alg,
                                                   stage,
                                                   &buffer_size,
                                                   dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_size(C, &rows_C, &cols_C, &nnz_C));

            device_vector<J> dcsr_col_ind_C(nnz_C);
            device_vector<T> dcsr_val_C(nnz_C);

            CHECK_ROCSPARSE_ERROR(
                rocsparse_csr_set_pointers(C, dcsr_row_ptr_C_1, dcsr_col_ind_C, dcsr_val_C));

            // Compute matrix matrix product
            CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                                   trans_A,
                                                   trans_B,
                                                   h_alpha_ptr,
                                                   A,
                                                   B,
                                                   h_beta_ptr,
                                                   D,
                                                   C,
                                                   ttype,
                                                   alg,
                                                   stage,
                                                   &buffer_size_1,
                                                   dbuffer));
        }

        // Sparse matrix descriptor C
        rocsparse_local_spmat C(
            M, N, 0, dcsr_row_ptr_C_1, nullptr, nullptr, itype, jtype, base_C, ttype);

        double gpu_analysis_time_used = get_time_us();

        // Query for buffer size
        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha_ptr,
                                               A,
                                               B,
                                               h_beta_ptr,
                                               D,
                                               C,
                                               ttype,
                                               alg,
                                               stage,
                                               &buffer_size,
                                               nullptr));

        // Query non-zeros of C
        CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha_ptr,
                                               A,
                                               B,
                                               h_beta_ptr,
                                               D,
                                               C,
                                               ttype,
                                               alg,
                                               stage,
                                               &buffer_size,
                                               dbuffer));

        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_size(C, &rows_C, &cols_C, &nnz_C));

        device_vector<J> dcsr_col_ind_C(nnz_C);
        device_vector<T> dcsr_val_C(nnz_C);

        CHECK_ROCSPARSE_ERROR(
            rocsparse_csr_set_pointers(C, dcsr_row_ptr_C_1, dcsr_col_ind_C, dcsr_val_C));

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spgemm(handle,
                                                   trans_A,
                                                   trans_B,
                                                   h_alpha_ptr,
                                                   A,
                                                   B,
                                                   h_beta_ptr,
                                                   D,
                                                   C,
                                                   ttype,
                                                   alg,
                                                   stage,
                                                   &buffer_size_1,
                                                   dbuffer));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gpu_gflops = csrgemm_gflop_count(M,
                                                h_alpha_ptr,
                                                hcsr_row_ptr_A,
                                                hcsr_col_ind_A,
                                                hcsr_row_ptr_B,
                                                h_beta_ptr,
                                                hcsr_row_ptr_D,
                                                base_A)
                            / gpu_solve_time_used * 1e6;
        double gpu_gbyte = csrgemm_gbyte_count<I, J, T>(
                               M, N, K, nnz_A, nnz_B, nnz_C, nnz_D, h_alpha_ptr, h_beta_ptr)
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

        std::cout << std::setw(12) << rocsparse_operation2string(trans_A) << std::setw(12)
                  << rocsparse_operation2string(trans_B) << std::setw(12) << M << std::setw(12) << N
                  << std::setw(12) << K << std::setw(12) << nnz_A << std::setw(12) << nnz_B
                  << std::setw(12) << nnz_C << std::setw(12) << nnz_D << std::setw(12) << h_alpha
                  << std::setw(12) << h_beta << std::setw(12) << gpu_gflops << std::setw(12)
                  << gpu_gbyte << std::setw(16) << gpu_analysis_time_used / 1e3 << std::setw(16)
                  << gpu_solve_time_used / 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // Free buffer
    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template void testing_spgemm_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spgemm_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
