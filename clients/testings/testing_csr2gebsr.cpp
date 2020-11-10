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

#include "utility.hpp"
#include <rocsparse.hpp>

#include "gbyte.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_init.hpp"
#include "rocsparse_math.hpp"
#include "rocsparse_random.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_vector.hpp"

template <typename T>
void testing_csr2gebsr_bad_arg(const Arguments& arg)
{

    static const size_t        safe_size     = 100;
    static const rocsparse_int row_block_dim = safe_size;
    static const rocsparse_int col_block_dim = safe_size;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind(safe_size);
    device_vector<T>             dcsr_val(safe_size);
    device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dbsr_col_ind(safe_size);
    device_vector<T>             dbsr_val(safe_size);
    device_vector<T>             dbuffer(safe_size);

    rocsparse_int hbsr_nnzb;
    size_t        buffer_size;

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dbsr_row_ptr || !dbsr_col_ind || !dbsr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_descr bsr_descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    //
    // Declaration of arguments.
    //
    rocsparse_direction  arg_direction;
    rocsparse_int        arg_m;
    rocsparse_int        arg_n;
    rocsparse_mat_descr  arg_csr_descr;
    const T*             arg_csr_val;
    const rocsparse_int* arg_csr_row_ptr;
    const rocsparse_int* arg_csr_col_ind;
    rocsparse_mat_descr  arg_bsr_descr;
    T*                   arg_bsr_val;
    rocsparse_int*       arg_bsr_row_ptr;
    rocsparse_int*       arg_bsr_col_ind;
    rocsparse_int        arg_row_block_dim;
    rocsparse_int        arg_col_block_dim;
    void*                arg_p_buffer;
    rocsparse_int*       arg_bsr_nnz_devhost;
    size_t*              arg_p_buffer_size;

    //
    // Macro to set arguments.
    //
#define ARGSET                                     \
    arg_direction       = rocsparse_direction_row; \
    arg_m               = safe_size;               \
    arg_n               = safe_size;               \
    arg_csr_descr       = csr_descr;               \
    arg_csr_val         = (T*)dcsr_val;            \
    arg_csr_row_ptr     = dcsr_row_ptr;            \
    arg_csr_col_ind     = dcsr_col_ind;            \
    arg_bsr_descr       = bsr_descr;               \
    arg_bsr_val         = (T*)dbsr_val;            \
    arg_bsr_row_ptr     = dbsr_row_ptr;            \
    arg_bsr_col_ind     = dbsr_col_ind;            \
    arg_row_block_dim   = safe_size;               \
    arg_col_block_dim   = safe_size;               \
    arg_p_buffer        = (void*)((T*)dbuffer);    \
    arg_bsr_nnz_devhost = &hbsr_nnzb;              \
    arg_p_buffer_size   = &buffer_size

    //
    // BUFFER_SIZE ############
    //
#define CALL_ARG_BUFFER_SIZE                                                                   \
    arg_direction, arg_m, arg_n, arg_csr_descr, arg_csr_val, arg_csr_row_ptr, arg_csr_col_ind, \
        arg_row_block_dim, arg_col_block_dim, arg_p_buffer_size

#define CALL_BUFFER_SIZE rocsparse_csr2gebsr_buffer_size(handle, CALL_ARG_BUFFER_SIZE)

    {
        ARGSET;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr_buffer_size(nullptr, CALL_ARG_BUFFER_SIZE),
                                rocsparse_status_invalid_handle);
    }

    {
        ARGSET;
        arg_direction = (rocsparse_direction)2;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_value);
    }

    {
        ARGSET;
        arg_m = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_n = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_csr_descr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_csr_val = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_csr_row_ptr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_csr_col_ind = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_row_block_dim = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_col_block_dim = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_p_buffer_size = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_BUFFER_SIZE, rocsparse_status_invalid_pointer);
    }

#undef CALL_ARG_BUFFER_SIZE
#undef CALL_BUFFER_SIZE

    //
    // NNZ ############
    //
#define CALL_ARG_NNZ                                                                             \
    arg_direction, arg_m, arg_n, arg_csr_descr, arg_csr_row_ptr, arg_csr_col_ind, arg_bsr_descr, \
        arg_bsr_row_ptr, arg_row_block_dim, arg_col_block_dim, arg_bsr_nnz_devhost, arg_p_buffer

#define CALL_NNZ rocsparse_csr2gebsr_nnz(handle, CALL_ARG_NNZ)

    {
        ARGSET;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr_nnz(nullptr, CALL_ARG_NNZ),
                                rocsparse_status_invalid_handle);
    }

    {
        ARGSET;
        arg_direction = (rocsparse_direction)2;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_value);
    }

    {
        ARGSET;
        arg_m = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_n = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_csr_descr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_csr_row_ptr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_csr_col_ind = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_bsr_descr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_bsr_row_ptr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_row_block_dim = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_col_block_dim = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_bsr_nnz_devhost = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_p_buffer = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_NNZ, rocsparse_status_invalid_pointer);
    }

#undef CALL_NNZ
#undef CALL_ARG_NNZ

#define CALL_ARG_FUNC                                                                          \
    arg_direction, arg_m, arg_n, arg_csr_descr, arg_csr_val, arg_csr_row_ptr, arg_csr_col_ind, \
        arg_bsr_descr, arg_bsr_val, arg_bsr_row_ptr, arg_bsr_col_ind, arg_row_block_dim,       \
        arg_col_block_dim, arg_p_buffer

#define CALL_FUNC rocsparse_csr2gebsr(handle, CALL_ARG_FUNC)

    {
        ARGSET;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr(nullptr, CALL_ARG_FUNC),
                                rocsparse_status_invalid_handle);
    }

    {
        ARGSET;
        arg_direction = (rocsparse_direction)2;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_value);
    }

    {
        ARGSET;
        arg_m = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_n = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_csr_descr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_csr_val = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_csr_row_ptr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_csr_col_ind = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_bsr_descr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_bsr_val = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_bsr_row_ptr = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_pointer);
    }
    {
        ARGSET;
        arg_bsr_col_ind = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_pointer);
    }

    {
        ARGSET;
        arg_row_block_dim = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_col_block_dim = -1;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_size);
    }

    {
        ARGSET;
        arg_p_buffer = nullptr;
        EXPECT_ROCSPARSE_STATUS(CALL_FUNC, rocsparse_status_invalid_pointer);
    }

#undef CALL_FUNC
#undef CALL_ARG_FUNC

#undef ARGSET
}

template <typename T>
void testing_csr2gebsr(const Arguments& arg)
{
    rocsparse_int         M             = arg.M;
    rocsparse_int         N             = arg.N;
    rocsparse_int         K             = arg.K;
    rocsparse_int         dim_x         = arg.dimx;
    rocsparse_int         dim_y         = arg.dimy;
    rocsparse_int         dim_z         = arg.dimz;
    rocsparse_index_base  csr_base      = arg.baseA;
    rocsparse_index_base  bsr_base      = arg.baseB;
    rocsparse_matrix_init mat           = arg.matrix;
    rocsparse_direction   direction     = arg.direction;
    rocsparse_int         row_block_dim = arg.row_block_dimA;
    rocsparse_int         col_block_dim = arg.col_block_dimA;
    bool                  full_rank     = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_descr bsr_descr;

    rocsparse_set_mat_index_base(csr_descr, csr_base);
    rocsparse_set_mat_index_base(bsr_descr, bsr_base);

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || row_block_dim <= 0 || col_block_dim <= 0)
    {
        static const size_t safe_size = 100;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr_nnz(handle,
                                                        rocsparse_direction_row,
                                                        M,
                                                        N,
                                                        csr_descr,
                                                        nullptr,
                                                        nullptr,
                                                        bsr_descr,
                                                        nullptr,
                                                        row_block_dim,
                                                        col_block_dim,
                                                        nullptr,
                                                        nullptr),
                                (M < 0 || N < 0 || row_block_dim < 0 || col_block_dim < 0)
                                    ? rocsparse_status_invalid_size
                                    : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr<T>(handle,
                                                       direction,
                                                       M,
                                                       N,
                                                       csr_descr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       bsr_descr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       row_block_dim,
                                                       col_block_dim,
                                                       nullptr),
                                (M < 0 || N < 0 || row_block_dim < 0)
                                    ? rocsparse_status_invalid_size
                                    : rocsparse_status_success);

        return;
    }

    // Allocate host memory for uncompressed CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_A;
    host_vector<rocsparse_int> hcsr_col_ind_A;
    host_vector<T>             hcsr_val_A;

    rocsparse_seedrand();

    // Generate (or load from file) uncompressed CSR matrix
    rocsparse_int nnz;
    rocsparse_init_csr_matrix(hcsr_row_ptr_A,
                              hcsr_col_ind_A,
                              hcsr_val_A,
                              M,
                              N,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz,
                              csr_base,
                              mat,
                              filename.c_str(),
                              false,
                              full_rank);

    // Uncompressed CSR matrix on device
    device_vector<rocsparse_int> dcsr_row_ptr_A(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_A(nnz);
    device_vector<T>             dcsr_val_A(nnz);

    // Copy uncompressed host data to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Compress CSR matrix to ensure it contains no zeros (some matrices loaded from files will have zeros)
    T                            tol = static_cast<T>(0);
    rocsparse_int                nnz_C;
    device_vector<rocsparse_int> dnnz_per_row(M);
    CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
        handle, M, csr_descr, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &nnz_C, tol));

    // Allocate device memory for the compressed version of the CSR matrix
    device_vector<rocsparse_int> dcsr_row_ptr_C(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_C(nnz_C);
    device_vector<T>             dcsr_val_C(nnz_C);

    // Finish compression
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2csr_compress<T>(handle,
                                                        M,
                                                        N,
                                                        csr_descr,
                                                        dcsr_val_A,
                                                        dcsr_row_ptr_A,
                                                        dcsr_col_ind_A,
                                                        nnz,
                                                        dnnz_per_row,
                                                        dcsr_val_C,
                                                        dcsr_row_ptr_C,
                                                        dcsr_col_ind_C,
                                                        tol));

    // Allocate host memory for compressed CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_C(M + 1);
    host_vector<rocsparse_int> hcsr_col_ind_C(nnz_C);
    host_vector<T>             hcsr_val_C(nnz_C);

    // Copy compressed CSR matrix to host
    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_row_ptr_C, dcsr_row_ptr_C, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_col_ind_C, dcsr_col_ind_C, sizeof(rocsparse_int) * nnz_C, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hcsr_val_C, dcsr_val_C, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

    // M and N can be modified in rocsparse_init_csr_matrix
    rocsparse_int Mb = (M + row_block_dim - 1) / row_block_dim;
    rocsparse_int Nb = (N + col_block_dim - 1) / col_block_dim;

    // Allocate host memory for BSR row ptr array
    host_vector<rocsparse_int> hbsr_row_ptr(Mb + 1);

    // Allocate device memory for BSR row ptr array
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);
    if(!dcsr_row_ptr_C || !dcsr_col_ind_C || !dcsr_val_C || !dbsr_row_ptr)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    size_t buffer_size;

    CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_buffer_size(handle,
                                                          direction,
                                                          M,
                                                          N,
                                                          csr_descr,
                                                          (const T*)dcsr_val_C,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          row_block_dim,
                                                          col_block_dim,
                                                          &buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    device_vector<size_t> dbuffer_size(1);
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_buffer_size(handle,
                                                          direction,
                                                          M,
                                                          N,
                                                          csr_descr,
                                                          (const T*)dcsr_val_C,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          row_block_dim,
                                                          col_block_dim,
                                                          dbuffer_size));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    size_t buffer_size_copied_from_device = 0;
    CHECK_HIP_ERROR(hipMemcpy(
        &buffer_size_copied_from_device, dbuffer_size, sizeof(size_t), hipMemcpyDeviceToHost));

    // Confirm that nnzb is the same regardless of whether we use host or device pointers
    unit_check_general<size_t>(1, 1, 1, &buffer_size, &buffer_size_copied_from_device);

    device_vector<T> dbuffer(buffer_size);
    if(arg.unit_check)
    {
        // Obtain BSR nnzb twice, first using host pointer for nnzb and second using device pointer
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int hbsr_nnzb;
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                      direction,
                                                      M,
                                                      N,
                                                      csr_descr,
                                                      dcsr_row_ptr_C,
                                                      dcsr_col_ind_C,
                                                      bsr_descr,
                                                      dbsr_row_ptr,
                                                      row_block_dim,
                                                      col_block_dim,
                                                      &hbsr_nnzb,
                                                      (void*)dbuffer));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        // CHECK_HIP_ERROR(hipMemcpy(
        // hbsr_row_ptr, dbsr_row_ptr, sizeof(rocsparse_int) * (Mb + 1), hipMemcpyDeviceToHost));
        // std::cout << "disp " << std::endl;
        // for (int i=0;i<10;++i) std::cout << hbsr_row_ptr[i] << std::endl;

#if 0
        device_vector<rocsparse_int> dbsr_nnzb(1);
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                    direction,
                                                    M,
                                                    N,
                                                    csr_descr,
						      dcsr_row_ptr_C,
						      dcsr_col_ind_C,
						      bsr_descr,
						      dbsr_row_ptr,
						      row_block_dim,
						      col_block_dim,
						      dbsr_nnzb,
						      (void*)dbuffer));

        rocsparse_int hbsr_nnzb_copied_from_device = 0;
        CHECK_HIP_ERROR(hipMemcpy(&hbsr_nnzb_copied_from_device,
                                  dbsr_nnzb,
                                  sizeof(rocsparse_int),
                                  hipMemcpyDeviceToHost));

        // Confirm that nnzb is the same regardless of whether we use host or device pointers
        unit_check_general<rocsparse_int>(1, 1, 1, &hbsr_nnzb, &hbsr_nnzb_copied_from_device);

        // Allocate device memory for BSR col indices and values array
#endif
        device_vector<rocsparse_int> dbsr_col_ind(hbsr_nnzb);
        device_vector<T>             dbsr_val(hbsr_nnzb * row_block_dim * col_block_dim);

        // Finish conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr<T>(handle,
                                                     direction,
                                                     M,
                                                     N,
                                                     csr_descr,
                                                     dcsr_val_C,
                                                     dcsr_row_ptr_C,
                                                     dcsr_col_ind_C,
                                                     bsr_descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     row_block_dim,
                                                     col_block_dim,
                                                     (void*)dbuffer));

        // Allocate host memory for BSR col indices and values array
        host_vector<rocsparse_int> hbsr_col_ind(hbsr_nnzb);
        host_vector<T>             hbsr_val(hbsr_nnzb * row_block_dim * col_block_dim);

        // Copy BSR matrix output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_row_ptr, dbsr_row_ptr, sizeof(rocsparse_int) * (Mb + 1), hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind, dbsr_col_ind, sizeof(rocsparse_int) * hbsr_nnzb, hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipMemcpy(hbsr_val,
                                  dbsr_val,
                                  sizeof(T) * hbsr_nnzb * row_block_dim * col_block_dim,
                                  hipMemcpyDeviceToHost));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Convert BSR matrix back to CSR for comparison with original compressed CSR matrix
        M = Mb * row_block_dim;
        N = Nb * col_block_dim;

        device_vector<rocsparse_int> dcsr_row_ptr_gold_A(M + 1);
        device_vector<rocsparse_int> dcsr_col_ind_gold_A(hbsr_nnzb * row_block_dim * col_block_dim);
        device_vector<T>             dcsr_val_gold_A(hbsr_nnzb * row_block_dim * col_block_dim);
        CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2csr<T>(handle,
                                                     direction,
                                                     Mb,
                                                     Nb,
                                                     bsr_descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     row_block_dim,
                                                     col_block_dim,
                                                     csr_descr,
                                                     dcsr_val_gold_A,
                                                     dcsr_row_ptr_gold_A,
                                                     dcsr_col_ind_gold_A));

        // Compress the CSR matrix (the matrix may have retained zeros when we converted the BSR matrix back to CSR format)
        rocsparse_int                nnz_gold_C;
        device_vector<rocsparse_int> dnnz_per_row_gold(M);
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(handle,
                                                        M,
                                                        csr_descr,
                                                        dcsr_val_gold_A,
                                                        dcsr_row_ptr_gold_A,
                                                        dnnz_per_row_gold,
                                                        &nnz_gold_C,
                                                        tol));

        // Allocate device memory for the compressed version of the CSR matrix
        device_vector<rocsparse_int> dcsr_row_ptr_gold_C(M + 1);
        device_vector<rocsparse_int> dcsr_col_ind_gold_C(nnz_gold_C);
        device_vector<T>             dcsr_val_gold_C(nnz_gold_C);

        // Finish compression
        CHECK_ROCSPARSE_ERROR(
            rocsparse_csr2csr_compress<T>(handle,
                                          M,
                                          N,
                                          csr_descr,
                                          dcsr_val_gold_A,
                                          dcsr_row_ptr_gold_A,
                                          dcsr_col_ind_gold_A,
                                          hbsr_nnzb * row_block_dim * col_block_dim,
                                          dnnz_per_row_gold,
                                          dcsr_val_gold_C,
                                          dcsr_row_ptr_gold_C,
                                          dcsr_col_ind_gold_C,
                                          tol));

        // Allocate host memory for compressed CSR matrix
        host_vector<rocsparse_int> hcsr_row_ptr_gold_C(M + 1);
        host_vector<rocsparse_int> hcsr_col_ind_gold_C(nnz_gold_C);
        host_vector<T>             hcsr_val_gold_C(nnz_gold_C);

        // Copy compressed CSR matrix to host
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_gold_C,
                                  dcsr_row_ptr_C,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_gold_C,
                                  dcsr_col_ind_C,
                                  sizeof(rocsparse_int) * nnz_gold_C,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_gold_C, dcsr_val_C, sizeof(T) * nnz_gold_C, hipMemcpyDeviceToHost));

        // Compare with the original compressed CSR matrix. Note: The compressed CSR matrix we found when converting
        // from BSR back to CSR format may contain extra rows that are zero. Therefore just compare the rows found
        // in the original CSR matrix
        unit_check_general<rocsparse_int>(
            1, hcsr_row_ptr_C.size(), 1, hcsr_row_ptr_gold_C, hcsr_row_ptr_C);
        // for (int i=0;i<nnz_C;++i) std::cout << hcsr_col_ind_gold_C[i] << " " << hcsr_col_ind_C[i] << std::endl;
        unit_check_general<rocsparse_int>(1, nnz_C, 1, hcsr_col_ind_gold_C, hcsr_col_ind_C);
        // unit_check_general<T>(1, nnz_C, 1, hcsr_val_gold_C, hcsr_val_C);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int hbsr_nnzb;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                          direction,
                                                          M,
                                                          N,
                                                          csr_descr,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          bsr_descr,
                                                          dbsr_row_ptr,
                                                          row_block_dim,
                                                          col_block_dim,
                                                          &hbsr_nnzb,
                                                          (void*)dbuffer));

            // Allocate device memory for BSR col indices and values array
            device_vector<rocsparse_int> dbsr_col_ind(hbsr_nnzb);
            device_vector<T>             dbsr_val(hbsr_nnzb * row_block_dim * col_block_dim);

            CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr<T>(handle,
                                                         direction,
                                                         M,
                                                         N,
                                                         csr_descr,
                                                         dcsr_val_C,
                                                         dcsr_row_ptr_C,
                                                         dcsr_col_ind_C,
                                                         bsr_descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         row_block_dim,
                                                         col_block_dim,
                                                         (void*)dbuffer));
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                      direction,
                                                      M,
                                                      N,
                                                      csr_descr,
                                                      dcsr_row_ptr_C,
                                                      dcsr_col_ind_C,
                                                      bsr_descr,
                                                      dbsr_row_ptr,

                                                      row_block_dim,
                                                      col_block_dim,
                                                      &hbsr_nnzb,
                                                      (void*)dbuffer));

        // Allocate device memory for BSR col indices and values array
        device_vector<rocsparse_int> dbsr_col_ind(hbsr_nnzb);
        device_vector<T>             dbsr_val(hbsr_nnzb * row_block_dim * col_block_dim);

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr<T>(handle,
                                                         direction,
                                                         M,
                                                         N,
                                                         csr_descr,
                                                         dcsr_val_C,
                                                         dcsr_row_ptr_C,
                                                         dcsr_col_ind_C,

                                                         bsr_descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         row_block_dim,
                                                         col_block_dim,
                                                         (void*)dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte
            = csr2gebsr_gbyte_count<T>(M, Mb, nnz, hbsr_nnzb, row_block_dim, col_block_dim)
              / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "Mb"
                  << std::setw(12) << "Nb" << std::setw(12) << "rowblockdim" << std::setw(12)
                  << "colblockdim" << std::setw(12) << "nnzb" << std::setw(12) << "GB/s"
                  << std::setw(12) << "msec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << Mb
                  << std::setw(12) << Nb << std::setw(12) << row_block_dim << std::setw(12)
                  << col_block_dim << std::setw(12) << hbsr_nnzb << std::setw(12) << gpu_gbyte
                  << std::setw(12) << gpu_time_used / 1e3 << std::setw(12) << number_hot_calls
                  << std::setw(12) << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_csr2gebsr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2gebsr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
