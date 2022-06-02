/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#include "rocsparse_enum.hpp"
#include "testing.hpp"

#include "auto_testing_bad_arg.hpp"

template <typename T>
inline void rocsparse_init_csr_and_bsr_matrix(const Arguments&            arg,
                                              std::vector<rocsparse_int>& csr_row_ptr,
                                              std::vector<rocsparse_int>& csr_col_ind,
                                              std::vector<T>&             csr_val,
                                              rocsparse_int&              M,
                                              rocsparse_int&              N,
                                              rocsparse_index_base        csr_base,
                                              std::vector<rocsparse_int>& bsr_row_ptr,
                                              std::vector<rocsparse_int>& bsr_col_ind,
                                              std::vector<T>&             bsr_val,
                                              rocsparse_direction         direction,
                                              rocsparse_int&              Mb,
                                              rocsparse_int&              Nb,
                                              rocsparse_int               block_dim,
                                              rocsparse_int&              nnzb,
                                              rocsparse_index_base        bsr_base)
{
    // Matrix handle and descriptors used for conversion
    rocsparse_local_handle handle;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_descr bsr_descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, bsr_base));

    // Uncompressed CSR matrix on host
    host_vector<rocsparse_int> hcsr_row_ptr_A;
    host_vector<rocsparse_int> hcsr_col_ind_A;
    host_vector<T>             hcsr_val_A;

    // Generate uncompressed CSR matrix on host (or read from file)
    rocsparse_int nnz = 0;

    rocsparse_matrix_factory<T> factory(arg);
    factory.init_csr(hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, M, N, nnz, csr_base);

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
    csr_row_ptr.resize(M + 1);
    csr_col_ind.resize(nnz_C);
    csr_val.resize(nnz_C);

    // Copy compressed CSR matrix to host
    CHECK_HIP_ERROR(hipMemcpy(csr_row_ptr.data(),
                              dcsr_row_ptr_C,
                              sizeof(rocsparse_int) * (M + 1),
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        csr_col_ind.data(), dcsr_col_ind_C, sizeof(rocsparse_int) * nnz_C, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(csr_val.data(), dcsr_val_C, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

    // M and N can be modified by rocsparse_init_csr_matrix if reading from a file
    Mb = (M + block_dim - 1) / block_dim;
    Nb = (N + block_dim - 1) / block_dim;

    // Allocate device memory for BSR row pointer array
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);

    // Convert sample CSR matrix to bsr
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                direction,
                                                M,
                                                N,
                                                csr_descr,
                                                dcsr_row_ptr_C,
                                                dcsr_col_ind_C,
                                                block_dim,
                                                bsr_descr,
                                                dbsr_row_ptr,
                                                &nnzb));

    // Allocate device memory for BSR col indices and values array
    device_vector<rocsparse_int> dbsr_col_ind(nnzb);
    device_vector<T>             dbsr_val(size_t(nnzb) * block_dim * block_dim);

    CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                               direction,
                                               M,
                                               N,
                                               csr_descr,
                                               dcsr_val_C,
                                               dcsr_row_ptr_C,
                                               dcsr_col_ind_C,
                                               block_dim,
                                               bsr_descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind));

    // Resize BSR arrays
    bsr_row_ptr.resize(Mb + 1);
    bsr_col_ind.resize(nnzb);
    bsr_val.resize(size_t(nnzb) * block_dim * block_dim);

    // Copy BSR matrix output to host
    CHECK_HIP_ERROR(hipMemcpy(
        bsr_row_ptr.data(), dbsr_row_ptr, sizeof(rocsparse_int) * (Mb + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        bsr_col_ind.data(), dbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        bsr_val.data(), dbsr_val, sizeof(T) * nnzb * block_dim * block_dim, hipMemcpyDeviceToHost));
}

template <typename T>
void testing_bsr2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create rocsparse decriptors
    rocsparse_local_mat_descr local_bsr_descr;
    rocsparse_local_mat_descr local_csr_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_direction       dir         = rocsparse_direction_row;
    rocsparse_int             mb          = safe_size;
    rocsparse_int             nb          = safe_size;
    const rocsparse_mat_descr bsr_descr   = local_bsr_descr;
    const T*                  bsr_val     = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_int             block_dim   = safe_size;
    const rocsparse_mat_descr csr_descr   = local_csr_descr;
    T*                        csr_val     = (T*)0x4;
    rocsparse_int*            csr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind = (rocsparse_int*)0x4;

#define PARAMS                                                                               \
    handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, csr_descr, \
        csr_val, csr_row_ptr, csr_col_ind

    auto_testing_bad_arg(rocsparse_bsr2csr<T>, PARAMS);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(PARAMS), rocsparse_status_not_implemented);

#undef PARAMS
}

template <typename T>
void testing_bsr2csr(const Arguments& arg)
{
    static constexpr bool       toint     = false;
    static constexpr bool       full_rank = true;
    rocsparse_matrix_factory<T> matrix_factory(arg, toint, full_rank);

    rocsparse_int        M         = arg.M;
    rocsparse_int        N         = arg.N;
    rocsparse_index_base bsr_base  = arg.baseA;
    rocsparse_index_base csr_base  = arg.baseB;
    rocsparse_direction  direction = arg.direction;
    rocsparse_int        block_dim = arg.block_dim;

    rocsparse_int Mb = -1;
    rocsparse_int Nb = -1;
    if(block_dim > 0)
    {
        Mb = (M + block_dim - 1) / block_dim;
        Nb = (N + block_dim - 1) / block_dim;
    }

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr bsr_descr;
    rocsparse_local_mat_descr csr_descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, bsr_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));

    // Argument sanity check before allocating invalid memory
    if(Mb <= 0 || Nb <= 0 || block_dim <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind(safe_size);
        device_vector<T>             dbsr_val(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                     direction,
                                                     Mb,
                                                     Nb,
                                                     bsr_descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     block_dim,
                                                     csr_descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind),
                                (Mb < 0 || Nb < 0 || block_dim <= 0) ? rocsparse_status_invalid_size
                                                                     : rocsparse_status_success);

        return;
    }

    // Allocate host memory for original CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_orig;
    host_vector<rocsparse_int> hcsr_col_ind_orig;
    host_vector<T>             hcsr_val_orig;

    // Allocate host memory for output BSR matrix
    host_vector<rocsparse_int> hbsr_row_ptr;
    host_vector<rocsparse_int> hbsr_col_ind;
    host_vector<T>             hbsr_val;

    // Generate original host CSR matrix and then use it to fill in the host BSR matrix
    rocsparse_int nnzb = 0;
    rocsparse_init_csr_and_bsr_matrix(arg,
                                      hcsr_row_ptr_orig,
                                      hcsr_col_ind_orig,
                                      hcsr_val_orig,
                                      M,
                                      N,
                                      csr_base,
                                      hbsr_row_ptr,
                                      hbsr_col_ind,
                                      hbsr_val,
                                      direction,
                                      Mb,
                                      Nb,
                                      block_dim,
                                      nnzb,
                                      bsr_base);

    // M and N and Mb and Nb can be modified by rocsparse_init_csr_and_bsr_matrix
    M = Mb * block_dim;
    N = Nb * block_dim;

    // Allocate device memory for input BSR matrix
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);
    device_vector<rocsparse_int> dbsr_col_ind(nnzb);
    device_vector<T>             dbsr_val(size_t(nnzb) * block_dim * block_dim);

    // Allocate device memory for output CSR matrix
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(size_t(nnzb) * block_dim * block_dim);
    device_vector<T>             dcsr_val(size_t(nnzb) * block_dim * block_dim);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (Mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_val, hbsr_val, sizeof(T) * nnzb * block_dim * block_dim, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(rocsparse_bsr2csr<T>(handle,
                                                   direction,
                                                   Mb,
                                                   Nb,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   block_dim,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind));

        // Compress the output CSR matrix
        T                            tol = static_cast<T>(0);
        rocsparse_int                hnnz_C;
        device_vector<rocsparse_int> dnnz_per_row(M);

        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
            handle, M, csr_descr, dcsr_val, dcsr_row_ptr, dnnz_per_row, &hnnz_C, tol));

        // Allocate device memory for the compressed version of the CSR matrix
        device_vector<rocsparse_int> dcsr_row_ptr_C(M + 1);
        device_vector<rocsparse_int> dcsr_col_ind_C(hnnz_C);
        device_vector<T>             dcsr_val_C(hnnz_C);

        // Finish compression
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2csr_compress<T>(handle,
                                                            M,
                                                            N,
                                                            csr_descr,
                                                            dcsr_val,
                                                            dcsr_row_ptr,
                                                            dcsr_col_ind,
                                                            nnzb * block_dim * block_dim,
                                                            dnnz_per_row,
                                                            dcsr_val_C,
                                                            dcsr_row_ptr_C,
                                                            dcsr_col_ind_C,
                                                            tol));

        // Allocate host memory for compressed CSR matrix
        host_vector<rocsparse_int> hcsr_row_ptr_C(M + 1);
        host_vector<rocsparse_int> hcsr_col_ind_C(hnnz_C);
        host_vector<T>             hcsr_val_C(hnnz_C);

        // Copy compressed CSR matrix to host
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C,
                                  dcsr_row_ptr_C,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C, dcsr_col_ind_C, sizeof(rocsparse_int) * hnnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C, dcsr_val_C, sizeof(T) * hnnz_C, hipMemcpyDeviceToHost));

        // Compare compressed output CSR matrix with the original compressed CSR matrix
        // Note: We may have added rows to the compressed output CSR matrix, but these extra
        // rows will be zeros. Therefore only check rows up to the the size of the original
        // compressed CSR matrix.
        unit_check_segments<rocsparse_int>(
            hcsr_row_ptr_orig.size(), hcsr_row_ptr_orig, hcsr_row_ptr_C);
        unit_check_segments<rocsparse_int>(
            hcsr_col_ind_orig.size(), hcsr_col_ind_orig, hcsr_col_ind_C);
        unit_check_segments<T>(hcsr_val_orig.size(), hcsr_val_orig, hcsr_val_C);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsr2csr<T>(handle,
                                                       direction,
                                                       Mb,
                                                       Nb,
                                                       bsr_descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       block_dim,
                                                       csr_descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsr2csr<T>(handle,
                                                       direction,
                                                       Mb,
                                                       Nb,
                                                       bsr_descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       block_dim,
                                                       csr_descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = bsr2csr_gbyte_count<T>(Mb, block_dim, nnzb);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "Mb",
                            Mb,
                            "Nb",
                            Nb,
                            "blockdim",
                            block_dim,
                            "nnzb",
                            nnzb,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_bsr2csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsr2csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
#undef INSTANTIATE
