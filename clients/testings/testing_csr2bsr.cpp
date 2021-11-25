/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_csr2bsr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptors
    rocsparse_local_mat_descr local_csr_descr;
    rocsparse_local_mat_descr local_bsr_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_direction       dir         = rocsparse_direction_row;
    rocsparse_int             m           = safe_size;
    rocsparse_int             n           = safe_size;
    const rocsparse_mat_descr csr_descr   = local_csr_descr;
    const T*                  csr_val     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_int             block_dim   = safe_size;
    const rocsparse_mat_descr bsr_descr   = local_bsr_descr;
    T*                        bsr_val     = (T*)0x4;
    rocsparse_int*            bsr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            bsr_col_ind = (rocsparse_int*)0x4;
    rocsparse_int*            bsr_nnz     = (rocsparse_int*)0x4;

    int       nargs_to_exclude   = 1;
    const int args_to_exclude[1] = {6};

#define PARAMS_NNZ                                                                             \
    handle, dir, m, n, csr_descr, csr_row_ptr, csr_col_ind, block_dim, bsr_descr, bsr_row_ptr, \
        bsr_nnz
#define PARAMS                                                                             \
    handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, csr_col_ind, block_dim, bsr_descr, \
        bsr_val, bsr_row_ptr, bsr_col_ind
    auto_testing_bad_arg(rocsparse_csr2bsr_nnz, nargs_to_exclude, args_to_exclude, PARAMS_NNZ);
    auto_testing_bad_arg(rocsparse_csr2bsr<T>, PARAMS);
#undef PARAMS
#undef PARAMS_NNZ
}

template <typename T>
void testing_csr2bsr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M         = arg.M;
    rocsparse_int               N         = arg.N;
    rocsparse_index_base        csr_base  = arg.baseA;
    rocsparse_index_base        bsr_base  = arg.baseB;
    rocsparse_direction         direction = arg.direction;
    rocsparse_int               block_dim = arg.block_dim;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_descr bsr_descr;

    rocsparse_set_mat_index_base(csr_descr, csr_base);
    rocsparse_set_mat_index_base(bsr_descr, bsr_base);

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || block_dim <= 0)
    {
        static const size_t safe_size = 100;
        rocsparse_int       hbsr_nnzb;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);
        device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind(safe_size);
        device_vector<T>             dbsr_val(safe_size);

        if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dbsr_row_ptr || !dbsr_col_ind
           || !dbsr_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                      rocsparse_direction_row,
                                                      M,
                                                      N,
                                                      csr_descr,
                                                      dcsr_row_ptr,
                                                      dcsr_col_ind,
                                                      block_dim,
                                                      bsr_descr,
                                                      dbsr_row_ptr,
                                                      &hbsr_nnzb),
                                (M < 0 || N < 0 || block_dim < 0) ? rocsparse_status_invalid_size
                                                                  : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                     direction,
                                                     M,
                                                     N,
                                                     csr_descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     block_dim,
                                                     bsr_descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind),
                                (M < 0 || N < 0 || block_dim < 0) ? rocsparse_status_invalid_size
                                                                  : rocsparse_status_success);

        return;
    }

    // Allocate host memory for uncompressed CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_A;
    host_vector<rocsparse_int> hcsr_col_ind_A;
    host_vector<T>             hcsr_val_A;

    // Generate (or load from file) uncompressed CSR matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, M, N, nnz, csr_base);

    // Uncompressed CSR matrix on device
    device_vector<rocsparse_int> dcsr_row_ptr_A(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_A(nnz);
    device_vector<T>             dcsr_val_A(nnz);

    if(!dcsr_row_ptr_A)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    if((nnz > 0) && (!dcsr_col_ind_A || !dcsr_val_A))
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

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

    if(!dnnz_per_row)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
        handle, M, csr_descr, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &nnz_C, tol));

    // Allocate device memory for the compressed version of the CSR matrix
    device_vector<rocsparse_int> dcsr_row_ptr_C(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_C(nnz_C);
    device_vector<T>             dcsr_val_C(nnz_C);

    if(!dcsr_row_ptr_C)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    if((nnz_C > 0) && (!dcsr_col_ind_C || !dcsr_val_C))
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

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
    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb = (N + block_dim - 1) / block_dim;

    // Allocate host memory for BSR row ptr array
    host_vector<rocsparse_int> hbsr_row_ptr(Mb + 1);

    // Allocate device memory for BSR row ptr array
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);

    if(!dbsr_row_ptr)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    if(arg.unit_check)
    {
        // Obtain BSR nnzb twice, first using host pointer for nnzb and second using device pointer
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int hbsr_nnzb;
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
                                                    &hbsr_nnzb));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        device_vector<rocsparse_int> dbsr_nnzb(1);
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
                                                    dbsr_nnzb));

        rocsparse_int hbsr_nnzb_copied_from_device = 0;
        CHECK_HIP_ERROR(hipMemcpy(&hbsr_nnzb_copied_from_device,
                                  dbsr_nnzb,
                                  sizeof(rocsparse_int),
                                  hipMemcpyDeviceToHost));

        // Confirm that nnzb is the same regardless of whether we use host or device pointers
        unit_check_scalar(hbsr_nnzb, hbsr_nnzb_copied_from_device);

        // Allocate device memory for BSR col indices and values array
        device_vector<rocsparse_int> dbsr_col_ind(hbsr_nnzb);
        device_vector<T>             dbsr_val(hbsr_nnzb * block_dim * block_dim);

        if((hbsr_nnzb > 0) && (!dbsr_col_ind || !dbsr_val))
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Finish conversion
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

        // Allocate host memory for BSR col indices and values array
        host_vector<rocsparse_int> hbsr_col_ind(hbsr_nnzb);
        host_vector<T>             hbsr_val(hbsr_nnzb * block_dim * block_dim);

        // Copy BSR matrix output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_row_ptr, dbsr_row_ptr, sizeof(rocsparse_int) * (Mb + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind, dbsr_col_ind, sizeof(rocsparse_int) * hbsr_nnzb, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val,
                                  dbsr_val,
                                  sizeof(T) * hbsr_nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Convert BSR matrix back to CSR for comparison with original compressed CSR matrix
        M = Mb * block_dim;
        N = Nb * block_dim;

        device_vector<rocsparse_int> dcsr_row_ptr_gold_A(M + 1);
        device_vector<rocsparse_int> dcsr_col_ind_gold_A(hbsr_nnzb * block_dim * block_dim);
        device_vector<T>             dcsr_val_gold_A(hbsr_nnzb * block_dim * block_dim);

        if(!dcsr_row_ptr_gold_A)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        if((hbsr_nnzb > 0) && (!dcsr_col_ind_gold_A || !dcsr_val_gold_A))
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

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
                                                   dcsr_val_gold_A,
                                                   dcsr_row_ptr_gold_A,
                                                   dcsr_col_ind_gold_A));

        // Compress the CSR matrix (the matrix may have retained zeros when we converted the BSR matrix back to CSR format)
        rocsparse_int                nnz_gold_C;
        device_vector<rocsparse_int> dnnz_per_row_gold(M);

        if((M > 0) && !dnnz_per_row_gold)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

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

        if(!dcsr_row_ptr_gold_C)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        if((nnz_gold_C > 0) && (!dcsr_col_ind_gold_C || !dcsr_val_gold_C))
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Finish compression
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2csr_compress<T>(handle,
                                                            M,
                                                            N,
                                                            csr_descr,
                                                            dcsr_val_gold_A,
                                                            dcsr_row_ptr_gold_A,
                                                            dcsr_col_ind_gold_A,
                                                            hbsr_nnzb * block_dim * block_dim,
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
        unit_check_segments<rocsparse_int>(
            hcsr_row_ptr_C.size(), hcsr_row_ptr_gold_C, hcsr_row_ptr_C);
        unit_check_segments<rocsparse_int>(
            hcsr_col_ind_C.size(), hcsr_col_ind_gold_C, hcsr_col_ind_C);
        unit_check_segments<T>(hcsr_val_C.size(), hcsr_val_gold_C, hcsr_val_C);
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
                                                        &hbsr_nnzb));

            // Allocate device memory for BSR col indices and values array
            device_vector<rocsparse_int> dbsr_col_ind(hbsr_nnzb);
            device_vector<T>             dbsr_val(hbsr_nnzb * block_dim * block_dim);

            if((hbsr_nnzb > 0) && (!dbsr_col_ind || !dbsr_val))
            {
                CHECK_HIP_ERROR(hipErrorOutOfMemory);
                return;
            }

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
        }

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
                                                    &hbsr_nnzb));

        // Allocate device memory for BSR col indices and values array
        device_vector<rocsparse_int> dbsr_col_ind(hbsr_nnzb);
        device_vector<T>             dbsr_val(hbsr_nnzb * block_dim * block_dim);

        if((hbsr_nnzb > 0) && (!dbsr_col_ind || !dbsr_val))
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
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
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csr2bsr_gbyte_count<T>(M, Mb, nnz, hbsr_nnzb, block_dim);
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
                            hbsr_nnzb,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used),
                            "iter",
                            number_hot_calls,
                            "verified",
                            arg.unit_check ? "yes" : "no");
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csr2bsr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2bsr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
