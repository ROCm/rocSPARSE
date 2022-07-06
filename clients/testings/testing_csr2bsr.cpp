/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(PARAMS_NNZ), rocsparse_status_not_implemented);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(PARAMS), rocsparse_status_not_implemented);

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

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, bsr_base));

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

    //
    // Initialize.
    //
    host_csr_matrix<T> hcsrA;
    matrix_factory.init_csr(hcsrA, M, N);

    //
    // Create and transfer.
    //
    device_csr_matrix<T> dcsrA(hcsrA);

    //
    // Compress dcsrA into dcsrC.
    //
    device_csr_matrix<T> dcsrC;
    rocsparse_matrix_utils::compress(dcsrC, dcsrA, dcsrA.base);

    //
    // Transfer dcsrC to host.
    //
    host_csr_matrix<T> hcsrC(dcsrC);

    // M and N can be modified in rocsparse_init_csr_matrix
    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb = (N + block_dim - 1) / block_dim;

    if(arg.unit_check)
    {
        //
        // Convert csr to bsr
        //
        device_gebsr_matrix<T> dbsr(direction, Mb, Nb, 0, block_dim, block_dim, bsr_base);

        // Obtain BSR nnzb twice, first using host pointer for nnzb and second using device pointer
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        host_scalar<rocsparse_int> hbsr_nnzb;
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                    direction,
                                                    M,
                                                    N,
                                                    csr_descr,
                                                    dcsrC.ptr, // dcsr_row_ptr_C,
                                                    dcsrC.ind, // dcsr_col_ind_C,
                                                    block_dim,
                                                    bsr_descr,
                                                    dbsr.ptr, // dbsr_row_ptr,
                                                    hbsr_nnzb));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        device_scalar<rocsparse_int> dbsr_nnzb;
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                    direction,
                                                    M,
                                                    N,
                                                    csr_descr,
                                                    dcsrC.ptr,
                                                    dcsrC.ind,
                                                    block_dim,
                                                    bsr_descr,
                                                    dbsr.ptr,
                                                    dbsr_nnzb));

        dbsr_nnzb.unit_check(hbsr_nnzb);

        dbsr.define(direction, Mb, Nb, *hbsr_nnzb, block_dim, block_dim, bsr_base);

        // Finish conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                                   direction,
                                                   M,
                                                   N,
                                                   csr_descr,
                                                   dcsrC.val, // dcsr_val_C,
                                                   dcsrC.ptr, // dcsr_row_ptr_C,
                                                   dcsrC.ind, // dcsr_col_ind_C,
                                                   block_dim,
                                                   bsr_descr,
                                                   dbsr.val, // dbsr_val,
                                                   dbsr.ptr, // dbsr_row_ptr,
                                                   dbsr.ind // dbsr_col_ind
                                                   ));

        //
        // Convert dbsr to dcsrD.
        //
        device_csr_matrix<T> dcsrD;

        {
            device_csr_matrix<T> dcsrE(dbsr.mb * block_dim,
                                       dbsr.nb * block_dim,
                                       dbsr.nnzb * block_dim * block_dim,
                                       csr_base);
            CHECK_ROCSPARSE_ERROR(rocsparse_bsr2csr<T>(handle,
                                                       dbsr.block_direction,
                                                       dbsr.mb,
                                                       dbsr.nb,
                                                       bsr_descr,
                                                       dbsr.val,
                                                       dbsr.ptr,
                                                       dbsr.ind,
                                                       block_dim,
                                                       csr_descr,
                                                       dcsrE.val,
                                                       dcsrE.ptr,
                                                       dcsrE.ind));

            //
            // Compress dcsrE to dcsrD.
            //
            rocsparse_matrix_utils::compress(dcsrD, dcsrE, dcsrE.base);
        }

        host_csr_matrix<T> hcsrD(dcsrD);

        // Compare with the original compressed CSR matrix. Note: The compressed CSR matrix we found when converting
        // from BSR back to CSR format may contain extra rows that are zero. Therefore just compare the rows found
        // in the original CSR matrix
        unit_check_segments<rocsparse_int>(hcsrC.ptr.size(), hcsrC.ptr, hcsrD.ptr);
        unit_check_segments<rocsparse_int>(hcsrC.ind.size(), hcsrC.ind, hcsrD.ind);
        unit_check_segments<T>(hcsrC.val.size(), hcsrC.val, hcsrD.val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        device_gebsr_matrix<T> dbsr(direction, Mb, Nb, 0, block_dim, block_dim, bsr_base);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        host_scalar<rocsparse_int> hbsr_nnzb;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                        direction,
                                                        M,
                                                        N,
                                                        csr_descr,
                                                        dcsrC.ptr,
                                                        dcsrC.ind,
                                                        block_dim,
                                                        bsr_descr,
                                                        dbsr.ptr,
                                                        hbsr_nnzb));

            //redefine dbsr
            dbsr.define(direction, Mb, Nb, *hbsr_nnzb, block_dim, block_dim, bsr_base);

            CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                                       direction,
                                                       M,
                                                       N,
                                                       csr_descr,
                                                       dcsrC.val, // dcsr_val_C,
                                                       dcsrC.ptr, // dcsr_row_ptr_C,
                                                       dcsrC.ind, // dcsr_col_ind_C,
                                                       block_dim,
                                                       bsr_descr,
                                                       dbsr.val, // dbsr_val,
                                                       dbsr.ptr, // dbsr_row_ptr,
                                                       dbsr.ind // dbsr_col_ind
                                                       ));
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                    direction,
                                                    M,
                                                    N,
                                                    csr_descr,
                                                    dcsrC.ptr,
                                                    dcsrC.ind,
                                                    block_dim,
                                                    bsr_descr,
                                                    dbsr.ptr,
                                                    hbsr_nnzb));

        //redefine dbsr
        dbsr.define(direction, Mb, Nb, *hbsr_nnzb, block_dim, block_dim, bsr_base);

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                                       direction,
                                                       M,
                                                       N,
                                                       csr_descr,
                                                       dcsrC.val,
                                                       dcsrC.ptr,
                                                       dcsrC.ind,
                                                       block_dim,
                                                       bsr_descr,
                                                       dbsr.val,
                                                       dbsr.ptr,
                                                       dbsr.ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csr2bsr_gbyte_count<T>(M, Mb, hcsrA.nnz, *hbsr_nnzb, block_dim);
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
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csr2bsr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2bsr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
