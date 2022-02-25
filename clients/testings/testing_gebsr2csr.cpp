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

#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

template <typename T>
void testing_gebsr2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create descriptors
    rocsparse_local_mat_descr local_bsr_descr;
    rocsparse_local_mat_descr local_csr_descr;

    rocsparse_handle          handle        = local_handle;
    rocsparse_direction       dir           = rocsparse_direction_row;
    rocsparse_int             mb            = safe_size;
    rocsparse_int             nb            = safe_size;
    const rocsparse_mat_descr bsr_descr     = local_bsr_descr;
    const T*                  bsr_val       = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr   = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind   = (const rocsparse_int*)0x4;
    rocsparse_int             row_block_dim = safe_size;
    rocsparse_int             col_block_dim = safe_size;
    const rocsparse_mat_descr csr_descr     = local_csr_descr;
    T*                        csr_val       = (T*)0x4;
    rocsparse_int*            csr_row_ptr   = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind   = (rocsparse_int*)0x4;

#define PARAMS                                                                        \
    handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, \
        col_block_dim, csr_descr, csr_val, csr_row_ptr, csr_col_ind
    auto_testing_bad_arg(rocsparse_gebsr2csr<T>, PARAMS);
#undef PARAMS
}

template <typename T>
void testing_gebsr2csr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M             = arg.M;
    rocsparse_int               N             = arg.N;
    rocsparse_direction         direction     = arg.direction;
    rocsparse_index_base        bsr_base      = arg.baseA;
    rocsparse_index_base        csr_base      = arg.baseB;
    rocsparse_int               row_block_dim = arg.row_block_dimA;
    rocsparse_int               col_block_dim = arg.col_block_dimA;

    rocsparse_int Mb = M;
    rocsparse_int Nb = N;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr bsr_descr;
    rocsparse_local_mat_descr csr_descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, bsr_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));

    // Argument sanity check before allocating invalid memory
    if(Mb <= 0 || Nb <= 0 || row_block_dim <= 0 || col_block_dim <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind(safe_size);
        device_vector<T>             dbsr_val(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);

        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
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
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind),
                                (Mb < 0 || Nb < 0 || row_block_dim <= 0 || col_block_dim <= 0)
                                    ? rocsparse_status_invalid_size
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

    rocsparse_int nnzb = 0;
    rocsparse_init_gebsr_matrix_from_csr(matrix_factory,
                                         hbsr_row_ptr,
                                         hbsr_col_ind,
                                         hbsr_val,
                                         direction,
                                         Mb,
                                         Nb,
                                         row_block_dim,
                                         col_block_dim,
                                         nnzb,
                                         bsr_base);

    M                 = Mb * row_block_dim;
    N                 = Nb * col_block_dim;
    rocsparse_int nnz = nnzb * row_block_dim * col_block_dim;
    // Allocate device memory for input BSR matrix
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);
    device_vector<rocsparse_int> dbsr_col_ind(nnzb);
    device_vector<T>             dbsr_val(nnz);

    // Allocate device memory for output CSR matrix
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (Mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val, hbsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        //
        // Compute on host.
        //
        host_vector<rocsparse_int> hcsr_ptr_ref(M + 1);
        host_vector<rocsparse_int> hcsr_ind_ref(nnz);
        host_vector<T>             hcsr_val_ref(nnz);

        host_gebsr_to_csr(direction,
                          Mb,
                          Nb,
                          nnzb,
                          hbsr_val,
                          hbsr_row_ptr,
                          hbsr_col_ind,
                          row_block_dim,
                          col_block_dim,
                          bsr_base,
                          hcsr_val_ref,
                          hcsr_ptr_ref,
                          hcsr_ind_ref,
                          csr_base);

        //
        // Check values of hcsr_val_ref, must be 1,2,3,4,5,6,7,...
        //
        for(rocsparse_int i = 0; i < nnzb * row_block_dim * col_block_dim; ++i)
        {
            T ref = static_cast<T>(i + 1);
            unit_check_scalar(hcsr_val_ref[i], ref);
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
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
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind));

        host_vector<T>             hcsr_val(nnz);
        host_vector<rocsparse_int> hcsr_ind(nnz);
        host_vector<rocsparse_int> hcsr_ptr(M + 1);

        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_ind, dcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_ptr, dcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_val, dcsr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        hcsr_ptr.unit_check(hcsr_ptr_ref);
        hcsr_ind.unit_check(hcsr_ind_ref);
        hcsr_val.unit_check(hcsr_val_ref);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
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
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
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
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = gebsr2csr_gbyte_count<T>(Mb, row_block_dim, col_block_dim, nnzb);

        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "Mb",
                            Mb,
                            "Nb",
                            Nb,
                            "row_blockdim",
                            row_block_dim,
                            "col_blockdim",
                            col_block_dim,
                            "nnzb",
                            nnzb,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_gebsr2csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gebsr2csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
