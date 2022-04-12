/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
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
void testing_csr2hyb_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptors
    rocsparse_local_mat_descr local_descr;

    // Create hyb matrix
    rocsparse_local_hyb_mat local_hyb;

    rocsparse_handle          handle         = local_handle;
    rocsparse_int             m              = safe_size;
    rocsparse_int             n              = safe_size;
    const rocsparse_mat_descr descr          = local_descr;
    const T*                  csr_val        = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr    = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind    = (const rocsparse_int*)0x4;
    rocsparse_hyb_mat         hyb            = local_hyb;
    rocsparse_hyb_partition   partition_type = rocsparse_hyb_partition_auto;

    int           nargs_to_exclude   = 1;
    const int     args_to_exclude[1] = {8};
    rocsparse_int user_ell_width     = 0;

#define PARAMS \
    handle, m, n, descr, csr_val, csr_row_ptr, csr_col_ind, hyb, user_ell_width, partition_type
    auto_testing_bad_arg(rocsparse_csr2hyb<T>, nargs_to_exclude, args_to_exclude, PARAMS);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2hyb<T>(PARAMS), rocsparse_status_not_implemented);
#undef PARAMS
}

template <typename T>
void testing_csr2hyb(const Arguments& arg)
{

    // Sample matrix
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M              = arg.M;
    rocsparse_int               N              = arg.N;
    rocsparse_index_base        base           = arg.baseA;
    rocsparse_hyb_partition     part           = arg.part;
    rocsparse_int               user_ell_width = arg.algo;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create hyb matrix
    rocsparse_local_hyb_mat hyb;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);

        if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2hyb<T>(handle,
                                                     M,
                                                     N,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     hyb,
                                                     user_ell_width,
                                                     part),
                                (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                 : rocsparse_status_success);

        return;
    }

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;
    host_vector<rocsparse_int> hhyb_ell_col_ind_gold;
    host_vector<T>             hhyb_ell_val_gold;
    host_vector<rocsparse_int> hhyb_coo_row_ind_gold;
    host_vector<rocsparse_int> hhyb_coo_col_ind_gold;
    host_vector<T>             hhyb_coo_val_gold;

    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val, M, N, nnz, base);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // User width check
    if(part == rocsparse_hyb_partition_user)
    {
        // ELL width -33 means we take a reasonable pre-computed width
        user_ell_width = (user_ell_width == -33) ? nnz / M : user_ell_width;

        // Test invalid user_ell_width
        rocsparse_int max_allowed = (2 * nnz - 1) / M + 1;

        if(user_ell_width < 0 || user_ell_width > max_allowed)
        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_csr2hyb<T>(handle,
                                                         M,
                                                         N,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         hyb,
                                                         user_ell_width,
                                                         part),
                                    rocsparse_status_invalid_value);

            return;
        }
    }

    // Max width check
    if(part == rocsparse_hyb_partition_max)
    {
        // Compute max ELL width
        rocsparse_int ell_max_width = 0;
        for(rocsparse_int i = 0; i < M; ++i)
        {
            ell_max_width = std::max(hcsr_row_ptr[i + 1] - hcsr_row_ptr[i], ell_max_width);
        }

        rocsparse_int width_limit = (2 * nnz - 1) / M + 1;

        if(ell_max_width > width_limit)
        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_csr2hyb<T>(handle,
                                                         M,
                                                         N,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         hyb,
                                                         user_ell_width,
                                                         part),
                                    rocsparse_status_invalid_value);

            return;
        }
    }

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(
            handle, M, N, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, hyb, user_ell_width, part));

        // Copy output to host
        rocsparse_hyb_mat ptr  = hyb;
        test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

        rocsparse_int ell_nnz = dhyb->ell_nnz;
        rocsparse_int coo_nnz = dhyb->coo_nnz;

        host_vector<rocsparse_int> hhyb_ell_col_ind(ell_nnz);
        host_vector<T>             hhyb_ell_val(ell_nnz);
        host_vector<rocsparse_int> hhyb_coo_row_ind(coo_nnz);
        host_vector<rocsparse_int> hhyb_coo_col_ind(coo_nnz);
        host_vector<T>             hhyb_coo_val(coo_nnz);

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hhyb_ell_col_ind,
                                  dhyb->ell_col_ind,
                                  sizeof(rocsparse_int) * ell_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hhyb_ell_val, dhyb->ell_val, sizeof(T) * ell_nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_row_ind,
                                  dhyb->coo_row_ind,
                                  sizeof(rocsparse_int) * coo_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_col_ind,
                                  dhyb->coo_col_ind,
                                  sizeof(rocsparse_int) * coo_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hhyb_coo_val, dhyb->coo_val, sizeof(T) * coo_nnz, hipMemcpyDeviceToHost));

        // CPU csr2hyb
        rocsparse_int ell_width_gold = user_ell_width;
        rocsparse_int ell_nnz_gold;
        rocsparse_int coo_nnz_gold;

        host_csr_to_hyb<T>(M,
                           nnz,
                           hcsr_row_ptr,
                           hcsr_col_ind,
                           hcsr_val,
                           hhyb_ell_col_ind_gold,
                           hhyb_ell_val_gold,
                           ell_width_gold,
                           ell_nnz_gold,
                           hhyb_coo_row_ind_gold,
                           hhyb_coo_col_ind_gold,
                           hhyb_coo_val_gold,
                           coo_nnz_gold,
                           part,
                           base);

        unit_check_scalar<rocsparse_int>(M, dhyb->m);
        unit_check_scalar<rocsparse_int>(N, dhyb->n);
        unit_check_scalar<rocsparse_int>(ell_width_gold, dhyb->ell_width);
        unit_check_scalar<rocsparse_int>(ell_nnz_gold, dhyb->ell_nnz);
        unit_check_scalar<rocsparse_int>(coo_nnz_gold, dhyb->coo_nnz);
        hhyb_ell_col_ind_gold.unit_check(hhyb_ell_col_ind);
        hhyb_ell_val_gold.unit_check(hhyb_ell_val);
        hhyb_coo_row_ind_gold.unit_check(hhyb_coo_row_ind);
        hhyb_coo_col_ind_gold.unit_check(hhyb_coo_col_ind);
        hhyb_coo_val_gold.unit_check(hhyb_coo_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(handle,
                                                       M,
                                                       N,
                                                       descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind,
                                                       hyb,
                                                       user_ell_width,
                                                       part));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(handle,
                                                       M,
                                                       N,
                                                       descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind,
                                                       hyb,
                                                       user_ell_width,
                                                       part));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocsparse_hyb_mat ptr  = hyb;
        test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

        rocsparse_int ell_nnz = dhyb->ell_nnz;
        rocsparse_int coo_nnz = dhyb->coo_nnz;

        double gbyte_count = csr2hyb_gbyte_count<T>(M, nnz, ell_nnz, coo_nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "ELL nnz",
                            ell_nnz,
                            "COO nnz",
                            coo_nnz,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csr2hyb_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2hyb<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
