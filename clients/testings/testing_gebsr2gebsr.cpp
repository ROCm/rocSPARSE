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
#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

template <typename T>
void testing_gebsr2gebsr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create descriptors
    rocsparse_local_mat_descr local_descr_A;
    rocsparse_local_mat_descr local_descr_C;

    rocsparse_handle          handle          = local_handle;
    rocsparse_direction       dir             = rocsparse_direction_row;
    rocsparse_int             mb              = safe_size;
    rocsparse_int             nb              = safe_size;
    rocsparse_int             nnzb            = safe_size;
    const rocsparse_mat_descr descr_A         = local_descr_A;
    const T*                  bsr_val_A       = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr_A   = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind_A   = (const rocsparse_int*)0x4;
    rocsparse_int             row_block_dim_A = safe_size;
    rocsparse_int             col_block_dim_A = safe_size;
    const rocsparse_mat_descr descr_C         = local_descr_C;
    T*                        bsr_val_C       = (T*)0x4;
    rocsparse_int*            bsr_row_ptr_C   = (rocsparse_int*)0x4;
    rocsparse_int*            bsr_col_ind_C   = (rocsparse_int*)0x4;
    rocsparse_int             row_block_dim_C = safe_size;
    rocsparse_int             col_block_dim_C = safe_size;
    size_t*                   buffer_size     = (size_t*)0x4;
    void*                     temp_buffer     = (void*)0x4;

#define PARAMS_BUFFER_SIZE                                                                        \
    handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, bsr_col_ind_A, row_block_dim_A, \
        col_block_dim_A, row_block_dim_C, col_block_dim_C, buffer_size
#define PARAMS                                                                                    \
    handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, bsr_col_ind_A, row_block_dim_A, \
        col_block_dim_A, descr_C, bsr_val_C, bsr_row_ptr_C, bsr_col_ind_C, row_block_dim_C,       \
        col_block_dim_C, temp_buffer
    auto_testing_bad_arg(rocsparse_gebsr2gebsr_buffer_size<T>, PARAMS_BUFFER_SIZE);
    auto_testing_bad_arg(rocsparse_gebsr2gebsr<T>, PARAMS);
#undef PARAMS
#undef PARAMS_BUFFER_SIZE
}

template <typename T>
void testing_gebsr2gebsr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M               = arg.M;
    rocsparse_int               N               = arg.N;
    rocsparse_index_base        base_A          = arg.baseA;
    rocsparse_index_base        base_C          = arg.baseB;
    rocsparse_direction         direction       = arg.direction;
    rocsparse_int               row_block_dim_A = arg.row_block_dimA;
    rocsparse_int               col_block_dim_A = arg.col_block_dimA;
    rocsparse_int               row_block_dim_C = arg.row_block_dimB;
    rocsparse_int               col_block_dim_C = arg.col_block_dimB;

    rocsparse_int Mb = -1;
    rocsparse_int Nb = -1;
    if(row_block_dim_A > 0 && col_block_dim_A > 0)
    {
        Mb = (M + row_block_dim_A - 1) / row_block_dim_A;
        Nb = (N + col_block_dim_A - 1) / col_block_dim_A;
    }

    M = Mb * row_block_dim_A;
    N = Nb * col_block_dim_A;

    rocsparse_int Mb_C = (M + row_block_dim_C - 1) / row_block_dim_C;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr descr_A;
    rocsparse_local_mat_descr descr_C;

    rocsparse_set_mat_index_base(descr_A, base_A);
    rocsparse_set_mat_index_base(descr_C, base_C);

    // Argument sanity check before allocating invalid memory
    if(Mb <= 0 || Nb <= 0 || row_block_dim_A <= 0 || col_block_dim_A <= 0 || row_block_dim_C <= 0
       || col_block_dim_C <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dbsr_row_ptr_A(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind_A(safe_size);
        device_vector<T>             dbsr_val_A(safe_size);
        device_vector<rocsparse_int> dbsr_row_ptr_C(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind_C(safe_size);
        device_vector<T>             dbsr_val_C(safe_size);

        if(!dbsr_row_ptr_A || !dbsr_col_ind_A || !dbsr_val_A || !dbsr_row_ptr_C || !dbsr_col_ind_C
           || !dbsr_val_C)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsr<T>(handle,
                                                         rocsparse_direction_row,
                                                         Mb,
                                                         Nb,
                                                         safe_size,
                                                         descr_A,
                                                         dbsr_val_A,
                                                         dbsr_row_ptr_A,
                                                         dbsr_col_ind_A,
                                                         row_block_dim_A,
                                                         col_block_dim_A,
                                                         descr_C,
                                                         dbsr_val_C,
                                                         dbsr_row_ptr_C,
                                                         dbsr_col_ind_C,
                                                         row_block_dim_C,
                                                         col_block_dim_C,
                                                         nullptr),
                                (Mb < 0 || Nb < 0 || row_block_dim_A <= 0 || col_block_dim_A <= 0
                                 || row_block_dim_C <= 0 || col_block_dim_C <= 0)
                                    ? rocsparse_status_invalid_size
                                    : rocsparse_status_success);

        return;
    }

    // Allocate host memory for input BSR matrix
    host_vector<rocsparse_int> hbsr_row_ptr_A;
    host_vector<rocsparse_int> hbsr_col_ind_A;
    host_vector<T>             hbsr_val_A;

    rocsparse_int hnnzb_A = 0;
    rocsparse_init_gebsr_matrix_from_csr(matrix_factory,
                                         hbsr_row_ptr_A,
                                         hbsr_col_ind_A,
                                         hbsr_val_A,
                                         direction,
                                         Mb,
                                         Nb,
                                         row_block_dim_A,
                                         col_block_dim_A,
                                         hnnzb_A,
                                         base_A);

    // Mb and Nb can be modified by rocsparse_init_csr_matrix if reading from a file
    M    = Mb * row_block_dim_A;
    N    = Nb * col_block_dim_A;
    Mb_C = (M + row_block_dim_C - 1) / row_block_dim_C;

    // Allocate device memory for input BSR matrix
    device_vector<rocsparse_int> dbsr_row_ptr_A(Mb + 1);
    device_vector<rocsparse_int> dbsr_col_ind_A(hnnzb_A);
    device_vector<T>             dbsr_val_A(hnnzb_A * row_block_dim_A * col_block_dim_A);

    // Allocate device memory for output BSR row pointer array
    device_vector<rocsparse_int> dbsr_row_ptr_C(Mb_C + 1);

    if(!dbsr_row_ptr_A || !dbsr_col_ind_A || !dbsr_val_A || !dbsr_row_ptr_C)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dbsr_row_ptr_A,
                              hbsr_row_ptr_A.data(),
                              sizeof(rocsparse_int) * (Mb + 1),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_col_ind_A,
                              hbsr_col_ind_A.data(),
                              sizeof(rocsparse_int) * hnnzb_A,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val_A,
                              hbsr_val_A.data(),
                              sizeof(T) * hnnzb_A * row_block_dim_A * col_block_dim_A,
                              hipMemcpyHostToDevice));

    size_t buffer_size = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsr_buffer_size<T>(handle,
                                                               direction,
                                                               Mb,
                                                               Nb,
                                                               hnnzb_A,
                                                               descr_A,
                                                               dbsr_val_A,
                                                               dbsr_row_ptr_A,
                                                               dbsr_col_ind_A,
                                                               row_block_dim_A,
                                                               col_block_dim_A,
                                                               row_block_dim_C,
                                                               col_block_dim_C,
                                                               &buffer_size));

    T* dtemp_buffer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&dtemp_buffer, buffer_size));

    if(!dtemp_buffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    host_vector<rocsparse_int> hnnzb_C(1);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsr_nnz(handle,
                                                    direction,
                                                    Mb,
                                                    Nb,
                                                    hnnzb_A,
                                                    descr_A,
                                                    dbsr_row_ptr_A,
                                                    dbsr_col_ind_A,
                                                    row_block_dim_A,
                                                    col_block_dim_A,
                                                    descr_C,
                                                    dbsr_row_ptr_C,
                                                    row_block_dim_C,
                                                    col_block_dim_C,
                                                    hnnzb_C,
                                                    dtemp_buffer));

    device_vector<rocsparse_int> dnnzb_C(1);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsr_nnz(handle,
                                                    direction,
                                                    Mb,
                                                    Nb,
                                                    hnnzb_A,
                                                    descr_A,
                                                    dbsr_row_ptr_A,
                                                    dbsr_col_ind_A,
                                                    row_block_dim_A,
                                                    col_block_dim_A,
                                                    descr_C,
                                                    dbsr_row_ptr_C,
                                                    row_block_dim_C,
                                                    col_block_dim_C,
                                                    dnnzb_C,
                                                    dtemp_buffer));

    device_vector<rocsparse_int> dbsr_col_ind_C(hnnzb_C[0]);
    device_vector<T>             dbsr_val_C(hnnzb_C[0] * row_block_dim_C * col_block_dim_C);

    if(!dbsr_col_ind_C || !dbsr_val_C)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    if(arg.unit_check)
    {
        host_vector<rocsparse_int> hnnzb_C_copied_from_device(1);
        CHECK_HIP_ERROR(hipMemcpy(
            hnnzb_C_copied_from_device, dnnzb_C, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        hnnzb_C.unit_check(hnnzb_C_copied_from_device);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsr<T>(handle,
                                                       direction,
                                                       Mb,
                                                       Nb,
                                                       hnnzb_A,
                                                       descr_A,
                                                       dbsr_val_A,
                                                       dbsr_row_ptr_A,
                                                       dbsr_col_ind_A,
                                                       row_block_dim_A,
                                                       col_block_dim_A,
                                                       descr_C,
                                                       dbsr_val_C,
                                                       dbsr_row_ptr_C,
                                                       dbsr_col_ind_C,
                                                       row_block_dim_C,
                                                       col_block_dim_C,
                                                       dtemp_buffer));

        host_vector<rocsparse_int> hbsr_row_ptr_C(Mb_C + 1);
        host_vector<rocsparse_int> hbsr_col_ind_C(hnnzb_C[0]);
        host_vector<T>             hbsr_val_C(hnnzb_C[0] * row_block_dim_C * col_block_dim_C);

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hbsr_row_ptr_C,
                                  dbsr_row_ptr_C,
                                  sizeof(rocsparse_int) * (Mb_C + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_col_ind_C,
                                  dbsr_col_ind_C,
                                  sizeof(rocsparse_int) * hnnzb_C[0],
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val_C,
                                  dbsr_val_C,
                                  sizeof(T) * hnnzb_C[0] * row_block_dim_C * col_block_dim_C,
                                  hipMemcpyDeviceToHost));

        // call host and check results
        host_vector<rocsparse_int> hbsr_row_ptr_cpu;
        host_vector<rocsparse_int> hbsr_col_ind_cpu;
        host_vector<T>             hbsr_val_cpu;
        host_vector<rocsparse_int> hnnzb_cpu(1);

        host_gebsr_to_gebsr(direction,
                            Mb,
                            Nb,
                            hnnzb_A,
                            hbsr_val_A,
                            hbsr_row_ptr_A,
                            hbsr_col_ind_A,
                            row_block_dim_A,
                            col_block_dim_A,
                            base_A,
                            hbsr_val_cpu,
                            hbsr_row_ptr_cpu,
                            hbsr_col_ind_cpu,
                            row_block_dim_C,
                            col_block_dim_C,
                            base_C);

        hnnzb_cpu[0] = hbsr_col_ind_cpu.size();

        hnnzb_cpu.unit_check(hnnzb_C);
        hbsr_row_ptr_cpu.unit_check(hbsr_row_ptr_C);
        hbsr_col_ind_cpu.unit_check(hbsr_col_ind_C);
        hbsr_val_cpu.unit_check(hbsr_val_C);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsr<T>(handle,
                                                           direction,
                                                           Mb,
                                                           Nb,
                                                           hnnzb_A,
                                                           descr_A,
                                                           dbsr_val_A,
                                                           dbsr_row_ptr_A,
                                                           dbsr_col_ind_A,
                                                           row_block_dim_A,
                                                           col_block_dim_A,
                                                           descr_C,
                                                           dbsr_val_C,
                                                           dbsr_row_ptr_C,
                                                           dbsr_col_ind_C,
                                                           row_block_dim_C,
                                                           col_block_dim_C,
                                                           dtemp_buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsr<T>(handle,
                                                           direction,
                                                           Mb,
                                                           Nb,
                                                           hnnzb_A,
                                                           descr_A,
                                                           dbsr_val_A,
                                                           dbsr_row_ptr_A,
                                                           dbsr_col_ind_A,
                                                           row_block_dim_A,
                                                           col_block_dim_A,
                                                           descr_C,
                                                           dbsr_val_C,
                                                           dbsr_row_ptr_C,
                                                           dbsr_col_ind_C,
                                                           row_block_dim_C,
                                                           col_block_dim_C,
                                                           dtemp_buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = gebsr2gebsr_gbyte_count<T>(Mb,
                                                      Mb_C,
                                                      row_block_dim_A,
                                                      col_block_dim_A,
                                                      row_block_dim_C,
                                                      col_block_dim_C,
                                                      hnnzb_A,
                                                      hnnzb_C[0])
                           / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "Mb"
                  << std::setw(12) << "Nb" << std::setw(12) << "rblockdimA" << std::setw(12)
                  << "cblockdimA" << std::setw(12) << "rblockdimC" << std::setw(12) << "cblockdimC"
                  << std::setw(12) << "nnzbC" << std::setw(12) << "GB/s" << std::setw(12) << "msec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << Mb
                  << std::setw(12) << Nb << std::setw(12) << row_block_dim_A << std::setw(12)
                  << col_block_dim_A << std::setw(12) << row_block_dim_C << std::setw(12)
                  << col_block_dim_C << std::setw(12) << hnnzb_C[0] << std::setw(12) << gpu_gbyte
                  << std::setw(12) << gpu_time_used / 1e3 << std::setw(12) << number_hot_calls
                  << std::setw(12) << (arg.unit_check ? "yes" : "no") << std::endl;

        // Free buffer
        CHECK_HIP_ERROR(hipFree(dtemp_buffer));
    }
}

#define INSTANTIATE(TYPE)                                                  \
    template void testing_gebsr2gebsr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gebsr2gebsr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
