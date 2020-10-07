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

#pragma once
#ifndef TESTING_GEBSR2CSR_HPP
#define TESTING_GEBSR2CSR_HPP

#include <rocsparse.hpp>

#include "gbyte.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_init.hpp"
#include "rocsparse_math.hpp"
#include "rocsparse_random.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_vector.hpp"
#include "utility.hpp"

//
// Transform a csr matrix in a general bsr matrix.
// It fills the values such as the conversion to the csr matrix
// will give to 1,2,3,4,5,6,7,8,9, etc...
//
template <typename T>
inline void rocsparse_init_gebsr_matrix_from_csr(std::vector<rocsparse_int>& bsr_row_ptr,
                                                 std::vector<rocsparse_int>& bsr_col_ind,
                                                 std::vector<T>&             bsr_val,
                                                 rocsparse_direction         direction,
                                                 rocsparse_int&              Mb,
                                                 rocsparse_int&              Nb,
                                                 rocsparse_int               row_block_dim,
                                                 rocsparse_int               col_block_dim,
                                                 rocsparse_int&              K,
                                                 rocsparse_int               dim_x,
                                                 rocsparse_int               dim_y,
                                                 rocsparse_int               dim_z,
                                                 rocsparse_int&              nnzb,
                                                 rocsparse_index_base        bsr_base,
                                                 rocsparse_matrix_init       matrix,
                                                 const char*                 filename,
                                                 bool                        toint     = false,
                                                 bool                        full_rank = false)
{

    // Matrix handle and descriptors used for conversion
    rocsparse_local_handle handle;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_descr bsr_descr;

    //  rocsparse_set_mat_index_base(csr_descr, csr_base);
    rocsparse_set_mat_index_base(bsr_descr, bsr_base);

    // Uncompressed CSR matrix on host
    host_vector<T> hcsr_val_A;

    // Generate uncompressed CSR matrix on host (or read from file)

    rocsparse_init_csr_matrix(bsr_row_ptr,
                              bsr_col_ind,
                              hcsr_val_A,
                              Mb,
                              Nb,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnzb,
                              bsr_base,
                              matrix,
                              filename,
                              false,
                              full_rank);

    bsr_val.resize(row_block_dim * col_block_dim * nnzb);
    rocsparse_int idx = 0;
    switch(direction)
    {
    case rocsparse_direction_column:
    {
        for(rocsparse_int i = 0; i < Mb; ++i)
        {
            for(rocsparse_int r = 0; r < row_block_dim; ++r)
            {
                for(rocsparse_int k = bsr_row_ptr[i] - bsr_base; k < bsr_row_ptr[i + 1] - bsr_base;
                    ++k)
                {
                    rocsparse_int j = bsr_col_ind[k] - bsr_base;
                    for(rocsparse_int c = 0; c < col_block_dim; ++c)
                    {
                        bsr_val[k * row_block_dim * col_block_dim + c * row_block_dim + r]
                            = static_cast<T>(++idx);
                    }
                }
            }
        }
        break;
    }

    case rocsparse_direction_row:
    {
        for(rocsparse_int i = 0; i < Mb; ++i)
        {
            for(rocsparse_int r = 0; r < row_block_dim; ++r)
            {
                for(rocsparse_int k = bsr_row_ptr[i] - bsr_base; k < bsr_row_ptr[i + 1] - bsr_base;
                    ++k)
                {
                    rocsparse_int j = bsr_col_ind[k] - bsr_base;
                    for(rocsparse_int c = 0; c < col_block_dim; ++c)
                    {
                        bsr_val[k * row_block_dim * col_block_dim + r * col_block_dim + c]
                            = static_cast<T>(++idx);
                    }
                }
            }
        }
        break;
    }
    }
}

template <typename T>
void testing_gebsr2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dbsr_col_ind(safe_size);
    device_vector<T>             dbsr_val(safe_size);
    device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind(safe_size);
    device_vector<T>             dcsr_val(safe_size);

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_local_mat_descr bsr_descr;
    rocsparse_local_mat_descr csr_descr;

    rocsparse_set_mat_index_base(bsr_descr, rocsparse_index_base_zero);
    rocsparse_set_mat_index_base(csr_descr, rocsparse_index_base_zero);

    // Test invalid handle
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(nullptr,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_handle);

    // Test invalid descriptors
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   nullptr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   nullptr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_pointer);

    // Test invalid pointers
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   nullptr,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   nullptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   nullptr,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   nullptr,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   nullptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   nullptr),
                            rocsparse_status_invalid_pointer);

    // Test invalid direction
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   (rocsparse_direction)2,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_value);

    // Test invalid Mb
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   -1,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_size);

    // Test invalid Nb
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   -1,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_size);

    // Test invalid row block dimension
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   0,
                                                   safe_size,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_size);

    // Test invalid col block dimension
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2csr<T>(handle,
                                                   rocsparse_direction_row,
                                                   safe_size,
                                                   safe_size,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   safe_size,
                                                   0,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind),
                            rocsparse_status_invalid_size);
}

template <typename T>
void testing_gebsr2csr(const Arguments& arg)
{
    rocsparse_int         M             = arg.M;
    rocsparse_int         N             = arg.N;
    rocsparse_int         K             = arg.K;
    rocsparse_int         dim_x         = arg.dimx;
    rocsparse_int         dim_y         = arg.dimy;
    rocsparse_int         dim_z         = arg.dimz;
    rocsparse_index_base  bsr_base      = arg.baseA;
    rocsparse_index_base  csr_base      = arg.baseB;
    rocsparse_matrix_init mat           = arg.matrix;
    rocsparse_direction   direction     = arg.direction;
    rocsparse_int         row_block_dim = arg.row_block_dim;
    rocsparse_int         col_block_dim = arg.col_block_dim;
    bool                  full_rank     = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    rocsparse_int Mb = M;
    rocsparse_int Nb = N;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr bsr_descr;
    rocsparse_local_mat_descr csr_descr;

    rocsparse_set_mat_index_base(bsr_descr, bsr_base);
    rocsparse_set_mat_index_base(csr_descr, csr_base);

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

        if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dcsr_row_ptr || !dcsr_col_ind
           || !dcsr_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

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

    rocsparse_seedrand();
    rocsparse_int nnzb = 0;
    rocsparse_init_gebsr_matrix_from_csr(hbsr_row_ptr,
                                         hbsr_col_ind,
                                         hbsr_val,
                                         direction,
                                         Mb,
                                         Nb,
                                         row_block_dim,
                                         col_block_dim,
                                         K,
                                         dim_x,
                                         dim_y,
                                         dim_z,
                                         nnzb,
                                         bsr_base,
                                         mat,
                                         filename.c_str(),
                                         false,
                                         full_rank);

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

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

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
            unit_check_general<T>(1, 1, 1, &hcsr_val_ref[i], &ref);
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

        unit_check_general<rocsparse_int>(1, hcsr_ptr.size(), 1, hcsr_ptr, hcsr_ptr_ref);
        unit_check_general<rocsparse_int>(1, hcsr_ind.size(), 1, hcsr_ind, hcsr_ind_ref);
        unit_check_general<T>(1, hcsr_val.size(), 1, hcsr_val, hcsr_val_ref);
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

        double gpu_gbyte = gebsr2csr_gbyte_count<T>(Mb, row_block_dim, col_block_dim, nnzb)
                           / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "Mb"
                  << std::setw(12) << "Nb" << std::setw(12) << "row_blockdim" << std::setw(12)
                  << "col_blockdim" << std::setw(12) << "nnzb" << std::setw(12) << "GB/s"
                  << std::setw(12) << "msec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << Mb
                  << std::setw(12) << Nb << std::setw(12) << row_block_dim << std::setw(12)
                  << col_block_dim << std::setw(12) << nnzb << std::setw(12) << gpu_gbyte
                  << std::setw(12) << gpu_time_used / 1e3 << std::setw(12) << number_hot_calls
                  << std::setw(12) << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#endif // TESTING_GEBSR2CSR_HPP
