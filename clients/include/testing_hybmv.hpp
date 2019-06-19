/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#ifndef TESTING_HYBMV_HPP
#define TESTING_HYBMV_HPP

#include "rocsparse.hpp"
#include "rocsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <string>

using namespace rocsparse;
using namespace rocsparse_test;

#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

struct testhyb
{
    rocsparse_int           m;
    rocsparse_int           n;
    rocsparse_hyb_partition partition;
    rocsparse_int           ell_nnz;
    rocsparse_int           ell_width;
    rocsparse_int*          ell_col_ind;
    void*                   ell_val;
    rocsparse_int           coo_nnz;
    rocsparse_int*          coo_row_ind;
    rocsparse_int*          coo_col_ind;
    void*                   coo_val;
};

template <typename T>
void testing_hybmv_bad_arg(void)
{
    rocsparse_int       safe_size = 100;
    T                   alpha     = 0.6;
    T                   beta      = 0.2;
    rocsparse_operation transA    = rocsparse_operation_none;
    rocsparse_status    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    rocsparse_mat_descr           descr = unique_ptr_descr->descr;

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    rocsparse_hyb_mat           hyb = unique_ptr_hyb->hyb;

    auto dx_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dy_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T* dx = (T*)dx_managed.get();
    T* dy = (T*)dy_managed.get();

    if(!dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing for(nullptr == dx)
    {
        T* dx_null = nullptr;

        status = rocsparse_hybmv(handle, transA, &alpha, descr, hyb, dx_null, &beta, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: dx is nullptr");
    }
    // testing for(nullptr == dy)
    {
        T* dy_null = nullptr;

        status = rocsparse_hybmv(handle, transA, &alpha, descr, hyb, dx, &beta, dy_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: dy is nullptr");
    }
    // testing for(nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = rocsparse_hybmv(handle, transA, d_alpha_null, descr, hyb, dx, &beta, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == d_beta)
    {
        T* d_beta_null = nullptr;

        status = rocsparse_hybmv(handle, transA, &alpha, descr, hyb, dx, d_beta_null, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: beta is nullptr");
    }
    // testing for(nullptr == hyb)
    {
        rocsparse_hyb_mat hyb_null = nullptr;

        status = rocsparse_hybmv(handle, transA, &alpha, descr, hyb_null, dx, &beta, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == descr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_hybmv(handle, transA, &alpha, descr_null, hyb, dx, &beta, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_hybmv(handle_null, transA, &alpha, descr, hyb, dx, &beta, dy);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
rocsparse_status testing_hybmv(Arguments argus)
{
    rocsparse_int           safe_size      = 100;
    rocsparse_int           m              = argus.M;
    rocsparse_int           n              = argus.N;
    T                       h_alpha        = argus.alpha;
    T                       h_beta         = argus.beta;
    rocsparse_operation     transA         = argus.transA;
    rocsparse_index_base    idx_base       = argus.idx_base;
    rocsparse_hyb_partition part           = argus.part;
    rocsparse_int           user_ell_width = argus.ell_width;
    std::string             binfile        = "";
    std::string             filename       = "";
    rocsparse_status        status;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(m == -99 && n == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        m = n = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    rocsparse_handle               handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    rocsparse_mat_descr           descr = test_descr->descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, idx_base));

    std::unique_ptr<hyb_struct> test_hyb(new hyb_struct);
    rocsparse_hyb_mat           hyb = test_hyb->hyb;

    // Determine number of non-zero elements
    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    rocsparse_int nnz = m * scale * n;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto dptr_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dcol_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dx_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dy_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        rocsparse_int* dptr = (rocsparse_int*)dptr_managed.get();
        rocsparse_int* dcol = (rocsparse_int*)dcol_managed.get();
        T*             dval = (T*)dval_managed.get();
        T*             dx   = (T*)dx_managed.get();
        T*             dy   = (T*)dy_managed.get();

        if(!dval || !dptr || !dcol || !dx || !dy)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dptr || !dcol || !dval || !dx || !dy");
            return rocsparse_status_memory_error;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        status
            = rocsparse_csr2hyb(handle, m, n, descr, dval, dptr, dcol, hyb, user_ell_width, part);

        if(m < 0 || n < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
        }

        // hybmv should be able to deal with m <= 0 || n <= 0 || nnz <= 0 even if csr2hyb fails
        // because hyb structures is allocated with n = m = 0 - so nothing should happen
        status = rocsparse_hybmv(handle, transA, &h_alpha, descr, hyb, dx, &h_beta, dy);
        verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");

        return rocsparse_status_success;
    }

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr;
    std::vector<rocsparse_int> hcoo_row_ind;
    std::vector<rocsparse_int> hcol_ind;
    std::vector<T>             hval;

    // Initial Data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcol_ind, hval, idx_base) != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcol_ind, hval, idx_base);
        nnz   = hcsr_row_ptr[m];
    }
    else
    {
        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(), m, n, nnz, hcoo_row_ind, hcol_ind, hval, idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcol_ind, hval, idx_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr.resize(m + 1, 0);
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr[hcoo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr[0] = idx_base;
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
        }
    }

    std::vector<T> hx(n);
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);
    std::vector<T> hy_gold(m);

    rocsparse_init<T>(hx, 1, n);
    rocsparse_init<T>(hy_1, 1, m);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};
    auto dcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dval_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_managed      = rocsparse_unique_ptr{device_malloc(sizeof(T) * n), device_free};
    auto dy_1_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    rocsparse_int* dptr    = (rocsparse_int*)dptr_managed.get();
    rocsparse_int* dcol    = (rocsparse_int*)dcol_managed.get();
    T*             dval    = (T*)dval_managed.get();
    T*             dx      = (T*)dx_managed.get();
    T*             dy_1    = (T*)dy_1_managed.get();
    T*             dy_2    = (T*)dy_2_managed.get();
    T*             d_alpha = (T*)d_alpha_managed.get();
    T*             d_beta  = (T*)d_beta_managed.get();

    if(!dval || !dptr || !dcol || !dx || !dy_1 || !dy_2 || !d_alpha || !d_beta)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dval || !dptr || !dcol || !dx || "
                                        "!dy_1 || !dy_2 || !d_alpha || !d_beta");
        return rocsparse_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcol, hcol_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // ELL width limit
    rocsparse_int width_limit = (2 * nnz - 1) / m + 1;

    // Limit ELL user width
    if(part == rocsparse_hyb_partition_user)
    {
        user_ell_width = user_ell_width * nnz / m;
        user_ell_width = std::min(width_limit, user_ell_width);
    }

    // Convert CSR to HYB
    status = rocsparse_csr2hyb(handle, m, n, descr, dval, dptr, dcol, hyb, user_ell_width, part);

    if(part == rocsparse_hyb_partition_max)
    {
        // Compute max ELL width
        rocsparse_int ell_max_width = 0;
        for(rocsparse_int i = 0; i < m; ++i)
        {
            ell_max_width = std::max(hcsr_row_ptr[i + 1] - hcsr_row_ptr[i], ell_max_width);
        }

        if(ell_max_width > width_limit)
        {
            verify_rocsparse_status_invalid_value(status, "ell_max_width > width_limit");
            return rocsparse_status_success;
        }
    }

    if(argus.unit_check)
    {
        // Copy HYB structure to CPU
        testhyb* dhyb = (testhyb*)hyb;

        rocsparse_int ell_nnz = dhyb->ell_nnz;
        rocsparse_int coo_nnz = dhyb->coo_nnz;

        std::vector<rocsparse_int> hell_col(ell_nnz);
        std::vector<T>             hell_val(ell_nnz);
        std::vector<rocsparse_int> hcoo_row(coo_nnz);
        std::vector<rocsparse_int> hcoo_col(coo_nnz);
        std::vector<T>             hcoo_val(coo_nnz);

        if(ell_nnz > 0)
        {
            CHECK_HIP_ERROR(hipMemcpy(hell_col.data(),
                                      dhyb->ell_col_ind,
                                      sizeof(rocsparse_int) * ell_nnz,
                                      hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(
                hell_val.data(), dhyb->ell_val, sizeof(T) * ell_nnz, hipMemcpyDeviceToHost));
        }

        if(coo_nnz > 0)
        {
            CHECK_HIP_ERROR(hipMemcpy(hcoo_row.data(),
                                      dhyb->coo_row_ind,
                                      sizeof(rocsparse_int) * coo_nnz,
                                      hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hcoo_col.data(),
                                      dhyb->coo_col_ind,
                                      sizeof(rocsparse_int) * coo_nnz,
                                      hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(
                hcoo_val.data(), dhyb->coo_val, sizeof(T) * coo_nnz, hipMemcpyDeviceToHost));
        }

        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_hybmv(handle, transA, &h_alpha, descr, hyb, dx, &h_beta, dy_1));

        // ROCSPARSE pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_hybmv(handle, transA, d_alpha, descr, hyb, dx, d_beta, dy_2));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

        // CPU
        double cpu_time_used = get_time_us();

        // ELL part
        if(ell_nnz > 0)
        {
            for(rocsparse_int i = 0; i < m; ++i)
            {
                T sum = static_cast<T>(0);
                for(rocsparse_int p = 0; p < dhyb->ell_width; ++p)
                {
                    rocsparse_int idx = ELL_IND(i, p, m, dhyb->ell_width);
                    rocsparse_int col = hell_col[idx] - idx_base;

                    if(col >= 0 && col < n)
                    {
                        sum += hell_val[idx] * hx[col];
                    }
                    else
                    {
                        break;
                    }
                }

                if(h_beta != static_cast<T>(0))
                {
                    hy_gold[i] = h_beta * hy_gold[i] + h_alpha * sum;
                }
                else
                {
                    hy_gold[i] = h_alpha * sum;
                }
            }
        }

        // COO part
        if(coo_nnz > 0)
        {
            T coo_beta = (ell_nnz > 0) ? static_cast<T>(1) : h_beta;

            for(rocsparse_int i = 0; i < m; ++i)
            {
                hy_gold[i] *= coo_beta;
            }

            for(rocsparse_int i = 0; i < coo_nnz; ++i)
            {
                rocsparse_int row = hcoo_row[i] - idx_base;
                rocsparse_int col = hcoo_col[i] - idx_base;

                hy_gold[row] += h_alpha * hcoo_val[i] * hx[col];
            }
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
        unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocsparse_hybmv(handle, transA, &h_alpha, descr, hyb, dx, &h_beta, dy_1);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocsparse_hybmv(handle, transA, &h_alpha, descr, hyb, dx, &h_beta, dy_1);
        }

        testhyb* dhyb = (testhyb*)hyb;

        // Convert to miliseconds per call
        gpu_time_used     = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);
        size_t flops      = (h_alpha != 1.0) ? 3.0 * nnz : 2.0 * nnz;
        flops             = (h_beta != 0.0) ? flops + m : flops;
        double gpu_gflops = flops / gpu_time_used / 1e6;
        size_t ell_mem    = dhyb->ell_nnz * (sizeof(rocsparse_int) + sizeof(T));
        size_t coo_mem    = dhyb->coo_nnz * (sizeof(rocsparse_int) * 2 + sizeof(T));
        size_t memtrans   = (m + n) * sizeof(T) + ell_mem + coo_mem;
        memtrans          = (h_beta != 0.0) ? memtrans + m : memtrans;
        double bandwidth  = memtrans / gpu_time_used / 1e6;

        printf("m\t\tn\t\tnnz\t\talpha\tbeta\tGFlops\tGB/s\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
               m,
               n,
               dhyb->ell_nnz + dhyb->coo_nnz,
               h_alpha,
               h_beta,
               gpu_gflops,
               bandwidth,
               gpu_time_used);
    }

    return rocsparse_status_success;
}

#endif // TESTING_HYBMV_HPP
