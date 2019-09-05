/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef TESTING_CSRGEMM_HPP
#define TESTING_CSRGEMM_HPP

#include "rocsparse.hpp"
#include "rocsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_csrgemm_bad_arg(void)
{
    rocsparse_int       M         = 100;
    rocsparse_int       N         = 100;
    rocsparse_int       K         = 100;
    rocsparse_int       nnz_A     = 100;
    rocsparse_int       nnz_B     = 100;
    rocsparse_operation trans_A   = rocsparse_operation_none;
    rocsparse_operation trans_B   = rocsparse_operation_none;
    rocsparse_int       safe_size = 100;
    rocsparse_status    status;

    T alpha = 1.0;

    size_t        size;
    rocsparse_int nnz_C;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    rocsparse_mat_descr           descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_B(new descr_struct);
    rocsparse_mat_descr           descr_B = unique_ptr_descr_B->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    rocsparse_mat_descr           descr_C = unique_ptr_descr_C->descr;

    std::unique_ptr<mat_info_struct> unique_ptr_info(new mat_info_struct);
    rocsparse_mat_info               info = unique_ptr_info->info;

    auto dAptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dAcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dAval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dBptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dBcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dBval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dCptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dCcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dCval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    rocsparse_int* dAptr   = (rocsparse_int*)dAptr_managed.get();
    rocsparse_int* dAcol   = (rocsparse_int*)dAcol_managed.get();
    T*             dAval   = (T*)dAval_managed.get();
    rocsparse_int* dBptr   = (rocsparse_int*)dBptr_managed.get();
    rocsparse_int* dBcol   = (rocsparse_int*)dBcol_managed.get();
    T*             dBval   = (T*)dBval_managed.get();
    rocsparse_int* dCptr   = (rocsparse_int*)dCptr_managed.get();
    rocsparse_int* dCcol   = (rocsparse_int*)dCcol_managed.get();
    T*             dCval   = (T*)dCval_managed.get();
    void*          dbuffer = (void*)dbuffer_managed.get();

    if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCval || !dCptr || !dCcol
       || !dbuffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // We need to test C = alpha * A * B ; other scenarios are not yet implemented

    // Scenario: alpha != nullptr and beta == nullptr

    // testing rocsparse_csrgemm_buffer_size

    // testing for(nullptr == handle)
    {
        status = rocsparse_csrgemm_buffer_size((rocsparse_handle) nullptr,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == alpha and nullptr == beta)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               (T*)nullptr,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: alpha and beta are nullptr");
    }
    // testing for(nullptr == descr_A)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               (rocsparse_mat_descr) nullptr,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == dAptr)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               descr_A,
                                               nnz_A,
                                               (rocsparse_int*)nullptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               (rocsparse_int*)nullptr,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               (rocsparse_mat_descr) nullptr,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               (rocsparse_int*)nullptr,
                                               dBcol,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               (rocsparse_int*)nullptr,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == info)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_mat_info) nullptr,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == size)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               (T*)nullptr,
                                               (rocsparse_mat_descr) nullptr,
                                               0,
                                               (rocsparse_int*)nullptr,
                                               (rocsparse_int*)nullptr,
                                               info,
                                               (size_t*)nullptr);
        verify_rocsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }

    // We need one successful call to create the info structure
    status = rocsparse_csrgemm_buffer_size(handle,
                                           trans_A,
                                           trans_B,
                                           M,
                                           N,
                                           K,
                                           &alpha,
                                           descr_A,
                                           nnz_A,
                                           dAptr,
                                           dAcol,
                                           descr_B,
                                           nnz_B,
                                           dBptr,
                                           dBcol,
                                           (T*)nullptr,
                                           (rocsparse_mat_descr) nullptr,
                                           0,
                                           (rocsparse_int*)nullptr,
                                           (rocsparse_int*)nullptr,
                                           info,
                                           &size);
    verify_rocsparse_status_success(status, "Success");

    // testing rocsparse_csrgemm_nnz

    // testing for(nullptr == handle)
    {
        status = rocsparse_csrgemm_nnz((rocsparse_handle) nullptr,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == descr_A)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       (rocsparse_mat_descr) nullptr,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == dAptr)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       (rocsparse_int*)nullptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       (rocsparse_int*)nullptr,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       (rocsparse_mat_descr) nullptr,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       (rocsparse_int*)nullptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_mat_descr) nullptr,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       (rocsparse_int*)nullptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == nnz_C)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       (rocsparse_int*)nullptr,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: nnz_C is nullptr");
    }
    // testing for(nullptr == info)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       (rocsparse_mat_info) nullptr,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       (rocsparse_mat_descr) nullptr,
                                       0,
                                       (rocsparse_int*)nullptr,
                                       (rocsparse_int*)nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       nullptr);
        verify_rocsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }

    // testing rocsparse_csrgemm

    // testing for(nullptr == handle)
    {
        status = rocsparse_csrgemm((rocsparse_handle) nullptr,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == alpha && nullptr == beta)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   (T*)nullptr,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: alpha and beta are nullptr");
    }
    // testing for(nullptr == descr_A)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   (rocsparse_mat_descr) nullptr,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == dAval)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   (T*)nullptr,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAval is nullptr");
    }
    // testing for(nullptr == dAptr)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   (rocsparse_int*)nullptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   (rocsparse_int*)nullptr,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   (rocsparse_mat_descr) nullptr,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == dBval)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   (T*)nullptr,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBval is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   (rocsparse_int*)nullptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   (rocsparse_int*)nullptr,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == dCval)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   (T*)nullptr,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCval is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   (rocsparse_int*)nullptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == dCcol)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   (rocsparse_int*)nullptr,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCcol is nullptr");
    }
    // testing for(nullptr == info)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   (rocsparse_mat_info) nullptr,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   (T*)nullptr,
                                   (rocsparse_mat_descr) nullptr,
                                   0,
                                   (T*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   (rocsparse_int*)nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   nullptr);
        verify_rocsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
}

template <typename T>
static rocsparse_int csrgemm_nnz(rocsparse_int        m,
                                 rocsparse_int        n,
                                 rocsparse_int        k,
                                 const T*             alpha,
                                 const rocsparse_int* csr_row_ptr_A,
                                 const rocsparse_int* csr_col_ind_A,
                                 const rocsparse_int* csr_row_ptr_B,
                                 const rocsparse_int* csr_col_ind_B,
                                 const T*             beta,
                                 const rocsparse_int* csr_row_ptr_D,
                                 const rocsparse_int* csr_col_ind_D,
                                 rocsparse_int*       csr_row_ptr_C,
                                 rocsparse_index_base idx_base_A,
                                 rocsparse_index_base idx_base_B,
                                 rocsparse_index_base idx_base_C,
                                 rocsparse_index_base idx_base_D)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<rocsparse_int> nnz(n, -1);

#ifdef _OPENMP
        rocsparse_int nthreads = omp_get_num_threads();
        rocsparse_int tid      = omp_get_thread_num();
#else
        rocsparse_int nthreads = 1;
        rocsparse_int tid      = 0;
#endif

        rocsparse_int rows_per_thread = (m + nthreads - 1) / nthreads;
        rocsparse_int chunk_begin     = rows_per_thread * tid;
        rocsparse_int chunk_end       = std::min(chunk_begin + rows_per_thread, m);

        // Index base
        csr_row_ptr_C[0] = idx_base_C;

        // Loop over rows of A
        for(rocsparse_int i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            if(alpha)
            {
                rocsparse_int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
                rocsparse_int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

                // Loop over columns of A
                for(rocsparse_int j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    rocsparse_int col_A = csr_col_ind_A[j] - idx_base_A;

                    rocsparse_int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                    rocsparse_int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                    // Loop over columns of B in row col_A
                    for(rocsparse_int k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        rocsparse_int col_B = csr_col_ind_B[k] - idx_base_B;

                        // Check if a new nnz is generated
                        if(nnz[col_B] != i)
                        {
                            nnz[col_B] = i;
                            ++csr_row_ptr_C[i + 1];
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                rocsparse_int row_begin_D = csr_row_ptr_D[i] - idx_base_D;
                rocsparse_int row_end_D   = csr_row_ptr_D[i + 1] - idx_base_D;

                // Loop over columns of D
                for(rocsparse_int j = row_begin_D; j < row_end_D; ++j)
                {
                    rocsparse_int col_D = csr_col_ind_D[j] - idx_base_D;

                    // Check if a new nnz is generated
                    if(nnz[col_D] != i)
                    {
                        nnz[col_D] = i;
                        ++csr_row_ptr_C[i + 1];
                    }
                }
            }
        }
    }

    // Scan to obtain row offsets
    for(rocsparse_int i = 0; i < m; ++i)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    return csr_row_ptr_C[m] - idx_base_C;
}

template <typename T>
static void csrgemm(rocsparse_int        m,
                    rocsparse_int        n,
                    rocsparse_int        k,
                    const T*             alpha,
                    const rocsparse_int* csr_row_ptr_A,
                    const rocsparse_int* csr_col_ind_A,
                    const T*             csr_val_A,
                    const rocsparse_int* csr_row_ptr_B,
                    const rocsparse_int* csr_col_ind_B,
                    const T*             csr_val_B,
                    const T*             beta,
                    const rocsparse_int* csr_row_ptr_D,
                    const rocsparse_int* csr_col_ind_D,
                    const T*             csr_val_D,
                    const rocsparse_int* csr_row_ptr_C,
                    rocsparse_int*       csr_col_ind_C,
                    T*                   csr_val_C,
                    rocsparse_index_base idx_base_A,
                    rocsparse_index_base idx_base_B,
                    rocsparse_index_base idx_base_C,
                    rocsparse_index_base idx_base_D)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<rocsparse_int> nnz(n, -1);

#ifdef _OPENMP
        rocsparse_int nthreads = omp_get_num_threads();
        rocsparse_int tid      = omp_get_thread_num();
#else
        rocsparse_int nthreads = 1;
        rocsparse_int tid      = 0;
#endif

        rocsparse_int rows_per_thread = (m + nthreads - 1) / nthreads;
        rocsparse_int chunk_begin     = rows_per_thread * tid;
        rocsparse_int chunk_end       = std::min(chunk_begin + rows_per_thread, m);

        // Loop over rows of A
        for(rocsparse_int i = chunk_begin; i < chunk_end; ++i)
        {
            rocsparse_int row_begin_C = csr_row_ptr_C[i] - idx_base_C;
            rocsparse_int row_end_C   = row_begin_C;

            if(alpha)
            {
                rocsparse_int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
                rocsparse_int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

                // Loop over columns of A
                for(rocsparse_int j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    rocsparse_int col_A = csr_col_ind_A[j] - idx_base_A;
                    // Current value of A
                    T val_A = *alpha * csr_val_A[j];

                    rocsparse_int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                    rocsparse_int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                    // Loop over columns of B in row col_A
                    for(rocsparse_int k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        rocsparse_int col_B = csr_col_ind_B[k] - idx_base_B;
                        // Current value of B
                        T val_B = csr_val_B[k];

                        // Check if a new nnz is generated or if the product is appended
                        if(nnz[col_B] < row_begin_C)
                        {
                            nnz[col_B]               = row_end_C;
                            csr_col_ind_C[row_end_C] = col_B + idx_base_C;
                            csr_val_C[row_end_C]     = val_A * val_B;
                            ++row_end_C;
                        }
                        else
                        {
                            csr_val_C[nnz[col_B]] += val_A * val_B;
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                rocsparse_int row_begin_D = csr_row_ptr_D[i] - idx_base_D;
                rocsparse_int row_end_D   = csr_row_ptr_D[i + 1] - idx_base_D;

                // Loop over columns of D
                for(rocsparse_int j = row_begin_D; j < row_end_D; ++j)
                {
                    // Current column of D
                    rocsparse_int col_D = csr_col_ind_D[j] - idx_base_D;
                    // Current value of D
                    T val_D = *beta * csr_val_D[j];

                    // Check if a new nnz is generated or if the value is added
                    if(nnz[col_D] < row_begin_C)
                    {
                        nnz[col_D] = row_end_C;

                        csr_col_ind_C[row_end_C] = col_D + idx_base_D;
                        csr_val_C[row_end_C]     = val_D;
                        ++row_end_C;
                    }
                    else
                    {
                        csr_val_C[nnz[col_D]] += val_D;
                    }
                }
            }
        }
    }

    rocsparse_int nnz = csr_row_ptr_C[m] - idx_base_C;

    std::vector<rocsparse_int> col(nnz);
    std::vector<T>             val(nnz);

    memcpy(col.data(), csr_col_ind_C, sizeof(rocsparse_int) * nnz);
    memcpy(val.data(), csr_val_C, sizeof(T) * nnz);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(rocsparse_int i = 0; i < m; ++i)
    {
        rocsparse_int row_begin = csr_row_ptr_C[i] - idx_base_C;
        rocsparse_int row_end   = csr_row_ptr_C[i + 1] - idx_base_C;
        rocsparse_int row_nnz   = row_end - row_begin;

        std::vector<rocsparse_int> perm(row_nnz);
        for(rocsparse_int j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        rocsparse_int* col_entry = &col[row_begin];
        T*             val_entry = &val[row_begin];

        std::sort(perm.begin(), perm.end(), [&](const rocsparse_int& a, const rocsparse_int& b) {
            return col_entry[a] <= col_entry[b];
        });

        for(rocsparse_int j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

template <typename T>
static size_t csrgemm_flops(rocsparse_int        m,
                            const T*             alpha,
                            const rocsparse_int* csr_row_ptr_A,
                            const rocsparse_int* csr_col_ind_A,
                            const rocsparse_int* csr_row_ptr_B,
                            const T*             beta,
                            const rocsparse_int* csr_row_ptr_D,
                            rocsparse_index_base idx_base_A)
{
    // Flop counter
    size_t flops = 0;

    // Loop over rows of A
    for(rocsparse_int i = 0; i < m; ++i)
    {
        if(alpha)
        {
            rocsparse_int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
            rocsparse_int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

            // Loop over columns of A
            for(rocsparse_int j = row_begin_A; j < row_end_A; ++j)
            {
                // Current column of A
                rocsparse_int col_A = csr_col_ind_A[j] - idx_base_A;

                // Count flops generated by alpha * A * B
                flops += 2 * (csr_row_ptr_B[col_A + 1] - csr_row_ptr_B[col_A]) + 1;
            }
        }

        if(beta)
        {
            // Count flops generated by beta * D
            flops += (csr_row_ptr_D[i + 1] - csr_row_ptr_D[i]);
        }
    }

    return flops;
}

template <typename T>
rocsparse_status testing_csrgemm(Arguments argus)
{
    rocsparse_int        safe_size  = 100;
    rocsparse_int        M          = argus.M;
    rocsparse_int        N          = argus.N;
    rocsparse_int        K          = argus.K;
    rocsparse_operation  trans_A    = argus.transA;
    rocsparse_operation  trans_B    = argus.transB;
    rocsparse_index_base idx_base_A = argus.idx_base;
    rocsparse_index_base idx_base_B = argus.idx_base2;
    rocsparse_index_base idx_base_C = argus.idx_base3;
    rocsparse_index_base idx_base_D = argus.idx_base4;
    T                    alpha      = argus.alpha;
    std::string          binfile    = "";
    std::string          filename   = "";
    std::string          rocalution = "";

    rocsparse_status status;
    size_t           size;

    T* h_alpha = &alpha;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(M == -99 && N == -99 && K == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        M = N = K = safe_size;
    }

    if(argus.timing == 1)
    {
        if(argus.rocalution != "")
        {
            rocalution = argus.rocalution;
        }
        else if(argus.filename != "")
        {
            filename = argus.filename;
        }
    }

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    rocsparse_handle               handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr_A(new descr_struct);
    rocsparse_mat_descr           descr_A = test_descr_A->descr;

    std::unique_ptr<descr_struct> test_descr_B(new descr_struct);
    rocsparse_mat_descr           descr_B = test_descr_B->descr;

    std::unique_ptr<descr_struct> test_descr_C(new descr_struct);
    rocsparse_mat_descr           descr_C = test_descr_C->descr;

    std::unique_ptr<mat_info_struct> test_info(new mat_info_struct);
    rocsparse_mat_info               info = test_info->info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_A, idx_base_A));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_B, idx_base_B));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_C, idx_base_C));

    // Determine number of non-zero elements
    double scale = 0.02;
    if(M > 1000 || K > 1000)
    {
        scale = 2.0 / std::max(M, K);
    }
    rocsparse_int nnz_A = M * scale * K;

    scale = 0.02;
    if(K > 1000 || N > 1000)
    {
        scale = 2.0 / std::max(K, N);
    }
    rocsparse_int nnz_B = K * scale * N;

    // Argument sanity check before allocating invalid memory
    // alpha != nullptr and beta == nullptr
    {
        auto dAptr_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dAcol_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dAval_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dBptr_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dBcol_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dBval_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dCptr_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dCcol_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dCval_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto buffer_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        rocsparse_int* dAptr  = (rocsparse_int*)dAptr_managed.get();
        rocsparse_int* dAcol  = (rocsparse_int*)dAcol_managed.get();
        T*             dAval  = (T*)dAval_managed.get();
        rocsparse_int* dBptr  = (rocsparse_int*)dBptr_managed.get();
        rocsparse_int* dBcol  = (rocsparse_int*)dBcol_managed.get();
        T*             dBval  = (T*)dBval_managed.get();
        rocsparse_int* dCptr  = (rocsparse_int*)dCptr_managed.get();
        rocsparse_int* dCcol  = (rocsparse_int*)dCcol_managed.get();
        T*             dCval  = (T*)dCval_managed.get();
        void*          buffer = (void*)buffer_managed.get();

        if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCval || !dCptr || !dCcol
           || !buffer)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dAptr || !dAcol || !dAval || "
                                            "!dBptr || !dBcol || !dBval || "
                                            "!dCptr || !dCcol || !dCval || "
                                            "!buffer");
            return rocsparse_status_memory_error;
        }

        if(M <= 0 || N <= 0 || K <= 0 || nnz_A <= 0 || nnz_B <= 0)
        {
            rocsparse_int nnz_C;
            status = rocsparse_csrgemm_buffer_size(handle,
                                                   trans_A,
                                                   trans_B,
                                                   M,
                                                   N,
                                                   K,
                                                   h_alpha,
                                                   descr_A,
                                                   nnz_A,
                                                   dAptr,
                                                   dAcol,
                                                   descr_B,
                                                   nnz_B,
                                                   dBptr,
                                                   dBcol,
                                                   (T*)nullptr,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   nullptr,
                                                   info,
                                                   &size);

            if(M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0)
            {
                verify_rocsparse_status_invalid_size(
                    status, "Error: M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0");
            }
            else
            {
                verify_rocsparse_status_success(
                    status, "M >= 0 && N >= 0 && K >= 0 && nnz_A >= 0 && nnz_B >= 0");
            }

            status = rocsparse_csrgemm_nnz(handle,
                                           trans_A,
                                           trans_B,
                                           M,
                                           N,
                                           K,
                                           descr_A,
                                           nnz_A,
                                           dAptr,
                                           dAcol,
                                           descr_B,
                                           nnz_B,
                                           dBptr,
                                           dBcol,
                                           nullptr,
                                           0,
                                           nullptr,
                                           nullptr,
                                           descr_C,
                                           dCptr,
                                           &nnz_C,
                                           info,
                                           buffer);

            if(M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0)
            {
                verify_rocsparse_status_invalid_size(
                    status, "Error: M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0");
            }
            else
            {
                verify_rocsparse_status_success(
                    status, "M >= 0 && N >= 0 && K >= 0 && nnz_A >= 0 && nnz_B >= 0");
            }

            status = rocsparse_csrgemm(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       h_alpha,
                                       descr_A,
                                       nnz_A,
                                       dAval,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBval,
                                       dBptr,
                                       dBcol,
                                       (T*)nullptr,
                                       nullptr,
                                       0,
                                       (T*)nullptr,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCval,
                                       dCptr,
                                       dCcol,
                                       info,
                                       buffer);

            if(M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0)
            {
                verify_rocsparse_status_invalid_size(
                    status, "Error: M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0");
            }
            else
            {
                verify_rocsparse_status_success(
                    status, "M >= 0 && N >= 0 && K >= 0 && nnz_A >= 0 && nnz_B >= 0");
            }

            return rocsparse_status_success;
        }
    }

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr_A;
    std::vector<rocsparse_int> hcsr_col_ind_A;
    std::vector<T>             hcsr_val_A;
    std::vector<rocsparse_int> hcsr_row_ptr_B;
    std::vector<rocsparse_int> hcsr_col_ind_B;
    std::vector<T>             hcsr_val_B;
    std::vector<rocsparse_int> hcsr_row_ptr_D;
    std::vector<rocsparse_int> hcsr_col_ind_D;
    std::vector<T>             hcsr_val_D;

    // Initial Data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), M, K, nnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base_A)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(rocalution != "")
    {
        if(read_rocalution_matrix(rocalution.c_str(),
                                  M,
                                  K,
                                  nnz_A,
                                  hcsr_row_ptr_A,
                                  hcsr_col_ind_A,
                                  hcsr_val_A,
                                  idx_base_A)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", rocalution.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        M = K = gen_2d_laplacian(
            argus.laplacian, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base_A);
        nnz_A = hcsr_row_ptr_A[M];
    }
    else
    {
        std::vector<rocsparse_int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(),
                               M,
                               K,
                               nnz_A,
                               hcoo_row_ind,
                               hcsr_col_ind_A,
                               hcsr_val_A,
                               idx_base_A)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(M, K, nnz_A, hcoo_row_ind, hcsr_col_ind_A, hcsr_val_A, idx_base_A);
        }

        // Convert COO to CSR
        hcsr_row_ptr_A.resize(M + 1, 0);
        for(rocsparse_int i = 0; i < nnz_A; ++i)
        {
            ++hcsr_row_ptr_A[hcoo_row_ind[i] + 1 - idx_base_A];
        }

        hcsr_row_ptr_A[0] = idx_base_A;
        for(rocsparse_int i = 0; i < M; ++i)
        {
            hcsr_row_ptr_A[i + 1] += hcsr_row_ptr_A[i];
        }

        // TODO samples B matrix instead of squaring
    }

    // B = A^T so that we can compute the square of A
    N     = M;
    nnz_B = nnz_A;

    hcsr_row_ptr_B.resize(K + 1, 0);
    hcsr_col_ind_B.resize(nnz_B);
    hcsr_val_B.resize(nnz_B);

    // B = A^T
    transpose(M,
              K,
              nnz_A,
              hcsr_row_ptr_A.data(),
              hcsr_col_ind_A.data(),
              hcsr_val_A.data(),
              hcsr_row_ptr_B.data(),
              hcsr_col_ind_B.data(),
              hcsr_val_B.data(),
              idx_base_A,
              idx_base_B);

    // Allocate memory on device
    rocsparse_int one        = 1;
    rocsparse_int safe_K     = std::max(K, one);
    rocsparse_int safe_nnz_A = std::max(nnz_A, one);
    rocsparse_int safe_nnz_B = std::max(nnz_B, one);

    auto dAptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (M + 1)), device_free};
    auto dAcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_nnz_A), device_free};
    auto dAval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_nnz_A), device_free};
    auto dBptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (safe_K + 1)), device_free};
    auto dBcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_nnz_B), device_free};
    auto dBval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_nnz_B), device_free};
    auto dCptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (M + 1)), device_free};
    auto dalpha_managed = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    rocsparse_int* dAptr  = (rocsparse_int*)dAptr_managed.get();
    rocsparse_int* dAcol  = (rocsparse_int*)dAcol_managed.get();
    T*             dAval  = (T*)dAval_managed.get();
    rocsparse_int* dBptr  = (rocsparse_int*)dBptr_managed.get();
    rocsparse_int* dBcol  = (rocsparse_int*)dBcol_managed.get();
    T*             dBval  = (T*)dBval_managed.get();
    rocsparse_int* dCptr  = (rocsparse_int*)dCptr_managed.get();
    T*             dalpha = (T*)dalpha_managed.get();

    if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCptr || !dalpha)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dAval || !dAptr || !dAcol || "
                                        "!dBval || !dBptr || !dBcol || "
                                        "!dCptr || !dalpha");
        return rocsparse_status_memory_error;
    }

    CHECK_HIP_ERROR(hipMemcpy(
        dAptr, hcsr_row_ptr_A.data(), sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dAcol, hcsr_col_ind_A.data(), sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dAval, hcsr_val_A.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dBptr, hcsr_row_ptr_B.data(), sizeof(rocsparse_int) * (K + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dBcol, hcsr_col_ind_B.data(), sizeof(rocsparse_int) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dBval, hcsr_val_B.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Obtain csrgemm buffer size
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_buffer_size(handle,
                                                        trans_A,
                                                        trans_B,
                                                        M,
                                                        N,
                                                        K,
                                                        h_alpha,
                                                        descr_A,
                                                        nnz_A,
                                                        dAptr,
                                                        dAcol,
                                                        descr_B,
                                                        nnz_B,
                                                        dBptr,
                                                        dBcol,
                                                        (T*)nullptr,
                                                        nullptr,
                                                        0,
                                                        nullptr,
                                                        nullptr,
                                                        info,
                                                        &size));

    // Buffer size must be greater than 4
    if(size < 4)
    {
        return rocsparse_status_memory_error;
    }

    if(argus.unit_check)
    {
        // Obtain csrgemm buffer size
        size_t size2;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_buffer_size(handle,
                                                            trans_A,
                                                            trans_B,
                                                            M,
                                                            N,
                                                            K,
                                                            dalpha,
                                                            descr_A,
                                                            nnz_A,
                                                            dAptr,
                                                            dAcol,
                                                            descr_B,
                                                            nnz_B,
                                                            dBptr,
                                                            dBcol,
                                                            (T*)nullptr,
                                                            nullptr,
                                                            0,
                                                            nullptr,
                                                            nullptr,
                                                            info,
                                                            &size2));

        // Check if buffer size matches
        unit_check_general(1, 1, 1, &size, &size2);
    }

    // Allocate buffer on the device
    auto dbuffer_managed = rocsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dbuffer)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error, "!dbuffer");
        return rocsparse_status_memory_error;
    }

    // csrgemm nnz

    // rocsparse pointer mode host
    rocsparse_int hnnz_C_1;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                trans_A,
                                                trans_B,
                                                M,
                                                N,
                                                K,
                                                descr_A,
                                                nnz_A,
                                                dAptr,
                                                dAcol,
                                                descr_B,
                                                nnz_B,
                                                dBptr,
                                                dBcol,
                                                nullptr,
                                                0,
                                                nullptr,
                                                nullptr,
                                                descr_C,
                                                dCptr,
                                                &hnnz_C_1,
                                                info,
                                                dbuffer));

    // Allocate result matrix
    rocsparse_int safe_nnz_C = std::max(hnnz_C_1, one);

    auto dCcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_nnz_C), device_free};
    auto dCval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_nnz_C), device_free};

    rocsparse_int* dCcol = (rocsparse_int*)dCcol_managed.get();
    T*             dCval = (T*)dCval_managed.get();

    if(!dCval || !dCcol)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error, "!dCval || !dCcol");
        return rocsparse_status_memory_error;
    }

    if(argus.unit_check)
    {
        // rocsparse pointer mode device
        auto dnnz_C_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int)), device_free};
        rocsparse_int* dnnz_C = (rocsparse_int*)dnnz_C_managed.get();

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                    trans_A,
                                                    trans_B,
                                                    M,
                                                    N,
                                                    K,
                                                    descr_A,
                                                    nnz_A,
                                                    dAptr,
                                                    dAcol,
                                                    descr_B,
                                                    nnz_B,
                                                    dBptr,
                                                    dBcol,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    descr_C,
                                                    dCptr,
                                                    dnnz_C,
                                                    info,
                                                    dbuffer));

        // Compute csrgemm host solution
        std::vector<rocsparse_int> hcsr_row_ptr_C_gold(M + 1);

        double cpu_time_used = get_time_us();

        rocsparse_int nnz_C_gold = csrgemm_nnz(M,
                                               N,
                                               K,
                                               h_alpha,
                                               hcsr_row_ptr_A.data(),
                                               hcsr_col_ind_A.data(),
                                               hcsr_row_ptr_B.data(),
                                               hcsr_col_ind_B.data(),
                                               (T*)nullptr,
                                               hcsr_row_ptr_D.data(),
                                               hcsr_col_ind_D.data(),
                                               hcsr_row_ptr_C_gold.data(),
                                               idx_base_A,
                                               idx_base_B,
                                               idx_base_C,
                                               idx_base_D);

        // If nnz_C == 0, we are done
        if(nnz_C_gold == 0)
        {
            return rocsparse_status_success;
        }

        std::vector<rocsparse_int> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T>             hcsr_val_C_gold(nnz_C_gold);

        csrgemm(M,
                N,
                K,
                h_alpha,
                hcsr_row_ptr_A.data(),
                hcsr_col_ind_A.data(),
                hcsr_val_A.data(),
                hcsr_row_ptr_B.data(),
                hcsr_col_ind_B.data(),
                hcsr_val_B.data(),
                (T*)nullptr,
                hcsr_row_ptr_D.data(),
                hcsr_col_ind_D.data(),
                hcsr_val_D.data(),
                hcsr_row_ptr_C_gold.data(),
                hcsr_col_ind_C_gold.data(),
                hcsr_val_C_gold.data(),
                idx_base_A,
                idx_base_B,
                idx_base_C,
                idx_base_D);

        cpu_time_used = get_time_us() - cpu_time_used;

        // Copy output from device to CPU
        rocsparse_int hnnz_C_2;
        CHECK_HIP_ERROR(hipMemcpy(&hnnz_C_2, dnnz_C, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // Check nnz of C
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_1);
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_2);

        // Compute csrgemm
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm(handle,
                                                trans_A,
                                                trans_B,
                                                M,
                                                N,
                                                K,
                                                h_alpha,
                                                descr_A,
                                                nnz_A,
                                                dAval,
                                                dAptr,
                                                dAcol,
                                                descr_B,
                                                nnz_B,
                                                dBval,
                                                dBptr,
                                                dBcol,
                                                (T*)nullptr,
                                                nullptr,
                                                0,
                                                (T*)nullptr,
                                                nullptr,
                                                nullptr,
                                                descr_C,
                                                dCval,
                                                dCptr,
                                                dCcol,
                                                info,
                                                dbuffer));

        // Copy output from device to CPU
        std::vector<rocsparse_int> hcsr_row_ptr_C(M + 1);
        std::vector<rocsparse_int> hcsr_col_ind_C(nnz_C_gold);
        std::vector<T>             hcsr_val_C_1(nnz_C_gold);
        std::vector<T>             hcsr_val_C_2(nnz_C_gold);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr_C.data(), dCptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C.data(),
                                  dCcol,
                                  sizeof(rocsparse_int) * nnz_C_gold,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_1.data(), dCval, sizeof(T) * nnz_C_gold, hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipMemset(dCval, 0, sizeof(T) * nnz_C_gold));

        // Device pointer mode
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm(handle,
                                                trans_A,
                                                trans_B,
                                                M,
                                                N,
                                                K,
                                                dalpha,
                                                descr_A,
                                                nnz_A,
                                                dAval,
                                                dAptr,
                                                dAcol,
                                                descr_B,
                                                nnz_B,
                                                dBval,
                                                dBptr,
                                                dBcol,
                                                (T*)nullptr,
                                                nullptr,
                                                0,
                                                (T*)nullptr,
                                                nullptr,
                                                nullptr,
                                                descr_C,
                                                dCval,
                                                dCptr,
                                                dCcol,
                                                info,
                                                dbuffer));

        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_2.data(), dCval, sizeof(T) * nnz_C_gold, hipMemcpyDeviceToHost));

        // Check structure and entries of C
        unit_check_general(1, M + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C.data());
        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C.data());
        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_1.data());
        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocsparse_csrgemm_nnz(handle,
                                  trans_A,
                                  trans_B,
                                  M,
                                  N,
                                  K,
                                  descr_A,
                                  nnz_A,
                                  dAptr,
                                  dAcol,
                                  descr_B,
                                  nnz_B,
                                  dBptr,
                                  dBcol,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  descr_C,
                                  dCptr,
                                  &hnnz_C_1,
                                  info,
                                  dbuffer);

            rocsparse_csrgemm(handle,
                              trans_A,
                              trans_B,
                              M,
                              N,
                              K,
                              h_alpha,
                              descr_A,
                              nnz_A,
                              dAval,
                              dAptr,
                              dAcol,
                              descr_B,
                              nnz_B,
                              dBval,
                              dBptr,
                              dBcol,
                              (T*)nullptr,
                              nullptr,
                              0,
                              (T*)nullptr,
                              nullptr,
                              nullptr,
                              descr_C,
                              dCval,
                              dCptr,
                              dCcol,
                              info,
                              dbuffer);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            rocsparse_csrgemm_nnz(handle,
                                  trans_A,
                                  trans_B,
                                  M,
                                  N,
                                  K,
                                  descr_A,
                                  nnz_A,
                                  dAptr,
                                  dAcol,
                                  descr_B,
                                  nnz_B,
                                  dBptr,
                                  dBcol,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  descr_C,
                                  dCptr,
                                  &hnnz_C_1,
                                  info,
                                  dbuffer);

            rocsparse_csrgemm(handle,
                              trans_A,
                              trans_B,
                              M,
                              N,
                              K,
                              h_alpha,
                              descr_A,
                              nnz_A,
                              dAval,
                              dAptr,
                              dAcol,
                              descr_B,
                              nnz_B,
                              dBval,
                              dBptr,
                              dBcol,
                              (T*)nullptr,
                              nullptr,
                              0,
                              (T*)nullptr,
                              nullptr,
                              nullptr,
                              descr_C,
                              dCval,
                              dCptr,
                              dCcol,
                              info,
                              dbuffer);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        // GFlops
        size_t flop = csrgemm_flops(M,
                                    h_alpha,
                                    hcsr_row_ptr_A.data(),
                                    hcsr_col_ind_A.data(),
                                    hcsr_row_ptr_B.data(),
                                    (T*)nullptr,
                                    hcsr_row_ptr_D.data(),
                                    idx_base_A);

        double gflops = flop / gpu_time_used / 1e6;

        // Memory transfers
        size_t data_A = sizeof(rocsparse_int) * (M + 1 + nnz_A) + sizeof(T) * nnz_A;
        size_t data_B = sizeof(rocsparse_int) * (K + 1 + nnz_B) + sizeof(T) * nnz_B;
        size_t data_C = sizeof(rocsparse_int) * (M + 1 + hnnz_C_1) + sizeof(T) * hnnz_C_1;

        double bandwidth = (data_A + data_B + data_C) / gpu_time_used / 1e6;

        // Output
        printf("m\t\tn\t\tk\t\tnnz_A\t\tnnz_B\t\tnnz_C\t\talpha\tbeta\tGFlop/s\tGB/s\tmsec\n");
        printf("%8d\t%8d\t%8d\t%9d\t%9d\t%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
               M,
               N,
               K,
               nnz_A,
               nnz_B,
               hnnz_C_1,
               *h_alpha,
               0.0,
               gflops,
               bandwidth,
               gpu_time_used);
    }

    return rocsparse_status_success;
}

#endif // TESTING_CSRGEMM_HPP
