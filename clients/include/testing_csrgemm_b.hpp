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
#ifndef TESTING_CSRGEMM_B_HPP
#define TESTING_CSRGEMM_B_HPP

#include "rocsparse.hpp"
#include "rocsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <iomanip>
#include <iostream>
#include <rocsparse.h>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_csrgemm_b_bad_arg(void)
{
    rocsparse_int    M         = 100;
    rocsparse_int    N         = 100;
    rocsparse_int    nnz_D     = 100;
    rocsparse_int    safe_size = 100;
    rocsparse_status status;

    T beta = 1.0;

    size_t        size;
    rocsparse_int nnz_C;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    rocsparse_mat_descr           descr_C = unique_ptr_descr_C->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_D(new descr_struct);
    rocsparse_mat_descr           descr_D = unique_ptr_descr_D->descr;

    std::unique_ptr<mat_info_struct> unique_ptr_info(new mat_info_struct);
    rocsparse_mat_info               info = unique_ptr_info->info;

    auto dCptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dCcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dCval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dDptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dDcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dDval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    rocsparse_int* dCptr   = (rocsparse_int*)dCptr_managed.get();
    rocsparse_int* dCcol   = (rocsparse_int*)dCcol_managed.get();
    T*             dCval   = (T*)dCval_managed.get();
    rocsparse_int* dDptr   = (rocsparse_int*)dDptr_managed.get();
    rocsparse_int* dDcol   = (rocsparse_int*)dDcol_managed.get();
    T*             dDval   = (T*)dDval_managed.get();
    void*          dbuffer = (void*)dbuffer_managed.get();

    if(!dCval || !dCptr || !dCcol || !dDval || !dDptr || !dDcol || !dbuffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // We need to test C = beta * D

    // Scenario: alpha == nullptr and beta != nullptr

    // testing rocsparse_csrgemm_buffer_size

    // testing for(nullptr == handle)
    {
        status = rocsparse_csrgemm_buffer_size(nullptr,
                                               rocsparse_operation_none,
                                               rocsparse_operation_none,
                                               M,
                                               N,
                                               0,
                                               (T*)nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               &beta,
                                               descr_D,
                                               nnz_D,
                                               dDptr,
                                               dDcol,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == alpha and nullptr == beta)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_none,
                                               M,
                                               N,
                                               0,
                                               (T*)nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               (T*)nullptr,
                                               descr_D,
                                               nnz_D,
                                               dDptr,
                                               dDcol,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: alpha and beta are nullptr");
    }
    // testing for(nullptr == descr_D)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_none,
                                               M,
                                               N,
                                               0,
                                               (T*)nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               &beta,
                                               nullptr,
                                               nnz_D,
                                               dDptr,
                                               dDcol,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_D is nullptr");
    }
    // testing for(nullptr == dDptr)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_none,
                                               M,
                                               N,
                                               0,
                                               (T*)nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               &beta,
                                               descr_D,
                                               nnz_D,
                                               nullptr,
                                               dDcol,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dDptr is nullptr");
    }
    // testing for(nullptr == dDcol)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_none,
                                               M,
                                               N,
                                               0,
                                               (T*)nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               &beta,
                                               descr_D,
                                               nnz_D,
                                               dDptr,
                                               nullptr,
                                               info,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dDcol is nullptr");
    }

    // testing for(nullptr == info)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_none,
                                               M,
                                               N,
                                               0,
                                               (T*)nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               &beta,
                                               descr_D,
                                               nnz_D,
                                               dDptr,
                                               dDcol,
                                               nullptr,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == size)
    {
        status = rocsparse_csrgemm_buffer_size(handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_none,
                                               M,
                                               N,
                                               0,
                                               (T*)nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               0,
                                               nullptr,
                                               nullptr,
                                               &beta,
                                               descr_D,
                                               nnz_D,
                                               dDptr,
                                               dDcol,
                                               info,
                                               nullptr);
        verify_rocsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }

    // We need one successful call to create the info structure
    status = rocsparse_csrgemm_buffer_size(handle,
                                           rocsparse_operation_none,
                                           rocsparse_operation_none,
                                           M,
                                           N,
                                           0,
                                           (T*)nullptr,
                                           nullptr,
                                           0,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           0,
                                           nullptr,
                                           nullptr,
                                           &beta,
                                           descr_D,
                                           nnz_D,
                                           dDptr,
                                           dDcol,
                                           info,
                                           &size);
    verify_rocsparse_status_success(status, "Success");

    // testing rocsparse_csrgemm_nnz

    // testing for(nullptr == handle)
    {
        status = rocsparse_csrgemm_nnz(nullptr,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == descr_D)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_D is nullptr");
    }
    // testing for(nullptr == dDptr)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       nullptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dDptr is nullptr");
    }
    // testing for(nullptr == dDcol)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dDcol is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       nullptr,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       nullptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == nnz_C)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       nullptr,
                                       info,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: nnz_C is nullptr");
    }
    // testing for(nullptr == info)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       nullptr,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        status = rocsparse_csrgemm_nnz(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
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
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   dDcol,
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
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   (T*)nullptr,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   dDcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: alpha and beta are nullptr");
    }
    // testing for(nullptr == descr_D)
    {
        status = rocsparse_csrgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   nullptr,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   dDcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_D is nullptr");
    }
    // testing for(nullptr == dDval)
    {
        status = rocsparse_csrgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   (T*)nullptr,
                                   dDptr,
                                   dDcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dDval is nullptr");
    }
    // testing for(nullptr == dDptr)
    {
        status = rocsparse_csrgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   nullptr,
                                   dDcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dDptr is nullptr");
    }
    // testing for(nullptr == dDcol)
    {
        status = rocsparse_csrgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   nullptr,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dDcol is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        status = rocsparse_csrgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   dDcol,
                                   nullptr,
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
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   dDcol,
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
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   dDcol,
                                   descr_C,
                                   dCval,
                                   nullptr,
                                   dCcol,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == dCcol)
    {
        status = rocsparse_csrgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   dDcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   nullptr,
                                   info,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCcol is nullptr");
    }
    // testing for(nullptr == info)
    {
        status = rocsparse_csrgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   dDcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   nullptr,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        status = rocsparse_csrgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   M,
                                   N,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   0,
                                   (T*)nullptr,
                                   nullptr,
                                   nullptr,
                                   &beta,
                                   descr_D,
                                   nnz_D,
                                   dDval,
                                   dDptr,
                                   dDcol,
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
rocsparse_status testing_csrgemm_b(Arguments argus)
{
    rocsparse_int        safe_size  = 100;
    rocsparse_int        M          = argus.M;
    rocsparse_int        N          = argus.N;
    rocsparse_index_base idx_base_C = argus.idx_base3;
    rocsparse_index_base idx_base_D = argus.idx_base4;
    T                    beta       = argus.beta;
    std::string          binfile    = "";
    std::string          filename   = "";
    std::string          rocalution = "";

    rocsparse_status status;
    size_t           size;

    T* h_beta = &beta;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(M == -99 && N == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        M = N = safe_size;
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

    std::unique_ptr<descr_struct> test_descr_C(new descr_struct);
    rocsparse_mat_descr           descr_C = test_descr_C->descr;

    std::unique_ptr<descr_struct> test_descr_D(new descr_struct);
    rocsparse_mat_descr           descr_D = test_descr_D->descr;

    std::unique_ptr<mat_info_struct> test_info(new mat_info_struct);
    rocsparse_mat_info               info = test_info->info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_C, idx_base_C));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_D, idx_base_D));

    // Determine number of non-zero elements
    double scale = 0.02;
    if(M > 1000 || N > 1000)
    {
        scale = 2.0 / std::max(M, N);
    }
    rocsparse_int nnz_D = M * scale * N;

    // Argument sanity check before allocating invalid memory
    // alpha == nullptr and beta != nullptr
    {
        auto dCptr_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dCcol_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dCval_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dDptr_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dDcol_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dDval_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto buffer_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        rocsparse_int* dCptr  = (rocsparse_int*)dCptr_managed.get();
        rocsparse_int* dCcol  = (rocsparse_int*)dCcol_managed.get();
        T*             dCval  = (T*)dCval_managed.get();
        rocsparse_int* dDptr  = (rocsparse_int*)dDptr_managed.get();
        rocsparse_int* dDcol  = (rocsparse_int*)dDcol_managed.get();
        T*             dDval  = (T*)dDval_managed.get();
        void*          buffer = (void*)buffer_managed.get();

        if(!dCval || !dCptr || !dCcol || !dDval || !dDptr || !dDcol || !buffer)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dCptr || !dCcol || !dCval || "
                                            "!dDptr || !dDcol || !dDval || "
                                            "!buffer");
            return rocsparse_status_memory_error;
        }

        if(M <= 0 || N <= 0 || nnz_D <= 0)
        {
            rocsparse_int nnz_C;
            status = rocsparse_csrgemm_buffer_size(handle,
                                                   rocsparse_operation_none,
                                                   rocsparse_operation_none,
                                                   M,
                                                   N,
                                                   0,
                                                   (T*)nullptr,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   nullptr,
                                                   h_beta,
                                                   descr_D,
                                                   nnz_D,
                                                   dDptr,
                                                   dDcol,
                                                   info,
                                                   &size);

            if(M < 0 || N < 0 || nnz_D < 0)
            {
                verify_rocsparse_status_invalid_size(status, "Error: M < 0 || N < 0 || nnz_D < 0");
            }
            else
            {
                verify_rocsparse_status_success(status, "M >= 0 && N >= 0 && nnz_D >= 0");
            }

            status = rocsparse_csrgemm_nnz(handle,
                                           rocsparse_operation_none,
                                           rocsparse_operation_none,
                                           M,
                                           N,
                                           0,
                                           nullptr,
                                           0,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           0,
                                           nullptr,
                                           nullptr,
                                           descr_D,
                                           nnz_D,
                                           dDptr,
                                           dDcol,
                                           descr_C,
                                           dCptr,
                                           &nnz_C,
                                           info,
                                           buffer);

            if(M < 0 || N < 0 || nnz_D < 0)
            {
                verify_rocsparse_status_invalid_size(status, "Error: M < 0 || N < 0 || nnz_D < 0");
            }
            else
            {
                verify_rocsparse_status_success(status, "M >= 0 && N >= 0 && nnz_D >= 0");
            }

            status = rocsparse_csrgemm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       M,
                                       N,
                                       0,
                                       (T*)nullptr,
                                       nullptr,
                                       0,
                                       (T*)nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       (T*)nullptr,
                                       nullptr,
                                       nullptr,
                                       h_beta,
                                       descr_D,
                                       nnz_D,
                                       dDval,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       dCval,
                                       dCptr,
                                       dCcol,
                                       info,
                                       buffer);

            if(M < 0 || N < 0 || nnz_D < 0)
            {
                verify_rocsparse_status_invalid_size(status, "Error: M < 0 || N < 0 || nnz_D < 0");
            }
            else
            {
                verify_rocsparse_status_success(status, "M >= 0 && N >= 0 && nnz_D >= 0");
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
               binfile.c_str(), M, N, nnz_D, hcsr_row_ptr_D, hcsr_col_ind_D, hcsr_val_D, idx_base_D)
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
                                  N,
                                  nnz_D,
                                  hcsr_row_ptr_D,
                                  hcsr_col_ind_D,
                                  hcsr_val_D,
                                  idx_base_D)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", rocalution.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        M = N = gen_2d_laplacian(
            argus.laplacian, hcsr_row_ptr_D, hcsr_col_ind_D, hcsr_val_D, idx_base_D);
        nnz_D = hcsr_row_ptr_D[M];
    }
    else
    {
        std::vector<rocsparse_int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(),
                               M,
                               N,
                               nnz_D,
                               hcoo_row_ind,
                               hcsr_col_ind_D,
                               hcsr_val_D,
                               idx_base_D)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(M, N, nnz_D, hcoo_row_ind, hcsr_col_ind_D, hcsr_val_D, idx_base_D);
        }

        // Convert COO to CSR
        hcsr_row_ptr_D.resize(M + 1, 0);
        for(rocsparse_int i = 0; i < nnz_D; ++i)
        {
            ++hcsr_row_ptr_D[hcoo_row_ind[i] + 1 - idx_base_D];
        }

        hcsr_row_ptr_D[0] = idx_base_D;
        for(rocsparse_int i = 0; i < M; ++i)
        {
            hcsr_row_ptr_D[i + 1] += hcsr_row_ptr_D[i];
        }
    }

    // Allocate memory on device
    rocsparse_int one        = 1;
    rocsparse_int safe_nnz_D = std::max(nnz_D, one);

    auto dDptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (M + 1)), device_free};
    auto dDcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_nnz_D), device_free};
    auto dDval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_nnz_D), device_free};
    auto dCptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (M + 1)), device_free};
    auto dbeta_managed = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    rocsparse_int* dDptr = (rocsparse_int*)dDptr_managed.get();
    rocsparse_int* dDcol = (rocsparse_int*)dDcol_managed.get();
    T*             dDval = (T*)dDval_managed.get();
    rocsparse_int* dCptr = (rocsparse_int*)dCptr_managed.get();
    T*             dbeta = (T*)dbeta_managed.get();

    if(!dDval || !dDptr || !dDcol || !dCptr || !dbeta)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dDval || !dDptr || !dDcol || "
                                        "!dCptr || !dbeta");
        return rocsparse_status_memory_error;
    }

    CHECK_HIP_ERROR(hipMemcpy(
        dDptr, hcsr_row_ptr_D.data(), sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dDcol, hcsr_col_ind_D.data(), sizeof(rocsparse_int) * nnz_D, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dDval, hcsr_val_D.data(), sizeof(T) * nnz_D, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbeta, h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Obtain csrgemm buffer size
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_buffer_size(handle,
                                                        rocsparse_operation_none,
                                                        rocsparse_operation_none,
                                                        M,
                                                        N,
                                                        0,
                                                        (T*)nullptr,
                                                        nullptr,
                                                        0,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        0,
                                                        nullptr,
                                                        nullptr,
                                                        h_beta,
                                                        descr_D,
                                                        nnz_D,
                                                        dDptr,
                                                        dDcol,
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
                                                            rocsparse_operation_none,
                                                            rocsparse_operation_none,
                                                            M,
                                                            N,
                                                            0,
                                                            (T*)nullptr,
                                                            nullptr,
                                                            0,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            0,
                                                            nullptr,
                                                            nullptr,
                                                            dbeta,
                                                            descr_D,
                                                            nnz_D,
                                                            dDptr,
                                                            dDcol,
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
                                                rocsparse_operation_none,
                                                rocsparse_operation_none,
                                                M,
                                                N,
                                                0,
                                                nullptr,
                                                0,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                0,
                                                nullptr,
                                                nullptr,
                                                descr_D,
                                                nnz_D,
                                                dDptr,
                                                dDcol,
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
                                                    rocsparse_operation_none,
                                                    rocsparse_operation_none,
                                                    M,
                                                    N,
                                                    0,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    descr_D,
                                                    nnz_D,
                                                    dDptr,
                                                    dDcol,
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
                                               0,
                                               (T*)nullptr,
                                               hcsr_row_ptr_A.data(),
                                               hcsr_col_ind_A.data(),
                                               hcsr_row_ptr_B.data(),
                                               hcsr_col_ind_B.data(),
                                               h_beta,
                                               hcsr_row_ptr_D.data(),
                                               hcsr_col_ind_D.data(),
                                               hcsr_row_ptr_C_gold.data(),
                                               rocsparse_index_base_zero,
                                               rocsparse_index_base_zero,
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
                0,
                (T*)nullptr,
                hcsr_row_ptr_A.data(),
                hcsr_col_ind_A.data(),
                hcsr_val_A.data(),
                hcsr_row_ptr_B.data(),
                hcsr_col_ind_B.data(),
                hcsr_val_B.data(),
                h_beta,
                hcsr_row_ptr_D.data(),
                hcsr_col_ind_D.data(),
                hcsr_val_D.data(),
                hcsr_row_ptr_C_gold.data(),
                hcsr_col_ind_C_gold.data(),
                hcsr_val_C_gold.data(),
                rocsparse_index_base_zero,
                rocsparse_index_base_zero,
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
                                                rocsparse_operation_none,
                                                rocsparse_operation_none,
                                                M,
                                                N,
                                                0,
                                                (T*)nullptr,
                                                nullptr,
                                                0,
                                                (T*)nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                0,
                                                (T*)nullptr,
                                                nullptr,
                                                nullptr,
                                                h_beta,
                                                descr_D,
                                                nnz_D,
                                                dDval,
                                                dDptr,
                                                dDcol,
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
                                                rocsparse_operation_none,
                                                rocsparse_operation_none,
                                                M,
                                                N,
                                                0,
                                                (T*)nullptr,
                                                nullptr,
                                                0,
                                                (T*)nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                0,
                                                (T*)nullptr,
                                                nullptr,
                                                nullptr,
                                                dbeta,
                                                descr_D,
                                                nnz_D,
                                                dDval,
                                                dDptr,
                                                dDcol,
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
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  M,
                                  N,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  descr_D,
                                  nnz_D,
                                  dDptr,
                                  dDcol,
                                  descr_C,
                                  dCptr,
                                  &hnnz_C_1,
                                  info,
                                  dbuffer);

            rocsparse_csrgemm(handle,
                              rocsparse_operation_none,
                              rocsparse_operation_none,
                              M,
                              N,
                              0,
                              (T*)nullptr,
                              nullptr,
                              0,
                              (T*)nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              0,
                              (T*)nullptr,
                              nullptr,
                              nullptr,
                              h_beta,
                              descr_D,
                              nnz_D,
                              dDval,
                              dDptr,
                              dDcol,
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
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  M,
                                  N,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  descr_D,
                                  nnz_D,
                                  dDptr,
                                  dDcol,
                                  descr_C,
                                  dCptr,
                                  &hnnz_C_1,
                                  info,
                                  dbuffer);

            rocsparse_csrgemm(handle,
                              rocsparse_operation_none,
                              rocsparse_operation_none,
                              M,
                              N,
                              0,
                              (T*)nullptr,
                              nullptr,
                              0,
                              (T*)nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              0,
                              (T*)nullptr,
                              nullptr,
                              nullptr,
                              h_beta,
                              descr_D,
                              nnz_D,
                              dDval,
                              dDptr,
                              dDcol,
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
                                    (T*)nullptr,
                                    hcsr_row_ptr_A.data(),
                                    hcsr_col_ind_A.data(),
                                    hcsr_row_ptr_B.data(),
                                    h_beta,
                                    hcsr_row_ptr_D.data(),
                                    rocsparse_index_base_zero);

        double gflops = flop / gpu_time_used / 1e6;

        // Memory transfers
        size_t data_C = sizeof(rocsparse_int) * (M + 1 + hnnz_C_1) + sizeof(T) * hnnz_C_1;
        size_t data_D = sizeof(rocsparse_int) * (M + 1 + nnz_D) + sizeof(T) * nnz_D;

        double bandwidth = (data_C + data_D) / gpu_time_used / 1e6;

        // Output
        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);
        std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "k"
                  << std::setw(12) << "nnz_A" << std::setw(12) << "nnz_B" << std::setw(12)
                  << "nnz_D" << std::setw(12) << "nnz_C" << std::setw(12) << "alpha"
                  << std::setw(12) << "beta" << std::setw(12) << "GFlop/s" << std::setw(12)
                  << "GB/s" << std::setw(12) << "msec" << std::endl;
        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << 0 << std::setw(12)
                  << 0 << std::setw(12) << 0 << std::setw(12) << nnz_D << std::setw(12) << hnnz_C_1
                  << std::setw(12) << 0.0 << std::setw(12) << *h_beta << std::setw(12) << gflops
                  << std::setw(12) << bandwidth << std::setw(12) << gpu_time_used << std::endl;
    }

    return rocsparse_status_success;
}

#endif // TESTING_CSRGEMM_B_HPP
