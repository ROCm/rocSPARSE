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
#ifndef TESTING_CSRGEMM_HPP
#define TESTING_CSRGEMM_HPP

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <string>
#include <rocsparse.h>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_csrgemm_bad_arg(void)
{
    rocsparse_int M             = 100;
    rocsparse_int N             = 100;
    rocsparse_int K             = 100;
    rocsparse_int nnz_A         = 100;
    rocsparse_int nnz_B         = 100;
    rocsparse_operation trans_A = rocsparse_operation_none;
    rocsparse_operation trans_B = rocsparse_operation_none;
    rocsparse_int safe_size     = 100;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    rocsparse_mat_descr descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_B(new descr_struct);
    rocsparse_mat_descr descr_B = unique_ptr_descr_B->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    rocsparse_mat_descr descr_C = unique_ptr_descr_C->descr;

    auto dAptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dAcol_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dAval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dBptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dBcol_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dBval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dCptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dCcol_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dCval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    rocsparse_int* dAptr = (rocsparse_int*)dAptr_managed.get();
    rocsparse_int* dAcol = (rocsparse_int*)dAcol_managed.get();
    T* dAval             = (T*)dAval_managed.get();
    rocsparse_int* dBptr = (rocsparse_int*)dBptr_managed.get();
    rocsparse_int* dBcol = (rocsparse_int*)dBcol_managed.get();
    T* dBval             = (T*)dBval_managed.get();
    rocsparse_int* dCptr = (rocsparse_int*)dCptr_managed.get();
    rocsparse_int* dCcol = (rocsparse_int*)dCcol_managed.get();
    T* dCval             = (T*)dCval_managed.get();
    void* dbuffer        = (void*)dbuffer_managed.get();

    if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCval || !dCptr || !dCcol ||
       !dbuffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing rocsparse_csrgemm_buffer_size
    size_t size;

    // testing for(nullptr == dAptr)
    {
        rocsparse_int* dAptr_null = nullptr;

        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               descr_A,
                                               nnz_A,
                                               dAptr_null,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        rocsparse_int* dAcol_null = nullptr;

        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol_null,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        rocsparse_int* dBptr_null = nullptr;

        status = rocsparse_csrgemm_buffer_size(handle,
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
                                               dBptr_null,
                                               dBcol,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        rocsparse_int* dBcol_null = nullptr;

        status = rocsparse_csrgemm_buffer_size(handle,
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
                                               dBcol_null,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == buffer_size)
    {
        size_t* size_null = nullptr;

        status = rocsparse_csrgemm_buffer_size(handle,
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
                                               size_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }
    // testing for(nullptr == descr_A)
    {
        rocsparse_mat_descr descr_A_null = nullptr;

        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               descr_A_null,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        rocsparse_mat_descr descr_B_null = nullptr;

        status = rocsparse_csrgemm_buffer_size(handle,
                                               trans_A,
                                               trans_B,
                                               M,
                                               N,
                                               K,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B_null,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrgemm_buffer_size(handle_null,
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
                                               &size);
        verify_rocsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrgemm_nnz
    rocsparse_int nnz_C;

    // testing for(nullptr == dAptr)
    {
        rocsparse_int* dAptr_null = nullptr;

        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr_null,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        rocsparse_int* dAcol_null = nullptr;

        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol_null,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        rocsparse_int* dBptr_null = nullptr;

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
                                       dBptr_null,
                                       dBcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        rocsparse_int* dBcol_null = nullptr;

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
                                       dBcol_null,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        rocsparse_int* dCptr_null = nullptr;

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
                                       descr_C,
                                       dCptr_null,
                                       &nnz_C,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == nnz_C)
    {
        rocsparse_int* nnz_C_null = nullptr;

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
                                       descr_C,
                                       dCptr,
                                       nnz_C_null,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: nnz_C is nullptr");
    }
    // testing for(nullptr == nnz_C)
    {
        void* dbuffer_null = nullptr;

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
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       dbuffer_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr_A)
    {
        rocsparse_mat_descr descr_A_null = nullptr;

        status = rocsparse_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       M,
                                       N,
                                       K,
                                       descr_A_null,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        rocsparse_mat_descr descr_B_null = nullptr;

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
                                       descr_B_null,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        rocsparse_mat_descr descr_C_null = nullptr;

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
                                       descr_C_null,
                                       dCptr,
                                       &nnz_C,
                                       dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrgemm_nnz(handle_null,
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
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       dbuffer);
        verify_rocsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrgemm

    // testing for(nullptr == dAval)
    {
        T* dAval_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   descr_A,
                                   nnz_A,
                                   dAval_null,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAval is nullptr");
    }
    // testing for(nullptr == dAptr)
    {
        rocsparse_int* dAptr_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr_null,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        rocsparse_int* dAcol_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol_null,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == dBval)
    {
        T* dBval_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval_null,
                                   dBptr,
                                   dBcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBval is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        rocsparse_int* dBptr_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr_null,
                                   dBcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        rocsparse_int* dBcol_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol_null,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == dCval)
    {
        T* dCval_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval_null,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCval is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        rocsparse_int* dCptr_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr_null,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == dCcol)
    {
        rocsparse_int* dCcol_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol_null,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dCcol is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr_A)
    {
        rocsparse_mat_descr descr_A_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   descr_A_null,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        rocsparse_mat_descr descr_B_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
                                   descr_A,
                                   nnz_A,
                                   dAval,
                                   dAptr,
                                   dAcol,
                                   descr_B_null,
                                   nnz_B,
                                   dBval,
                                   dBptr,
                                   dBcol,
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        rocsparse_mat_descr descr_C_null = nullptr;

        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C_null,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrgemm(handle_null,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
                                   dbuffer);
        verify_rocsparse_status_invalid_handle(status);
    }
}

static rocsparse_int csrgemm_nnz(rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int k,
                                 const rocsparse_int* csr_row_ptr_A,
                                 const rocsparse_int* csr_col_ind_A,
                                 const rocsparse_int* csr_row_ptr_B,
                                 const rocsparse_int* csr_col_ind_B,
                                 rocsparse_int* csr_row_ptr_C,
                                 rocsparse_index_base idx_base_A,
                                 rocsparse_index_base idx_base_B,
                                 rocsparse_index_base idx_base_C)
{
    std::vector<rocsparse_int> nnz(n, -1);

    // Index base
    csr_row_ptr_C[0] = idx_base_C;

    // Loop over rows of A
    for(rocsparse_int i = 0; i < m; ++i)
    {
        // Initialize csr row pointer with previous row offset
        csr_row_ptr_C[i + 1] = csr_row_ptr_C[i];

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

    return csr_row_ptr_C[m];
}

template <typename T>
static void csrgemm(rocsparse_int m,
                    rocsparse_int n,
                    rocsparse_int k,
                    const rocsparse_int* csr_row_ptr_A,
                    const rocsparse_int* csr_col_ind_A,
                    const T* csr_val_A,
                    const rocsparse_int* csr_row_ptr_B,
                    const rocsparse_int* csr_col_ind_B,
                    const T* csr_val_B,
                    const rocsparse_int* csr_row_ptr_C,
                    rocsparse_int* csr_col_ind_C,
                    T* csr_val_C,
                    rocsparse_index_base idx_base_A,
                    rocsparse_index_base idx_base_B,
                    rocsparse_index_base idx_base_C)
{
    std::vector<rocsparse_int> nnz(n, -1);

    // Loop over rows of A
    for(rocsparse_int i = 0; i < m; ++i)
    {
        rocsparse_int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
        rocsparse_int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

        rocsparse_int row_begin_C = csr_row_ptr_C[i] - idx_base_C;
        rocsparse_int row_end_C   = row_begin_C;

        // Loop over columns of A
        for(rocsparse_int j = row_begin_A; j < row_end_A; ++j)
        {
            // Current column of A
            rocsparse_int col_A = csr_col_ind_A[j] - idx_base_A;
            // Current value of A
            T val_A = csr_val_A[j];

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
                    csr_col_ind_C[row_end_C] = col_B;
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

    // Sort column indices within each row
    for(rocsparse_int i = 0; i < m; ++i)
    {
        for(rocsparse_int j = csr_row_ptr_C[i]; j < csr_row_ptr_C[i + 1]; ++j)
        {
            for(rocsparse_int jj = csr_row_ptr_C[i]; jj < csr_row_ptr_C[i + 1] - 1; ++jj)
            {
                if(csr_col_ind_C[jj] > csr_col_ind_C[jj + 1])
                {
                    // swap elements
                    rocsparse_int ind = csr_col_ind_C[jj];
                    T val             = csr_val_C[jj];

                    csr_col_ind_C[jj] = csr_col_ind_C[jj + 1];
                    csr_val_C[jj]     = csr_val_C[jj + 1];

                    csr_col_ind_C[jj + 1] = ind;
                    csr_val_C[jj + 1]     = val;
                }
            }
        }
    }
}

template <typename T>
rocsparse_status testing_csrgemm(Arguments argus)
{
    rocsparse_int safe_size         = 100;
    rocsparse_int M                 = argus.M;
    rocsparse_int N                 = argus.N;
    rocsparse_int K                 = argus.K;
    rocsparse_operation trans_A     = argus.transA;
    rocsparse_operation trans_B     = argus.transB;
    rocsparse_index_base idx_base_A = argus.idx_base;
    rocsparse_index_base idx_base_B = argus.idx_base2;
    std::string binfile             = "";
    std::string filename            = "";
    std::string rocalution          = "";
    rocsparse_status status;
    size_t size;

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
    rocsparse_handle handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr_A(new descr_struct);
    rocsparse_mat_descr descr_A = test_descr_A->descr;

    std::unique_ptr<descr_struct> test_descr_B(new descr_struct);
    rocsparse_mat_descr descr_B = test_descr_B->descr;

    std::unique_ptr<descr_struct> test_descr_C(new descr_struct);
    rocsparse_mat_descr descr_C = test_descr_C->descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_A, idx_base_A));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_B, idx_base_B));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_index_base(descr_C, idx_base_A)); // TODO extra idx base for C?

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
    if(M <= 0 || N <= 0 || K <= 0 || nnz_A <= 0 || nnz_B <= 0)
    {
        auto dAptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dAcol_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dAval_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dBptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dBcol_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dBval_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dCptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dCcol_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dCval_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto buffer_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        rocsparse_int* dAptr = (rocsparse_int*)dAptr_managed.get();
        rocsparse_int* dAcol = (rocsparse_int*)dAcol_managed.get();
        T* dAval             = (T*)dAval_managed.get();
        rocsparse_int* dBptr = (rocsparse_int*)dBptr_managed.get();
        rocsparse_int* dBcol = (rocsparse_int*)dBcol_managed.get();
        T* dBval             = (T*)dBval_managed.get();
        rocsparse_int* dCptr = (rocsparse_int*)dCptr_managed.get();
        rocsparse_int* dCcol = (rocsparse_int*)dCcol_managed.get();
        T* dCval             = (T*)dCval_managed.get();
        void* buffer         = (void*)buffer_managed.get();

        if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCval || !dCptr || !dCcol ||
           !buffer)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dAptr || !dAcol || !dAval || "
                                            "!dBptr || !dBcol || !dBval || "
                                            "!dCptr || !dCcol || !dCval || "
                                            "!buffer");
            return rocsparse_status_memory_error;
        }

        // Test rocsparse_csrgemm_buffer_size
        status = rocsparse_csrgemm_buffer_size(handle,
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

        // Test rocsparse_csrgemm_nnz
        rocsparse_int nnz_C;
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
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
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

        // Test rocsparse_csrgemm
        status = rocsparse_csrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol,
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

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr_A;
    std::vector<rocsparse_int> hcsr_col_ind_A;
    std::vector<T> hcsr_val_A;

    // Initial Data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(binfile.c_str(),
                           M,
                           K,
                           nnz_A,
                           hcsr_row_ptr_A,
                           hcsr_col_ind_A,
                           hcsr_val_A,
                           idx_base_A) != 0)
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
                                  idx_base_A) != 0)
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
                               idx_base_A) != 0)
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
    std::vector<rocsparse_int> hcsr_row_ptr_B(K + 1, 0);
    std::vector<rocsparse_int> hcsr_col_ind_B(nnz_B);
    std::vector<T> hcsr_val_B(nnz_B);

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
    auto dAptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (M + 1)), device_free};
    auto dAcol_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz_A), device_free};
    auto dAval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dBptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (K + 1)), device_free};
    auto dBcol_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz_B), device_free};
    auto dBval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};

    rocsparse_int* dAptr = (rocsparse_int*)dAptr_managed.get();
    rocsparse_int* dAcol = (rocsparse_int*)dAcol_managed.get();
    T* dAval             = (T*)dAval_managed.get();
    rocsparse_int* dBptr = (rocsparse_int*)dBptr_managed.get();
    rocsparse_int* dBcol = (rocsparse_int*)dBcol_managed.get();
    T* dBval             = (T*)dBval_managed.get();

    if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dAval || !dAptr || !dAcol || "
                                        "!dBval || !dBptr || !dBcol");
        return rocsparse_status_memory_error;
    }

    // copy data from CPU to device
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

    // Obtain csrgemm buffer size
    CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_buffer_size(handle,
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
                                                        &size));

    // Allocate buffer on the device
    auto dbuffer_managed = rocsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dbuffer)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error, "!dbuffer");
        return rocsparse_status_memory_error;
    }

    // Allocate result matrix row pointer array
    auto dCptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (M + 1)), device_free};

    rocsparse_int* dCptr = (rocsparse_int*)dCptr_managed.get();

    if(!dCptr)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error, "!dCptr");
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
                                                descr_C,
                                                dCptr,
                                                &hnnz_C_1,
                                                dbuffer));

    // Allocate result matrix
    auto dCcol_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * hnnz_C_1), device_free};
    auto dCval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_C_1), device_free};

    rocsparse_int* dCcol = (rocsparse_int*)dCcol_managed.get();
    T* dCval             = (T*)dCval_managed.get();

    if(!dCval || !dCcol)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error, "!dCval || !dCcol");
        return rocsparse_status_memory_error;
    }

    if(argus.unit_check)
    {
        // rocsparse pointer mode device
        auto dnnz_C_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int)), device_free};
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
                                                    descr_C,
                                                    dCptr,
                                                    dnnz_C,
                                                    dbuffer));

        // Compute csrgemm host solution
        std::vector<rocsparse_int> hcsr_row_ptr_C_gold(M + 1);

        double cpu_time_used = get_time_us();

        rocsparse_int nnz_C_gold = csrgemm_nnz(M,
                                               N,
                                               K,
                                               hcsr_row_ptr_A.data(),
                                               hcsr_col_ind_A.data(),
                                               hcsr_row_ptr_B.data(),
                                               hcsr_col_ind_B.data(),
                                               hcsr_row_ptr_C_gold.data(),
                                               idx_base_A,
                                               idx_base_B,
                                               idx_base_A);

        std::vector<rocsparse_int> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T> hcsr_val_C_gold(nnz_C_gold);

        csrgemm(M,
                N,
                K,
                hcsr_row_ptr_A.data(),
                hcsr_col_ind_A.data(),
                hcsr_val_A.data(),
                hcsr_row_ptr_B.data(),
                hcsr_col_ind_B.data(),
                hcsr_val_B.data(),
                hcsr_row_ptr_C_gold.data(),
                hcsr_col_ind_C_gold.data(),
                hcsr_val_C_gold.data(),
                idx_base_A,
                idx_base_B,
                idx_base_A);

        cpu_time_used = get_time_us() - cpu_time_used;

        // Copy output from device to CPU
        rocsparse_int hnnz_C_2;
        CHECK_HIP_ERROR(hipMemcpy(&hnnz_C_2, dnnz_C, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // Check nnz of C
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_1);
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_2);

        // Compute csrgemm
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm(handle,
                                                trans_A,
                                                trans_B,
                                                M,
                                                N,
                                                K,
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
                                                descr_C,
                                                dCval,
                                                dCptr,
                                                dCcol,
                                                dbuffer));

        // Copy output from device to CPU
        std::vector<rocsparse_int> hcsr_row_ptr_C(M + 1);
        std::vector<rocsparse_int> hcsr_col_ind_C(nnz_C_gold);
        std::vector<T> hcsr_val_C(nnz_C_gold);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr_C.data(), dCptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C.data(),
                                  dCcol,
                                  sizeof(rocsparse_int) * nnz_C_gold,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C.data(), dCval, sizeof(T) * nnz_C_gold, hipMemcpyDeviceToHost));

        // Check structure and entries of C
        unit_check_general(1, M + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C.data());
        //        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(),
        //        hcsr_col_ind_C.data());
        //        unit_check_general(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C.data());
    }

    if(argus.timing)
    {
        // TODO
    }

    return rocsparse_status_success;
}

#endif // TESTING_CSRGEMM_HPP
