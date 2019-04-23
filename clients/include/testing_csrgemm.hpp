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

template <typename T>
rocsparse_status testing_csrgemm(Arguments argus)
{
    return rocsparse_status_success;
}

#endif // TESTING_CSRGEMM_HPP
