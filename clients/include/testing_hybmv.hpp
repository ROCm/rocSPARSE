/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_HYBMV_HPP
#define TESTING_HYBMV_HPP

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <string>
#include <rocsparse.h>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_hybmv_bad_arg(void)
{
    rocsparse_int safe_size   = 100;
    T alpha                   = 0.6;
    T beta                    = 0.2;
    rocsparse_operation trans = rocsparse_operation_none;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    rocsparse_mat_descr descr = unique_ptr_descr->descr;

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    rocsparse_hyb_mat hyb = unique_ptr_hyb->hyb;

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

        status = rocsparse_hybmv(handle, trans, &alpha, descr, hyb, dx_null, &beta, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: dx is nullptr");
    }
    // testing for(nullptr == dy)
    {
        T* dy_null = nullptr;

        status = rocsparse_hybmv(handle, trans, &alpha, descr, hyb, dx, &beta, dy_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: dy is nullptr");
    }
    // testing for(nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = rocsparse_hybmv(handle, trans, d_alpha_null, descr, hyb, dx, &beta, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == d_beta)
    {
        T* d_beta_null = nullptr;

        status = rocsparse_hybmv(handle, trans, &alpha, descr, hyb, dx, d_beta_null, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: beta is nullptr");
    }
    // testing for(nullptr == hyb)
    {
        rocsparse_hyb_mat hyb_null = nullptr;

        status = rocsparse_hybmv(handle, trans, &alpha, descr, hyb_null, dx, &beta, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == descr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_hybmv(handle, trans, &alpha, descr_null, hyb, dx, &beta, dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_hybmv(handle_null, trans, &alpha, descr, hyb, dx, &beta, dy);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
rocsparse_status testing_hybmv(Arguments argus)
{
    return rocsparse_status_success;
}

#endif // TESTING_HYBMV_HPP
