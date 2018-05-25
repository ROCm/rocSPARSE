/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef ARG_CHECK_HPP
#define ARG_CHECK_HPP

#include <rocsparse.h>

void verify_rocsparse_status_invalid_pointer(rocsparse_status status, const char* message);

void verify_rocsparse_status_invalid_size(rocsparse_status status, const char* message);

void verify_rocsparse_status_invalid_value(rocsparse_status status, const char* message);

void verify_rocsparse_status_invalid_handle(rocsparse_status status);

void verify_rocsparse_status_success(rocsparse_status status, const char* message);

#endif // ARG_CHECK_HPP
