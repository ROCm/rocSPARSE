/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_DOTCI_HPP
#define ROCSPARSE_DOTCI_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "dotci_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_dotci_template(rocsparse_handle handle,
                                          rocsparse_int nnz,
                                          const T* x_val,
                                          const rocsparse_int* x_ind,
                                          const T* y,
                                          T* result,
                                          rocsparse_index_base idx_base)
{
    return rocsparse_status_not_implemented;
}

#endif // ROCSPARSE_DOTCI_HPP
