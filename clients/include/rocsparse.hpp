/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _ROCSPARSE_HPP_
#define _ROCSPARSE_HPP_

#include <rocsparse.h>

namespace rocsparse {

template <typename T>
rocsparse_status rocsparse_axpyi(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const T *alpha,
                                 const T *xVal,
                                 const rocsparse_int *xInd,
                                 T *y,
                                 rocsparse_index_base idxBase);

}

#endif // _ROCSPARSE_HPP_
