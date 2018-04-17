/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "matrix.h"
#include "rocsparse.h"

extern "C" rocsparseStatus_t rocsparseCreateMatDescr(rocsparseMatDescr_t *descrA)
{
    if (descrA == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else
    {
        return ROCSPARSE_STATUS_SUCCESS;
    }
}

extern "C" rocsparseStatus_t rocsparseDestroyMatDescr(rocsparseMatDescr_t descrA)
{
    return ROCSPARSE_STATUS_SUCCESS;
}
