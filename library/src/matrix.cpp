/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "matrix.h"
#include "rocsparse.h"



rocsparseMatDescr::rocsparseMatDescr()
{
    base = ROCSPARSE_INDEX_BASE_ZERO;
    type = ROCSPARSE_MATRIX_TYPE_GENERAL;
}

rocsparseMatDescr::~rocsparseMatDescr()
{
}



extern "C" rocsparseStatus_t rocsparseCreateMatDescr(rocsparseMatDescr_t *descrA)
{
    if (descrA == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else
    {
        // Allocate
        try
        {
            *descrA = new rocsparseMatDescr;
        }
        catch(rocsparseStatus_t status)
        {
            return status;
        }
        return ROCSPARSE_STATUS_SUCCESS;
    }
}

extern "C" rocsparseStatus_t rocsparseDestroyMatDescr(rocsparseMatDescr_t descrA)
{
    // Destruct
    try
    {
        delete descrA;
    }
    catch(rocsparseStatus_t status)
    {
        return status;
    }
    return ROCSPARSE_STATUS_SUCCESS;
}
