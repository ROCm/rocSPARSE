/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "context.h"
#include "rocsparse.h"



rocsparseContext::rocsparseContext()
{
    // Default is system stream
    stream = 0;
    // Default pointer mode is host
    pointer_mode = ROCSPARSE_POINTER_MODE_HOST;

}

rocsparseContext::~rocsparseContext()
{
}



extern "C" rocsparseStatus_t rocsparseCreate(rocsparseHandle_t *handle)
{
    // Check if handle is valid
    if (handle == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else
    {
        // Allocate
        try
        {
            *handle = new rocsparseContext;
        }
        catch(rocsparseStatus_t status)
        {
            return status;
        }
        return ROCSPARSE_STATUS_SUCCESS;
    }
}

extern "C" rocsparseStatus_t rocsparseDestroy(rocsparseHandle_t handle)
{
    // Destruct
    try
    {
        delete handle;
    }
    catch(rocsparseStatus_t status)
    {
        return status;
    }
    return ROCSPARSE_STATUS_SUCCESS;
}
