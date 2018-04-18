/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "context.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime_api.h>

/********************************************************************************
 * \brief rocsparseHandle_t is a structure holding the rocsparse library context.
 * It must be initialized using rocsparseCreate()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparseDestroy().
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseCreate(rocsparseHandle_t *handle)
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
            log_trace(*handle, "rocsparseCreate");

        }
        catch(rocsparseStatus_t status)
        {
            return status;
        }
        return ROCSPARSE_STATUS_SUCCESS;
    }
}

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseDestroy(rocsparseHandle_t handle)
{
    log_trace(handle, "rocsparseDestroy");
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

/********************************************************************************
 * \brief Indicates whether the scalar value pointers are on the host or device.
 * Set pointer mode, can be host or device
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseSetPointerMode(rocsparseHandle_t handle,
                                          rocsparsePointerMode_t mode)
{
    // Check if handle is valid
    if (handle == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    handle->pointer_mode = mode;
    log_trace(handle, "rocsparseSetPointerMode", mode);
    return ROCSPARSE_STATUS_SUCCESS;
}

/********************************************************************************
 * \brief Get pointer mode, can be host or device.
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseGetPointerMode(rocsparseHandle_t handle,
                                          rocsparsePointerMode_t *mode)
{
    // Check if handle is valid
    if (handle == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    *mode = handle->pointer_mode;
    log_trace(handle, "rocsparseGetPointerMode", *mode);
    return ROCSPARSE_STATUS_SUCCESS;
}

/********************************************************************************
 *! \brief Set rocsparse stream used for all subsequent library function calls.
 * If not set, all hip kernels will take the default NULL stream.
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseSetStream(rocsparseHandle_t handle,
                                     hipStream_t streamId)
{
    // Check if handle is valid
    if (handle == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    log_trace(handle, "rocsparseSetStream", streamId);
    return handle->setStream(streamId);
}

/********************************************************************************
 *! \brief Get rocsparse stream used for all subsequent library function calls.
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseGetStream(rocsparseHandle_t handle,
                                     hipStream_t *streamId)
{
    // Check if handle is valid
    if (handle == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    log_trace(handle, "rocsparseGetStream", *streamId);
    return handle->getStream(streamId);
}

/********************************************************************************
 * \brief Get rocSPARSE version
 * version % 100        = patch level
 * version / 100 % 1000 = minor version
 * version / 100000     = major version
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseGetVersion(rocsparseHandle_t handle, int *version)
{
    // Check if handle is valid
    if (handle == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    *version = ROCSPARSE_VERSION_MAJOR * 100000
             + ROCSPARSE_VERSION_MINOR * 100
             + ROCSPARSE_VERSION_PATCH;
    log_trace(handle, "rocsparseGetVersion", *version);
    return ROCSPARSE_STATUS_SUCCESS;
}
