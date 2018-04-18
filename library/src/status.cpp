/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "status.h"
#include "rocsparse.h"

#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 * TODO - enumerate library calls to hip runtime, enumerate possible errors from those calls
 ******************************************************************************/
rocsparseStatus_t get_rocsparse_status_for_hip_status(hipError_t status)
{
    switch(status)
    {
        // success
        case hipSuccess:
            return ROCSPARSE_STATUS_SUCCESS;

        // internal hip memory allocation
        case hipErrorMemoryAllocation:
        case hipErrorLaunchOutOfResources:
            return ROCSPARSE_STATUS_MEMORY_ERROR;

        // user-allocated hip memory
        case hipErrorInvalidDevicePointer: // hip memory
            return ROCSPARSE_STATUS_INVALID_POINTER;

        // user-allocated device, stream, event
        case hipErrorInvalidDevice:
        case hipErrorInvalidResourceHandle:
            return ROCSPARSE_STATUS_INVALID_HANDLE;

        // library using hip incorrectly
        case hipErrorInvalidValue:
            return ROCSPARSE_STATUS_INTERNAL_ERROR;

        // hip runtime failing
        case hipErrorNoDevice: // no hip devices
        case hipErrorUnknown:
        default: return ROCSPARSE_STATUS_INTERNAL_ERROR;
    }
}
