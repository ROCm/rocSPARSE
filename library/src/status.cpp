/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "status.h"
#include "rocsparse.h"

#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * \brief convert hipError_t to rocsparse_status
 * TODO - enumerate library calls to hip runtime, enumerate possible errors from those calls
 ******************************************************************************/
rocsparse_status get_rocsparse_status_for_hip_status(hipError_t status)
{
    switch(status)
    {
        // success
        case hipSuccess:
            return rocsparse_status_success;

        // internal hip memory allocation
        case hipErrorMemoryAllocation:
        case hipErrorLaunchOutOfResources:
            return rocsparse_status_memory_error;

        // user-allocated hip memory
        case hipErrorInvalidDevicePointer: // hip memory
            return rocsparse_status_invalid_pointer;

        // user-allocated device, stream, event
        case hipErrorInvalidDevice:
        case hipErrorInvalidResourceHandle:
            return rocsparse_status_invalid_handle;

        // library using hip incorrectly
        case hipErrorInvalidValue:
            return rocsparse_status_internal_error;

        // hip runtime failing
        case hipErrorNoDevice: // no hip devices
        case hipErrorUnknown:
        default: return rocsparse_status_internal_error;
    }
}
