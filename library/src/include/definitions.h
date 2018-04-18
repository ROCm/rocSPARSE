/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_DEFINITIONS_H_
#define ROCSPARSE_DEFINITIONS_H_

#include "status.h"

/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                         \
    {                                                                      \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;          \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                             \
        {                                                                  \
            throw get_rocsparse_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                  \
    }

#endif // ROCSPARSE_DEFINITIONS_H_
