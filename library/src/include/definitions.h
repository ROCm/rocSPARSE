/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "status.h"

/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/

#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                           \
    {                                                                         \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;             \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                \
        {                                                                     \
            return get_rocsparse_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                     \
    }

#define RETURN_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                   \
        rocsparse_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)            \
        {                                                               \
            return TMP_STATUS_FOR_CHECK;                                \
        }                                                               \
    }

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                           \
    {                                                                        \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;            \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                               \
        {                                                                    \
            throw get_rocsparse_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                    \
    }

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

#endif // DEFINITIONS_H
