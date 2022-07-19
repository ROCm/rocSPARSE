/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include <iostream>
static constexpr const char* s_rocsparse_debug_str = "//rocsparse.degug: ";

//
// Trace message..
//
#undef ROCSPARSE_DEBUG_VERBOSE
#define ROCSPARSE_DEBUG_VERBOSE(msg__)                                                          \
    do                                                                                          \
    {                                                                                           \
        std::cout << s_rocsparse_debug_str << std::endl;                                        \
        std::cout << s_rocsparse_debug_str << "verbose" << std::endl;                           \
        std::cout << s_rocsparse_debug_str << "  function : '" << __FUNCTION__ << "'"           \
                  << std::endl;                                                                 \
        std::cout << s_rocsparse_debug_str << "  file     : '" << __FILE__ << "'" << std::endl; \
        std::cout << s_rocsparse_debug_str << "  line     : " << __LINE__ << std::endl;         \
        std::cout << s_rocsparse_debug_str << "  message  : " << msg__ << std::endl;            \
        std::cout << s_rocsparse_debug_str << std::endl;                                        \
    } while(false)

inline rocsparse_status rocsparse_return_status_trace(const char*      function,
                                                      const char*      file,
                                                      const int        line,
                                                      rocsparse_status status)
{
    if(rocsparse_status_success != status)
    {
        std::cerr << s_rocsparse_debug_str << std::endl
                  << s_rocsparse_debug_str << "invalid status" << std::endl
                  << s_rocsparse_debug_str << "function       : '" << function << "'" << std::endl
                  << s_rocsparse_debug_str << "line           :  " << line << std::endl
                  << s_rocsparse_debug_str << "file           : '" << file << "'" << std::endl
                  << s_rocsparse_debug_str << std::endl;
    }
    return status;
}

#undef ROCSPARSE_RETURN_STATUS
#define ROCSPARSE_RETURN_STATUS(token__)  \
    return rocsparse_return_status_trace( \
        __FUNCTION__, __FILE__, __LINE__, rocsparse_status_##token__)
