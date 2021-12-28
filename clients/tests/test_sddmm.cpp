/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_data.hpp"
#include "rocsparse_datatype2string.hpp"
#include "rocsparse_test.hpp"
#include "testing_sddmm.hpp"
#include "type_dispatch.hpp"

#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if below.
    template <typename T, typename I = int32_t, typename J = int32_t, typename = void>
    struct sddmm_testing : rocsparse_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename I, typename J, typename T>
    struct sddmm_testing<
        I,
        J,
        T,
        typename std::enable_if<std::is_same<T, float>() || std::is_same<T, double>()
                                || std::is_same<T, rocsparse_float_complex>()
                                || std::is_same<T, rocsparse_double_complex>()>::type>
    {
        explicit operator bool()
        {
            return true;
        }
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "sddmm"))
                testing_sddmm<I, J, T>(arg);
            else if(!strcmp(arg.function, "sddmm_bad_arg"))
                testing_sddmm_bad_arg<I, J, T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct sddmm : RocSPARSE_Test<sddmm, sddmm_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocsparse_ijt_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "sddmm") || !strcmp(arg.function, "sddmm_bad_arg");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            if(rocsparse_arguments_has_datafile(arg))
            {
                return RocSPARSE_TestName<sddmm>()
                       << rocsparse_indextype2string(arg.index_type_I) << '_'
                       << rocsparse_indextype2string(arg.index_type_J) << '_'
                       << rocsparse_datatype2string(arg.compute_type) << '_' << arg.alpha << '_'
                       << arg.alphai << '_' << arg.beta << '_' << arg.betai << '_'
                       << rocsparse_operation2string(arg.transA) << '_'
                       << rocsparse_operation2string(arg.transB) << '_'
                       << rocsparse_indexbase2string(arg.baseA) << '_'
                       << rocsparse_order2string(arg.order) << '_'
                       << rocsparse_format2string(arg.format) << '_'
                       << rocsparse_matrix2string(arg.matrix) << '_'
                       << rocsparse_filename2string(arg.filename);
            }
            else
            {
                return RocSPARSE_TestName<sddmm>()
                       << rocsparse_indextype2string(arg.index_type_I) << '_'
                       << rocsparse_indextype2string(arg.index_type_J) << '_'
                       << rocsparse_datatype2string(arg.compute_type) << '_' << arg.M << '_'
                       << arg.N << '_' << arg.K << '_' << arg.alpha << '_' << arg.alphai << '_'
                       << arg.beta << '_' << arg.betai << '_'
                       << rocsparse_operation2string(arg.transA) << '_'
                       << rocsparse_operation2string(arg.transB) << '_'
                       << rocsparse_indexbase2string(arg.baseA) << '_'
                       << rocsparse_order2string(arg.order) << '_'
                       << rocsparse_format2string(arg.format) << '_'
                       << rocsparse_matrix2string(arg.matrix);
            }
        }
    };

    TEST_P(sddmm, level3)
    {
        rocsparse_ijt_dispatch<sddmm_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(sddmm);

} // namespace
