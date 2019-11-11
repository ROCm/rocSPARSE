/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include "testing_csrsv.hpp"
#include "type_dispatch.hpp"

#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if below.
    template <typename, typename = void>
    struct csrsv_testing : rocsparse_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct csrsv_testing<
        T,
        typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}
                                || std::is_same<T, rocsparse_float_complex>{}
                                || std::is_same<T, rocsparse_double_complex>{}>::type>
    {
        explicit operator bool()
        {
            return true;
        }
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "csrsv"))
                testing_csrsv<T>(arg);
            else if(!strcmp(arg.function, "csrsv_bad_arg"))
                testing_csrsv_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct csrsv : RocSPARSE_Test<csrsv, csrsv_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocsparse_simple_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "csrsv") || !strcmp(arg.function, "csrsv_bad_arg");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            if(arg.matrix == rocsparse_matrix_file_rocalution
               || arg.matrix == rocsparse_matrix_file_mtx)
            {
                return RocSPARSE_TestName<csrsv>{}
                       << "DISABLED_" << rocsparse_datatype2string(arg.compute_type) << '_'
                       << arg.alpha << '_' << arg.alphai << '_'
                       << rocsparse_operation2string(arg.transA) << '_'
                       << rocsparse_diagtype2string(arg.diag) << '_'
                       << rocsparse_fillmode2string(arg.uplo) << '_'
                       << rocsparse_analysis2string(arg.apol) << '_'
                       << rocsparse_solve2string(arg.spol) << '_'
                       << rocsparse_indexbase2string(arg.baseA) << '_'
                       << rocsparse_matrix2string(arg.matrix) << '_' << arg.filename;
            }
            else
            {
                return RocSPARSE_TestName<csrsv>{}
                       << "DISABLED_" << rocsparse_datatype2string(arg.compute_type) << '_' << arg.M
                       << '_' << arg.alpha << '_' << arg.alphai << '_'
                       << rocsparse_operation2string(arg.transA) << '_'
                       << rocsparse_diagtype2string(arg.diag) << '_'
                       << rocsparse_fillmode2string(arg.uplo) << '_'
                       << rocsparse_analysis2string(arg.apol) << '_'
                       << rocsparse_solve2string(arg.spol) << '_'
                       << rocsparse_indexbase2string(arg.baseA) << '_'
                       << rocsparse_matrix2string(arg.matrix);
            }
        }
    };

    TEST_P(csrsv, level2)
    {
        rocsparse_simple_dispatch<csrsv_testing>(GetParam());
    }
    INSTANTIATE_TEST_CATEGORIES(csrsv);

} // namespace
