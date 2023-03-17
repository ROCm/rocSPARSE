/*! \file */
/* ************************************************************************
* Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_data.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_test_template_traits.hpp"

//
// INTERNAL MACRO TO SPECIALIZE TEST CALL NEEDED TO INSTANTIATE
//
#define SPECIALIZE_ROCSPARSE_TEST_CALL(ROUTINE)                                              \
    /**/ template <> /**/ struct rocsparse_test_call<rocsparse_test_enum::ROUTINE>           \
    /**/ {                                                                                   \
        /**/ template <typename... P> /**/ static void testing_bad_arg(const Arguments& arg) \
        /**/ {                                                                               \
            /**/ testing_##ROUTINE##_bad_arg<P...>(arg);                                     \
      /**/       }                                                                           \
        /**/ static void testing_extra(const Arguments& arg)                                 \
        /**/ {                                                                               \
            /**/ testing_##ROUTINE##_extra(arg);                                             \
      /**/       }                                                                           \
        /**/                                                                                 \
        /**/ template <typename... P> /**/ static void testing(const Arguments& arg)         \
        /**/ {                                                                               \
            /**/ try                                                                         \
            {                                                                                \
                testing_##ROUTINE<P...>(arg);                                                \
            }                                                                                \
            /**/ catch(rocsparse_status & status)                                            \
            {                                                                                \
                CHECK_ROCSPARSE_ERROR(status);                                               \
            }                                                                                \
            /**/ catch(hipError_t & error)                                                   \
            {                                                                                \
                CHECK_HIP_ERROR(error);                                                      \
            }                                                                                \
            /**/ catch(std::exception & error)                                               \
            {                                                                                \
                CHECK_ROCSPARSE_ERROR(rocsparse_status_thrown_exception);                    \
            }                                                                                \
      /**/       }                                                                           \
    /**/   }

/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// INTERNAL MACRO TO SPECIALIZE TEST FUNCTOR NEEDED TO INSTANTIATE
//
#define SPECIALIZE_ROCSPARSE_TEST_FUNCTORS(ROUTINE, ...)                               \
    /**/ template <> /**/ struct rocsparse_test_functors<rocsparse_test_enum::ROUTINE> \
    /**/ {                                                                             \
        /**/ static std::string name_suffix(const Arguments& arg)                      \
        /**/ {                                                                         \
            /**/ std::ostringstream s;                                                 \
            /**/ rocsparse_test_name_suffix_generator(s, __VA_ARGS__);                 \
            /**/ return s.str();                                                       \
      /**/       }                                                                     \
    /**/ }
/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// INTERNAL MACRO TO SPECIALIZE TEST TRAITS NEEDED TO INSTANTIATE
//
#define SPECIALIZE_ROCSPARSE_TEST_TRAITS(ROUTINE, CONFIG)                                     \
    /**/ template <> /**/ struct rocsparse_test_traits<rocsparse_test_enum::ROUTINE> : CONFIG \
    /**/ {                                                                                    \
  /**/ }
/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// INSTANTIATE TESTS
//

template <rocsparse_test_enum::value_type ROUTINE>
using test_template_traits_t
    = rocsparse_test_template_traits<ROUTINE, rocsparse_test_traits<ROUTINE>::s_dispatch>;

template <rocsparse_test_enum::value_type ROUTINE>
using test_dispatch_t = rocsparse_test_dispatch<rocsparse_test_traits<ROUTINE>::s_dispatch>;

#define INSTANTIATE_ROCSPARSE_TEST(ROUTINE, CATEGORY)                                          \
    /**/ using ROUTINE = test_template_traits_t<rocsparse_test_enum::ROUTINE>::filter;         \
    /**/                                                                                       \
    /**/ template <typename... P>                                                              \
    /**/ using ROUTINE##_call                                                                  \
        = test_template_traits_t<rocsparse_test_enum::ROUTINE>::caller<P...>;                  \
    /**/                                                                                       \
    /**/ TEST_P(ROUTINE, CATEGORY)                                                             \
    /**/ {                                                                                     \
        /**/ test_dispatch_t<rocsparse_test_enum::ROUTINE>::template dispatch<ROUTINE##_call>( \
            GetParam());                                                                       \
    /**/ }                                                                                     \
    /**/                                                                                       \
    /**/ INSTANTIATE_TEST_CATEGORIES(ROUTINE)

/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// DEFINE ALL REQUIRED INFORMATION FOR A TEST ROUTINE BUT WITH A PREDEFINED CONFIGURATION
// (i.e. [T (default) | <I,T> | <I,J,T>] + a selection of numeric types (all (default), real_only, complex_only, some other specific situations (?) ) )
//
#define TEST_ROUTINE_WITH_CONFIG(ROUTINE, CATEGORY, CONFIG, ...)   \
    /**/                                                           \
    /**/ SPECIALIZE_ROCSPARSE_TEST_TRAITS(ROUTINE, CONFIG);        \
    /**/ SPECIALIZE_ROCSPARSE_TEST_CALL(ROUTINE);                  \
    /**/ SPECIALIZE_ROCSPARSE_TEST_FUNCTORS(ROUTINE, __VA_ARGS__); \
    /**/ namespace                                                 \
    /**/ {                                                         \
        /**/ INSTANTIATE_ROCSPARSE_TEST(ROUTINE, CATEGORY);        \
    /**/ }

//
// DEFINE ALL REQUIRED INFORMATION FOR A TEST ROUTINE WITH A DEFAULT CONFIGURATION (i.e  T + all numeric types)
//
#define TEST_ROUTINE(ROUTINE, CATEGORY, ...) \
    TEST_ROUTINE_WITH_CONFIG(ROUTINE, CATEGORY, rocsparse_test_config, __VA_ARGS__)
/////////////////////////////////////////////////////////////////////////////////////////////////////
