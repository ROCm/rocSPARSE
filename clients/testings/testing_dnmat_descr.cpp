/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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

#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

template <typename T>
void testing_dnmat_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_dnmat_descr A{};
    int64_t               rows   = safe_size;
    int64_t               cols   = safe_size;
    int64_t               ld     = safe_size;
    void*                 values = (void*)0x4;
    rocsparse_order       order  = rocsparse_order_column;
    rocsparse_datatype    ttype  = get_datatype<T>();

#define PARAMS_CREATE &A, rows, cols, ld, values, ttype, order
    auto_testing_bad_arg(rocsparse_create_dnmat_descr, PARAMS_CREATE);
#undef PARAMS_CREATE

    // rocsparse_destroy_dnmat_descr
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, 0, cols, ld, nullptr, ttype, order),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, rows, 0, ld, nullptr, ttype, order),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, rows, cols, 0, nullptr, ttype, order),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);

    // Create valid descriptor
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, rows, cols, ld, values, ttype, order),
                            rocsparse_status_success);

    void** values_ptr = (void**)0x4;

#define PARAMS_GET A, &rows, &cols, &ld, values_ptr, &ttype, &order
    auto_testing_bad_arg(rocsparse_dnmat_get, PARAMS_GET);
#undef PARAMS_GET

    // rocsparse_dnmat_get_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_get_values(nullptr, (void**)&values),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_get_values(A, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_dnmat_set_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_values(nullptr, values),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_values(A, nullptr),
                            rocsparse_status_invalid_pointer);

    // Destroy valid descriptor
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
}

template <typename T>
void testing_dnmat_descr(const Arguments& arg)
{
}

#define INSTANTIATE(TTYPE)                                                  \
    template void testing_dnmat_descr_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_dnmat_descr<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
