/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

template <typename T>
void testing_const_dnmat_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_const_dnmat_descr A{};
    int64_t                     rows   = safe_size;
    int64_t                     cols   = safe_size;
    int64_t                     ld     = safe_size;
    const void*                 values = (const void*)0x4;
    rocsparse_order             order  = rocsparse_order_column;
    rocsparse_datatype          ttype  = get_datatype<T>();

#define PARAMS_CREATE &A, rows, cols, ld, values, ttype, order
    auto_testing_bad_arg(rocsparse_create_const_dnmat_descr, PARAMS_CREATE);
#undef PARAMS_CREATE

    // rocsparse_destroy_dnmat_descr_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_dnmat_descr(&A, 0, cols, ld, nullptr, ttype, order),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_dnmat_descr(&A, rows, 0, ld, nullptr, ttype, order),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_dnmat_descr(&A, rows, cols, 0, nullptr, ttype, order),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);

    // Create valid descriptor
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_dnmat_descr(&A, rows, cols, ld, values, ttype, order),
        rocsparse_status_success);

    const void** values_ptr = (const void**)0x4;

#define PARAMS_GET A, &rows, &cols, &ld, values_ptr, &ttype, &order
    auto_testing_bad_arg(rocsparse_const_dnmat_get, PARAMS_GET);
#undef PARAMS_GET

    // rocsparse_dnmat_get_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_dnmat_get_values(nullptr, (const void**)&values),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_dnmat_get_values(A, nullptr),
                            rocsparse_status_invalid_pointer);

    int     batch_count  = safe_size;
    int64_t batch_stride = safe_size;

    // rocsparse_dnmat_get_strided_batch_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_get_strided_batch(nullptr, &batch_count, &batch_stride),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_get_strided_batch(A, nullptr, &batch_stride),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_get_strided_batch(A, &batch_count, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_get_strided_batch(A, nullptr, nullptr),
                            rocsparse_status_invalid_pointer);

    // Destroy valid descriptor
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
}

template <typename T>
void testing_const_dnmat_descr(const Arguments& arg)
{
    int64_t         m     = arg.M;
    int64_t         n     = arg.N;
    int64_t         ld    = arg.denseld;
    rocsparse_order order = arg.order;

    rocsparse_datatype ttype = get_datatype<T>();

    device_vector<T> values(ld * ((order == rocsparse_order_column) ? n : m));

    if(arg.unit_check)
    {
        rocsparse_const_dnmat_descr A{};

        // Create valid descriptor
        CHECK_ROCSPARSE_ERROR(
            rocsparse_create_const_dnmat_descr(&A, m, n, ld, (const void*)values, ttype, order));

        int64_t            gold_m;
        int64_t            gold_n;
        int64_t            gold_ld;
        rocsparse_order    gold_order;
        rocsparse_datatype gold_ttype;
        const T*           gold_values;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_dnmat_get(
            A, &gold_m, &gold_n, &gold_ld, (const void**)&gold_values, &gold_ttype, &gold_order));

        unit_check_scalar<int64_t>(m, gold_m);
        unit_check_scalar<int64_t>(n, gold_n);
        unit_check_scalar<int64_t>(ld, gold_ld);
        unit_check_enum<rocsparse_order>(order, gold_order);
        unit_check_enum<rocsparse_datatype>(ttype, gold_ttype);

        ASSERT_EQ(values, (const T*)gold_values);

        gold_values = nullptr;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_dnmat_get_values(A, (const void**)&gold_values));

        ASSERT_EQ(values, (const T*)gold_values);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_dnmat_descr(A));
    }

    if(arg.timing)
    {
    }
}

#define INSTANTIATE(TTYPE)                                                        \
    template void testing_const_dnmat_descr_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_const_dnmat_descr<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_const_dnmat_descr_extra(const Arguments& arg) {}
