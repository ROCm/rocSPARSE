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
void testing_const_dnvec_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_const_dnvec_descr x{};
    int64_t                     size   = safe_size;
    const void*                 values = (void*)0x4;
    rocsparse_datatype          ttype  = get_datatype<T>();

#define PARAMS_CREATE &x, size, values, ttype
    auto_testing_bad_arg(rocsparse_create_const_dnvec_descr, PARAMS_CREATE);
#undef PARAMS_CREATE

    // rocsparse_destroy_dnvec_descr_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnvec_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_dnvec_descr(&x, 0, nullptr, ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnvec_descr(x), rocsparse_status_success);

    // Create valid descriptor
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_dnvec_descr(&x, size, values, ttype),
                            rocsparse_status_success);

    const void** values_ptr = (const void**)0x4;

#define PARAMS_GET x, &size, values_ptr, &ttype
    auto_testing_bad_arg(rocsparse_const_dnvec_get, PARAMS_GET);
#undef PARAMS_GET

    // rocsparse_dnvec_get_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_dnvec_get_values(nullptr, values_ptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_dnvec_get_values(x, nullptr),
                            rocsparse_status_invalid_pointer);

    // Destroy valid descriptor
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnvec_descr(x), rocsparse_status_success);
}

template <typename T>
void testing_const_dnvec_descr(const Arguments& arg)
{
    int64_t            m     = arg.M;
    rocsparse_datatype ttype = get_datatype<T>();

    if(m <= 0)
    {
        return;
    }

    device_vector<T> values(m);

    if(arg.unit_check)
    {
        rocsparse_const_dnvec_descr A{};

        // Create valid descriptor
        CHECK_ROCSPARSE_ERROR(
            rocsparse_create_const_dnvec_descr(&A, m, (const void*)values, ttype));

        int64_t            gold_m;
        rocsparse_datatype gold_ttype;
        const T*           gold_values;

        CHECK_ROCSPARSE_ERROR(
            rocsparse_const_dnvec_get(A, &gold_m, (const void**)&gold_values, &gold_ttype));

        unit_check_scalar<int64_t>(m, gold_m);
        unit_check_enum<rocsparse_datatype>(ttype, gold_ttype);

        ASSERT_EQ(values, (const T*)gold_values);

        gold_values = nullptr;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_dnvec_get_values(A, (const void**)&gold_values));

        ASSERT_EQ(values, (const T*)gold_values);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_dnvec_descr(A));
    }

    if(arg.timing)
    {
    }
}

#define INSTANTIATE(TTYPE)                                                        \
    template void testing_const_dnvec_descr_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_const_dnvec_descr<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_const_dnvec_descr_extra(const Arguments& arg) {}
