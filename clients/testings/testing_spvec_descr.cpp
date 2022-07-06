/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename I, typename T>
void testing_spvec_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_spvec_descr x{};
    int64_t               size     = safe_size;
    int64_t               nnz      = safe_size;
    void*                 val_data = (void*)0x4;
    void*                 idx_data = (void*)0x4;
    rocsparse_index_base  base     = rocsparse_index_base_zero;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

#define PARAMS_CREATE &x, size, nnz, idx_data, val_data, itype, base, ttype
    auto_testing_bad_arg(rocsparse_create_spvec_descr, PARAMS_CREATE);
#undef PARAMS_CREATE

    // rocsparse_destroy_spvec_descr
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, 0, 0, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, size, 0, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);

    // Create valid descriptor
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, size, nnz, idx_data, val_data, itype, base, ttype),
        rocsparse_status_success);

    void** val_data_ptr = (void**)0x4;
    void** idx_data_ptr = (void**)0x4;

#define PARAMS_GET x, &size, &nnz, idx_data_ptr, val_data_ptr, &itype, &base, &ttype
    auto_testing_bad_arg(rocsparse_spvec_get, PARAMS_GET);
#undef PARAMS_GET

    // rocsparse_spvec_get_index_base
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_index_base(nullptr, &base),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_index_base(x, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spvec_get_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_values(nullptr, val_data_ptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_values(x, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spvec_set_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_set_values(nullptr, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_set_values(x, nullptr),
                            rocsparse_status_invalid_pointer);

    // Destroy valid descriptors
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);
}

template <typename I, typename T>
void testing_spvec_descr(const Arguments& arg)
{
}

#define INSTANTIATE(ITYPE, TTYPE)                                                  \
    template void testing_spvec_descr_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spvec_descr<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
