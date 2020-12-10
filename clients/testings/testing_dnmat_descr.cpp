/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

void testing_dnmat_descr_bad_arg(const Arguments& arg)
{
    int64_t rows = 100;
    int64_t cols = 100;
    int64_t ld   = 100;

    rocsparse_datatype type  = rocsparse_datatype_f32_r;
    rocsparse_order    order = rocsparse_order_column;

    // Allocate memory on device
    device_vector<float> values(ld * cols);

    if(!values)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_dnmat_descr A;

    // rocsparse_create_dnmat_descr
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_dnmat_descr(nullptr, rows, cols, ld, values, type, order),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, -1, cols, ld, values, type, order),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, rows, -1, ld, values, type, order),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, rows, cols, -1, values, type, order),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, rows, cols, ld, nullptr, type, order),
                            rocsparse_status_invalid_pointer);

    // rocsparse_destroy_dnmat_descr
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, 0, cols, ld, nullptr, type, order),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, rows, 0, ld, nullptr, type, order),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, rows, cols, 0, nullptr, type, order),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);

    // Create valid descriptor
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&A, rows, cols, ld, values, type, order),
                            rocsparse_status_success);

    // rocsparse_dnmat_get
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_dnmat_get(nullptr, &rows, &cols, &ld, (void**)&values, &type, &order),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_dnmat_get(A, nullptr, &cols, &ld, (void**)&values, &type, &order),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_dnmat_get(A, &rows, nullptr, &ld, (void**)&values, &type, &order),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_dnmat_get(A, &rows, &cols, nullptr, (void**)&values, &type, &order),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_get(A, &rows, &cols, &ld, nullptr, &type, &order),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_dnmat_get(A, &rows, &cols, &ld, (void**)&values, nullptr, &order),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_dnmat_get(A, &rows, &cols, &ld, (void**)&values, &type, nullptr),
        rocsparse_status_invalid_pointer);

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
