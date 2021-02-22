/*! \file */
/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_MATRIX_DENSE_HPP
#define ROCSPARSE_MATRIX_DENSE_HPP

#include "rocsparse_vector.hpp"

template <memory_mode::value_t MODE, typename T>
struct dense_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;
    rocsparse_int m{};
    rocsparse_int n{};
    rocsparse_int ld{};
    array_t<T>    val{};
    dense_matrix(){};
    ~dense_matrix(){};

    dense_matrix(rocsparse_int m_, rocsparse_int n_)
        : m(m_)
        , n(n_)
        , ld(m_)
        , val(m_ * n_){};

    void print() const
    {
        switch(MODE)
        {
        case memory_mode::host:
        case memory_mode::managed:
        {
            for(int i = 0; i < std::min(this->m, this->m); ++i)
            {
                for(int j = 0; j < std::min(this->n, this->n); ++j)
                {
                    std::cout << " " << this->val[j * this->ld + i];
                }
                std::cout << std::endl;
            }
            break;
        }
        case memory_mode::device:
        {
            std::cout << "DO NOT PRINT MATRIX FROM DEVICE" << std::endl;
            break;
        }
        }
    };

    operator T*()
    {
        return this->val;
    }

    operator const T*() const
    {
        return this->val;
    }

    dense_matrix(const dense_matrix<MODE, T>& that, bool transfer = true)
        : m(that.m)
        , n(that.n)
        , ld(that.m)
        , val(that.m * that.n)
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    dense_matrix(const dense_matrix<THAT_MODE, T>& that, bool transfer = true)
        : m(that.m)
        , n(that.n)
        , ld(that.m)
        , val(that.m * that.n)
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const dense_matrix<THAT_MODE, T>& that_)
    {
        CHECK_HIP_ERROR((this->m == that_.m && this->n == that_.n) ? hipSuccess
                                                                   : hipErrorInvalidValue);
        CHECK_HIP_ERROR((this->m == this->ld) ? hipSuccess : hipErrorInvalidValue);
        CHECK_HIP_ERROR((that_.m == that_.ld) ? hipSuccess : hipErrorInvalidValue);
        this->val.transfer_from(that_.val);
    }

    template <memory_mode::value_t THAT_MODE>
    void unit_check(const dense_matrix<THAT_MODE, T>& that_) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            dense_matrix<memory_mode::host, T> on_host(*this);
            on_host.unit_check(that_);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::managed:
            case memory_mode::host:
            {
                unit_check_general<rocsparse_int>(1, 1, 1, &this->m, &that_.m);
                unit_check_general<rocsparse_int>(1, 1, 1, &this->n, &that_.n);
                unit_check_general<rocsparse_int>(1, 1, 1, &this->ld, &that_.ld);
                unit_check_general<T>(this->m, this->n, this->ld, this->val, that_.val);
                break;
            }
            case memory_mode::device:
            {
                dense_matrix<memory_mode::host, T> that(that_);
                this->unit_check(that);
                break;
            }
            }
            break;
        }
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const dense_matrix<THAT_MODE, T>& that_,
                    floating_data_t<T>                tol = default_tolerance<T>::value) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            dense_matrix<memory_mode::host, T> on_host(*this);
            on_host.near_check(that_, tol);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::host:
            case memory_mode::managed:
            {
                unit_check_general<rocsparse_int>(1, 1, 1, &this->m, &that_.m);
                unit_check_general<rocsparse_int>(1, 1, 1, &this->n, &that_.n);
                unit_check_general<rocsparse_int>(1, 1, 1, &this->ld, &that_.ld);
                near_check_general<T>(this->m, this->n, this->ld, this->val, that_.val, tol);
                break;
            }
            case memory_mode::device:
            {
                dense_matrix<memory_mode::host, T> that(that_);
                this->near_check(that, tol);
                break;
            }
            }
            break;
        }
        }
    }
};

template <typename T>
using host_dense_matrix = dense_matrix<memory_mode::host, T>;

template <typename T>
using device_dense_matrix = dense_matrix<memory_mode::device, T>;

template <typename T>
using managed_dense_matrix = dense_matrix<memory_mode::managed, T>;

#endif // ROCSPARSE_MATRIX_DENSE_HPP
