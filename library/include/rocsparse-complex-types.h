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

/*! \file
 *  \brief rocsparse-complex-types.h defines complex data types used in rocsparse
 */

#pragma once
#ifndef _ROCSPARSE_COMPLEX_TYPES_H_
#define _ROCSPARSE_COMPLEX_TYPES_H_

#if __cplusplus < 201402L || (!defined(__HCC__) && !defined(__HIPCC__))

/* If this is a C compiler, C++ compiler below C++14, or a host-only compiler, only
   include minimal definitions of rocsparse_float_complex and rocsparse_double_complex */

typedef struct
{
    float x, y;
} rocsparse_float_complex;

typedef struct
{
    double x, y;
} rocsparse_double_complex;

#else /* __cplusplus < 201402L || (!defined(__HCC__) && !defined(__HIPCC__)) */

// If this is a full internal build, add full support of complex arithmetic and classes
// including __host__ and __device__ and such we need to use <hip/hip_runtime.h>.

#include <cmath>
#include <complex>
#include <hip/hip_runtime.h>
#include <ostream>
#include <sstream>

template <typename T>
class rocsparse_complex_num
{
public:
    __device__ __host__ rocsparse_complex_num(void)                         = default;
    __device__ __host__ rocsparse_complex_num(const rocsparse_complex_num&) = default;
    __device__ __host__ rocsparse_complex_num(rocsparse_complex_num&&)      = default;
    __device__ __host__ rocsparse_complex_num& operator=(const rocsparse_complex_num& rhs)
        = default;
    __device__ __host__ rocsparse_complex_num& operator=(rocsparse_complex_num&& rhs) = default;

    __device__ __host__ ~rocsparse_complex_num(void) = default;

    // Constructors
    __device__ __host__ rocsparse_complex_num(T r, T i)
        : x(r)
        , y(i)
    {
    }

    __device__ __host__ rocsparse_complex_num(T r)
        : x(r)
        , y(static_cast<T>(0))
    {
    }

    // Conversion from std::complex<T>
    __device__ __host__ rocsparse_complex_num(const std::complex<T>& z)
        : x(reinterpret_cast<T (&)[2]>(z)[0])
        , y(reinterpret_cast<T (&)[2]>(z)[1])
    {
    }

    // Conversion to std::complex<T>
    __device__ __host__ operator std::complex<T>() const
    {
        return {x, y};
    }

    // Accessors
    friend __device__ __host__ T std::real(const rocsparse_complex_num& z);
    friend __device__ __host__ T std::imag(const rocsparse_complex_num& z);

    // Stream output
    friend auto& operator<<(std::ostream& out, const rocsparse_complex_num& z)
    {
        std::stringstream ss;
        ss << '(' << z.x << ',' << z.y << ')';
        return out << ss.str();
    }

    // complex fma
    friend __device__ rocsparse_complex_num fma(rocsparse_complex_num p,
                                                rocsparse_complex_num q,
                                                rocsparse_complex_num r)
    {
        T real = fma(-p.y, q.y, fma(p.x, q.x, r.x));
        T imag = fma(p.x, q.y, fma(p.y, q.x, r.y));

        return {real, imag};
    }

    friend __host__ rocsparse_complex_num fma(rocsparse_complex_num p,
                                              rocsparse_complex_num q,
                                              rocsparse_complex_num r)
    {
        T real = std::fma(-p.y, q.y, std::fma(p.x, q.x, r.x));
        T imag = std::fma(p.x, q.y, std::fma(p.y, q.x, r.y));

        return {real, imag};
    }

    // complex conjugate
    friend __device__ __host__ rocsparse_complex_num conj(const rocsparse_complex_num& z)
    {
        return {z.x, -z.y};
    }

    // Unary operations
    __device__ __host__ rocsparse_complex_num operator-() const
    {
        return {-x, -y};
    }

    // In-place complex-complex operations
    __device__ __host__ auto& operator*=(const rocsparse_complex_num& rhs)
    {
        T real = x * rhs.x - y * rhs.y;
        T imag = y * rhs.x + x * rhs.y;

        return *this = {real, imag};
    }

    __device__ __host__ auto& operator+=(const rocsparse_complex_num& rhs)
    {
        return *this = {x + rhs.x, y + rhs.y};
    }

    __device__ __host__ auto& operator-=(const rocsparse_complex_num& rhs)
    {
        return *this = {x - rhs.x, y - rhs.y};
    }

    __device__ __host__ auto& operator/=(const rocsparse_complex_num& rhs)
    {
        if(abs(rhs.x) > abs(rhs.y))
        {
            T ratio = rhs.y / rhs.x;
            T scale = static_cast<T>(1) / (rhs.x + rhs.y * ratio);
            *this   = {(x + y * ratio) * scale, (y - x * ratio) * scale};
        }
        else
        {
            T ratio = rhs.x / rhs.y;
            T scale = static_cast<T>(1) / (rhs.x * ratio + rhs.y);
            *this   = {(y + x * ratio) * scale, (y * ratio - x) * scale};
        }

        return *this;
    }

    // Out-of-place complex-complex operations
    __device__ __host__ auto operator+(const rocsparse_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs += rhs;
    }

    __device__ __host__ auto operator-(const rocsparse_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs -= rhs;
    }

    __device__ __host__ auto operator*(const rocsparse_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs *= rhs;
    }

    __device__ __host__ auto operator/(const rocsparse_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs /= rhs;
    }

    __device__ __host__ bool operator==(const rocsparse_complex_num& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }

    __device__ __host__ bool operator!=(const rocsparse_complex_num& rhs) const
    {
        return !(*this == rhs);
    }

private:
    // Internal real absolute function, to be sure we're on both device and host
    static __forceinline__ __device__ __host__ T abs(T x)
    {
        return x < 0 ? -x : x;
    }

    T x;
    T y;
};

// Inject standard functions into namespace std
namespace std
{
    template <typename T>
    __device__ __host__ inline T real(const rocsparse_complex_num<T>& z)
    {
        return z.x;
    }

    template <typename T>
    __device__ __host__ inline T imag(const rocsparse_complex_num<T>& z)
    {
        return z.y;
    }
}

// Test for C compatibility
template <typename T>
class rocsparse_complex_num_check
{
    static_assert(
        std::is_standard_layout<rocsparse_complex_num<T>>{},
        "rocsparse_complex_num<T> is not a standard layout type, and thus is incompatible with C.");
    static_assert(
        std::is_trivial<rocsparse_complex_num<T>>{},
        "rocsparse_complex_num<T> is not a trivial type, and thus is incompatible with C.");
    static_assert(
        sizeof(rocsparse_complex_num<T>) == 2 * sizeof(T),
        "rocsparse_complex_num<T> is not the correct size, and thus is incompatible with C.");
};

template class rocsparse_complex_num_check<float>;
template class rocsparse_complex_num_check<double>;

// rocSPARSE complex data types
using rocsparse_float_complex  = rocsparse_complex_num<float>;
using rocsparse_double_complex = rocsparse_complex_num<double>;

#endif /* __cplusplus < 201402L || (!defined(__HCC__) && !defined(__HIPCC__)) */

#endif /* _ROCSPARSE_COMPLEX_TYPES_H_ */
