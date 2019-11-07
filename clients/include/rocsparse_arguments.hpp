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
 *  \brief rocsparse_arguments.hpp provides a class to parse command arguments in both,
 *  clients and gtest. If class structure is changed, rocsparse_common.yaml must also be
 *  changed.
 */

#pragma once
#ifndef ROCSPARSE_ARGUMENTS_HPP
#define ROCSPARSE_ARGUMENTS_HPP

#include "rocsparse_datatype2string.hpp"
#include "rocsparse_math.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <rocsparse.h>

struct Arguments
{
    rocsparse_int M;
    rocsparse_int N;
    rocsparse_int K;
    rocsparse_int nnz;

    rocsparse_int dimx;
    rocsparse_int dimy;
    rocsparse_int dimz;

    rocsparse_datatype compute_type;

    double alpha;
    double alphai;
    double beta;
    double betai;

    rocsparse_operation       transA;
    rocsparse_operation       transB;
    rocsparse_index_base      baseA;
    rocsparse_index_base      baseB;
    rocsparse_index_base      baseC;
    rocsparse_index_base      baseD;
    rocsparse_action          action;
    rocsparse_hyb_partition   part;
    rocsparse_diag_type       diag;
    rocsparse_fill_mode       uplo;
    rocsparse_analysis_policy apol;
    rocsparse_solve_policy    spol;

    rocsparse_matrix_init matrix;

    rocsparse_int unit_check;
    rocsparse_int timing;
    rocsparse_int iters;

    uint32_t algo;

    char filename[64];
    char function[64];
    char name[64];
    char category[32];

    // Validate input format.
    // rocsparse_gentest.py is expected to conform to this format.
    // rocsparse_gentest.py uses rocsparse_common.yaml to generate this format.
    static void validate(std::istream& ifs)
    {
        auto error = [](auto name) {
            std::cerr << "Arguments field " << name << " does not match format.\n\n"
                      << "Fatal error: Binary test data does match input format.\n"
                         "Ensure that rocsparse_arguments.hpp and rocsparse_common.yaml\n"
                         "define exactly the same Arguments, that rocsparse_gentest.py\n"
                         "generates the data correctly, and that endianness is the same.\n";
            abort();
        };

        char      header[10]{}, trailer[10]{};
        Arguments arg{};
        ifs.read(header, sizeof(header));
        ifs >> arg;
        ifs.read(trailer, sizeof(trailer));

        if(strcmp(header, "rocSPARSE"))
            error("header");
        else if(strcmp(trailer, "ROCsparse"))
            error("trailer");

        auto check_func = [&, sig = (unsigned char)0](const auto& elem, auto name) mutable {
            static_assert(sizeof(elem) <= 255,
                          "One of the fields of Arguments is too large (> 255 bytes)");
            for(unsigned char i = 0; i < sizeof(elem); ++i)
                if(reinterpret_cast<const unsigned char*>(&elem)[i] ^ sig ^ i)
                    error(name);
            sig += 89;
        };

#define ROCSPARSE_FORMAT_CHECK(x) check_func(arg.x, #x)

        // Order is important
        ROCSPARSE_FORMAT_CHECK(M);
        ROCSPARSE_FORMAT_CHECK(N);
        ROCSPARSE_FORMAT_CHECK(K);
        ROCSPARSE_FORMAT_CHECK(nnz);
        ROCSPARSE_FORMAT_CHECK(dimx);
        ROCSPARSE_FORMAT_CHECK(dimy);
        ROCSPARSE_FORMAT_CHECK(dimz);
        ROCSPARSE_FORMAT_CHECK(compute_type);
        ROCSPARSE_FORMAT_CHECK(alpha);
        ROCSPARSE_FORMAT_CHECK(alphai);
        ROCSPARSE_FORMAT_CHECK(beta);
        ROCSPARSE_FORMAT_CHECK(betai);
        ROCSPARSE_FORMAT_CHECK(transA);
        ROCSPARSE_FORMAT_CHECK(transB);
        ROCSPARSE_FORMAT_CHECK(baseA);
        ROCSPARSE_FORMAT_CHECK(baseB);
        ROCSPARSE_FORMAT_CHECK(baseC);
        ROCSPARSE_FORMAT_CHECK(baseD);
        ROCSPARSE_FORMAT_CHECK(action);
        ROCSPARSE_FORMAT_CHECK(part);
        ROCSPARSE_FORMAT_CHECK(diag);
        ROCSPARSE_FORMAT_CHECK(uplo);
        ROCSPARSE_FORMAT_CHECK(apol);
        ROCSPARSE_FORMAT_CHECK(spol);
        ROCSPARSE_FORMAT_CHECK(matrix);
        ROCSPARSE_FORMAT_CHECK(unit_check);
        ROCSPARSE_FORMAT_CHECK(timing);
        ROCSPARSE_FORMAT_CHECK(iters);
        ROCSPARSE_FORMAT_CHECK(algo);
        ROCSPARSE_FORMAT_CHECK(filename);
        ROCSPARSE_FORMAT_CHECK(function);
        ROCSPARSE_FORMAT_CHECK(name);
        ROCSPARSE_FORMAT_CHECK(category);
    }

    template <typename T>
    T get_alpha() const
    {
        return (rocsparse_isnan(alpha) || rocsparse_isnan(alphai))
                   ? static_cast<T>(0)
                   : convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return (rocsparse_isnan(beta) || rocsparse_isnan(betai))
                   ? static_cast<T>(0)
                   : convert_alpha_beta<T>(beta, betai);
    }

private:
    template <typename T>
    static T convert_alpha_beta(double r, double i)
    {
        return static_cast<T>(r);
    }

    template <>
    rocsparse_float_complex convert_alpha_beta(double r, double i)
    {
        return rocsparse_float_complex(r, i);
    }

    template <>
    rocsparse_double_complex convert_alpha_beta(double r, double i)
    {
        return rocsparse_double_complex(r, i);
    }

    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg)
    {
        str.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return str;
    }

    // print_value is for formatting different data types

    // Default output
    template <typename T>
    static void print_value(std::ostream& str, const T& x)
    {
        str << x;
    }

    // Floating-point output
    static void print_value(std::ostream& str, double x)
    {
        if(std::isnan(x))
            str << ".nan";
        else if(std::isinf(x))
            str << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            str << s;
        }
    }

    // Character output
    static void print_value(std::ostream& str, char c)
    {
        char s[]{c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& str, bool b)
    {
        str << (b ? "true" : "false");
    }

    // string output
    static void print_value(std::ostream& str, const char* s)
    {
        str << std::quoted(s);
    }

    // Function to print Arguments out to stream in YAML format
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{'](const char* name, auto x) mutable {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

        print("function", arg.function);
        print("compute_type", rocsparse_datatype2string(arg.compute_type));
        print("transA", rocsparse_operation2string(arg.transA));
        print("transB", rocsparse_operation2string(arg.transB));
        print("baseA", rocsparse_indexbase2string(arg.baseA));
        print("baseB", rocsparse_indexbase2string(arg.baseB));
        print("baseC", rocsparse_indexbase2string(arg.baseC));
        print("baseD", rocsparse_indexbase2string(arg.baseD));
        print("M", arg.M);
        print("N", arg.N);
        print("K", arg.K);
        print("nnz", arg.nnz);
        print("dim_x", arg.dimx);
        print("dim_y", arg.dimy);
        print("dim_z", arg.dimz);
        print("alpha", arg.alpha);
        print("alphai", arg.alphai);
        print("beta", arg.beta);
        print("betai", arg.betai);
        print("action", rocsparse_action2string(arg.action));
        print("part", rocsparse_partition2string(arg.part));
        print("diag", rocsparse_diagtype2string(arg.diag));
        print("uplo", rocsparse_fillmode2string(arg.uplo));
        print("analysis_policy", rocsparse_analysis2string(arg.apol));
        print("solve_policy", rocsparse_solve2string(arg.spol));
        print("matrix", rocsparse_matrix2string(arg.matrix));
        print("file", arg.filename);
        print("algo", arg.algo);
        print("name", arg.name);
        print("category", arg.category);
        print("unit_check", arg.unit_check);
        print("timing", arg.timing);
        print("iters", arg.iters);

        return str << " }\n";
    }
};

static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<Arguments>{},
              "Arguments is not a trivial type, and thus is "
              "incompatible with C.");

#endif // ROCSPARSE_ARGUMENTS_HPP
