/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_IMPORTER_FORMAT_T_HPP
#define ROCSPARSE_IMPORTER_FORMAT_T_HPP

struct rocsparse_importer_format_t
{
#define LIST_IMPORTER_FORMATS \
    FORMAT(unknown)           \
    FORMAT(matrixmarket)      \
    FORMAT(rocalution)        \
    FORMAT(rocsparseio)

    typedef enum _
    {
#define FORMAT(x_) x_,
        LIST_IMPORTER_FORMATS
    } value_type;
    static constexpr value_type all_formats[] = {LIST_IMPORTER_FORMATS};
#undef FORMAT

#define FORMAT(x_) #x_,
    static constexpr const char* s_format_names[]{LIST_IMPORTER_FORMATS};
#undef FORMAT

    value_type value{};
    rocsparse_importer_format_t(){};

public:
    static const char* extension(const value_type val)
    {
        switch(val)
        {
        case matrixmarket:
            return ".mtx";
        case rocalution:
            return ".csr";
        case rocsparseio:
            return ".bin";
        case unknown:
            return "";
        }
        return nullptr;
    }

    rocsparse_importer_format_t& operator()(const char* filename)
    {
        this->value = unknown;

        const char* ext = nullptr;
        for(const char* p = filename; *p != '\0'; ++p)
        {
            if(*p == '.')
                ext = p;
        }
        if(ext)
        {
            for(auto format : all_formats)
            {
                if(!strcmp(ext, extension(format)))
                {
                    this->value = format;
                    break;
                }
            }
        }
        return *this;
    };
};

#endif // HEADER
