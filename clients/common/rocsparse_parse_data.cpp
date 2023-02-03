/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_parse_data.hpp"
#include "rocsparse_data.hpp"
#include "utility.hpp"

#include <fcntl.h>
#include <sys/types.h>

#ifdef WIN32

#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "rocsparse_clients_envariables.hpp"

// Parse YAML data
static std::string rocsparse_parse_yaml(const std::string& yaml)
{
#ifdef WIN32
    // Generate "/tmp/rocsparse-XXXXXX" like file name
    const std::string alphanum     = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv";
    int               stringlength = alphanum.length() - 1;
    std::string       uniquestr    = "rocsparse-";

    for(int n = 0; n <= 5; ++n)
    {
        uniquestr += alphanum.at(rand() % stringlength);
    }

    fs::path tmpname = fs::temp_directory_path() / uniquestr;

    auto exepath = rocsparse_exepath();
    auto cmd     = exepath + "rocsparse_gentest.py --template " + exepath
               + "rocsparse_template.yaml -o " + tmpname.string() + " " + yaml;
    std::cerr << cmd << std::endl;

    int status = std::system(cmd.c_str());
    if(status != 0)
    {
        perror("system cmd failed");
        exit(EXIT_FAILURE);
    }

    return tmpname.string(); // results to be read and removed later
#else

    char tmp[] = "/tmp/rocsparse-XXXXXX";
    int  fd    = mkostemp(tmp, O_CLOEXEC);
    if(fd == -1)
    {
        perror("Cannot open temporary file");
        exit(EXIT_FAILURE);
    }
    auto exepath = rocsparse_exepath();
    auto cmd     = exepath + "rocsparse_gentest.py --template " + exepath
               + "rocsparse_template.yaml -o " + tmp + " " + yaml;
    std::cerr << cmd << std::endl;
    int status = system(cmd.c_str());
    if(status == -1 || !WIFEXITED(status) || WEXITSTATUS(status))
        exit(EXIT_FAILURE);
    return tmp;

#endif
}

// Parse --data and --yaml command-line arguments
bool rocsparse_parse_data(int& argc, char** argv, const std::string& default_file)
{
    std::string filename;
    char**      argv_p = argv + 1;
    bool        help = false, yaml = false;
#ifdef ROCSPARSE_WITH_MEMSTAT
    const char* memory_report_filename = nullptr;
#endif
    // Scan, process and remove any --yaml or --data options
    for(int i = 1; argv[i]; ++i)
    {
        if(!strcmp(argv[i], "--data") || (yaml |= !strcmp(argv[i], "--yaml")))
        {
            if(filename != "")
            {
                std::cerr << "Only one of the --yaml and --data options may be specified"
                          << std::endl;
                exit(EXIT_FAILURE);
            }

            if(!argv[i + 1] || !argv[i + 1][0])
            {
                std::cerr << "The " << argv[i] << " option requires an argument" << std::endl;
                exit(EXIT_FAILURE);
            }
            filename = argv[++i];
        }
#ifdef ROCSPARSE_WITH_MEMSTAT
        else if(!strcmp(argv[i], "--memstat-report"))
        {
            if(!argv[i + 1] || !argv[i + 1][0])
            {
                std::cerr << "The " << argv[i] << " option requires an argument" << std::endl;
                exit(EXIT_FAILURE);
            }
            memory_report_filename = argv[++i];
        }
#endif
        else
        {
            *argv_p++ = argv[i];
            if(!help && (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")))
            {
                help = true;
                std::cout << "\n"
                          << argv[0] << " [ --data <path> | --yaml <path> ] <options> ...\n"
                          << std::endl;

                std::cout << "" << std::endl;
                std::cout << "Specific environment variables:" << std::endl;
                for(const auto v : rocsparse_clients_envariables::s_var_bool_all)
                {
                    std::cout << rocsparse_clients_envariables::get_name(v) << " "
                              << rocsparse_clients_envariables::get_description(v) << std::endl;
                }
                for(const auto v : rocsparse_clients_envariables::s_var_string_all)
                {
                    std::cout << rocsparse_clients_envariables::get_name(v) << " "
                              << rocsparse_clients_envariables::get_description(v) << std::endl;
                }
                std::cout << "" << std::endl;
            }
        }
    }

#ifdef ROCSPARSE_WITH_MEMSTAT
    rocsparse_status status = rocsparse_memstat_report(
        memory_report_filename ? memory_report_filename : "rocsparse_test_memstat.json");
    if(status != rocsparse_status_success)
    {
        std::cerr << "rocsparse_memstat_report failed " << std::endl;
        exit(EXIT_FAILURE);
    }
#endif

    // argc and argv contain remaining options and non-option arguments
    *argv_p = nullptr;
    argc    = argv_p - argv;

    if(filename == "-")
        filename = "/dev/stdin";
    else if(filename == "")
        filename = default_file;

    if(yaml)
        filename = rocsparse_parse_yaml(filename);

    if(filename != "")
    {
        RocSPARSE_TestData::set_filename(filename, yaml);
        return true;
    }

    return false;
}
