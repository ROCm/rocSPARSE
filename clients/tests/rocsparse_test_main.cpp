/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "utility.hpp"

#include <gtest/gtest.h>

using namespace testing;

bool display_timing_info_is_stdout_disabled()
{
    return false;
}

rocsparse_status rocsparse_record_output(const std::string& s)
{
    return rocsparse_status_success;
}

rocsparse_status rocsparse_record_output_legend(const std::string& s)
{
    return rocsparse_status_success;
}

rocsparse_status rocsparse_record_timing(double msec, double gflops, double gbs)
{
    return rocsparse_status_success;
}

class ConfigurableEventListener : public TestEventListener
{
    TestEventListener* eventListener;

public:
    bool showTestCases; // Show the names of each test case.
    bool showTestNames; // Show the names of each test.
    bool showSuccesses; // Show each success.
    bool showInlineFailures; // Show each failure as it occurs.
    bool showEnvironment; // Show the setup of the global environment.

    explicit ConfigurableEventListener(TestEventListener* theEventListener)
        : eventListener(theEventListener)
        , showTestCases(true)
        , showTestNames(true)
        , showSuccesses(true)
        , showInlineFailures(true)
        , showEnvironment(true)
    {
    }

    ~ConfigurableEventListener() override
    {
        delete eventListener;
    }

    void OnTestProgramStart(const UnitTest& unit_test) override
    {
        eventListener->OnTestProgramStart(unit_test);
    }

    void OnTestIterationStart(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationStart(unit_test, iteration);
    }

    void OnEnvironmentsSetUpStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsSetUpStart(unit_test);
        }
    }

    void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsSetUpEnd(unit_test);
        }
    }

    void OnTestCaseStart(const TestCase& test_case) override
    {
        if(showTestCases)
        {
            eventListener->OnTestCaseStart(test_case);
        }
    }

    void OnTestStart(const TestInfo& test_info) override
    {
        if(showTestNames)
        {
            eventListener->OnTestStart(test_info);
        }
    }

    void OnTestPartResult(const TestPartResult& result) override
    {
        eventListener->OnTestPartResult(result);
    }

    void OnTestEnd(const TestInfo& test_info) override
    {
        if(test_info.result()->Failed() ? showInlineFailures : showSuccesses)
        {
            eventListener->OnTestEnd(test_info);
        }
    }

    void OnTestCaseEnd(const TestCase& test_case) override
    {
        if(showTestCases)
        {
            eventListener->OnTestCaseEnd(test_case);
        }
    }

    void OnEnvironmentsTearDownStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsTearDownStart(unit_test);
        }
    }

    void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsTearDownEnd(unit_test);
        }
    }

    void OnTestIterationEnd(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationEnd(unit_test, iteration);
    }

    void OnTestProgramEnd(const UnitTest& unit_test) override
    {
        eventListener->OnTestProgramEnd(unit_test);
    }
};

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    // Get user device id from command line
    int dev = 0;

    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "--device") == 0 && argc > i + 1)
        {
            dev = atoi(argv[i + 1]);
        }
    }

    // Device query
    int devs;
    if(hipGetDeviceCount(&devs) != hipSuccess)
    {
        std::cerr << "Error: cannot get device count" << std::endl;
        return -1;
    }

    std::cout << "Query device success: there are " << devs << " devices" << std::endl;

    for(int i = 0; i < devs; ++i)
    {
        hipDeviceProp_t prop;

        if(hipGetDeviceProperties(&prop, i) != hipSuccess)
        {
            std::cerr << "Error: cannot get device properties" << std::endl;
            return -1;
        }

        std::cout << "Device ID " << i << ": " << prop.name << std::endl;
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
        std::cout << "with " << (prop.totalGlobalMem >> 20) << "MB memory, clock rate "
                  << prop.clockRate / 1000 << "MHz @ computing capability " << prop.major << "."
                  << prop.minor << std::endl;
        std::cout << "maxGridDimX " << prop.maxGridSize[0] << ", sharedMemPerBlock "
                  << (prop.sharedMemPerBlock >> 10) << "KB, maxThreadsPerBlock "
                  << prop.maxThreadsPerBlock << std::endl;
        std::cout << "wavefrontSize " << prop.warpSize << std::endl;
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
    }

    // Set device
    if(hipSetDevice(dev) != hipSuccess || dev >= devs)
    {
        std::cerr << "Error: cannot set device ID " << dev << std::endl;
        return -1;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev);

    std::cout << "Using device ID " << dev << " (" << prop.name << ") for rocSPARSE" << std::endl;
    std::cout << "-------------------------------------------------------------------------"
              << std::endl;

    // Print version
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    int  ver;
    char rev[64];

    rocsparse_get_version(handle, &ver);
    rocsparse_get_git_rev(handle, rev);

    std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << "-" << rev << std::endl;

    rocsparse_destroy_handle(handle);

    // Set data file path
    rocsparse_parse_data(argc, argv, rocsparse_exepath() + "rocsparse_test.data");

    // Initialize google test
    InitGoogleTest(&argc, argv);

    // Free up all temporary data generated during test creation
    test_cleanup::cleanup();

    // Remove the default listener
    auto& listeners       = UnitTest::GetInstance()->listeners();
    auto  default_printer = listeners.Release(listeners.default_result_printer());

    // Add our listener, by default everything is on (the same as using the default listener)
    // Here turning everything off so only the 3 lines for the result are visible
    // (plus any failures at the end), like:

    // [==========] Running 149 tests from 53 test cases.
    // [==========] 149 tests from 53 test cases ran. (1 ms total)
    // [  PASSED  ] 149 tests.
    //
    auto listener       = new ConfigurableEventListener(default_printer);
    auto gtest_listener = getenv("GTEST_LISTENER");

    if(gtest_listener && !strcmp(gtest_listener, "NO_PASS_LINE_IN_LOG"))
    {
        listener->showTestNames = listener->showSuccesses = listener->showInlineFailures = false;
    }

    listeners.Append(listener);

    // Run all tests
    int ret = RUN_ALL_TESTS();

    // Reset HIP device
    hipDeviceReset();

    return ret;
}
