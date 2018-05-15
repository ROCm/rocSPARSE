/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "utility.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    // Device Query
    int device_id    = 0;
    int device_count = query_device_property();

    if(device_count <= device_id)
    {
        fprintf(stderr, "Error: invalid device ID. There may not be such device ID. Will exit\n");
        return -1;
    }
    else
    {
        set_device(device_id);
    }

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
