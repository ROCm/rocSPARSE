=====================================
Installation and Building for Windows
=====================================

-------------
Prerequisites
-------------

- An AMD HIP SDK enabled platform. More information can be found `here <https://docs.amd.com/>`_.
- rocSPARSE is supported on the same Windows versions and toolchains that are supported by the HIP SDK.
- As the AMD HIP SDK is new and quickly evolving it will have more up to date information regarding the SDK's internal contents. Thus it may overrule statements found in this section on installing and building for Windows.


----------------------------
Installing Prebuilt Packages
----------------------------

rocSPARSE can be installed on Windows 11 or Windows 10 using the AMD HIP SDK installer.

The simplest way to use rocSPARSE in your code would be using CMake for which you would add the SDK installation location to your
`CMAKE_PREFIX_PATH`. Note you need to use quotes as the path contains a space, e.g.,

::

    -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\5.5"


in your CMake configure step and then in your CMakeLists.txt use

::

    find_package(rocsparse)

    target_link_libraries( your_exe PRIVATE roc::rocsparse )

Otherwise once installed, rocSPARSE can be used just like any other library with a C API.
The rocsparse.h header file must be included in the user code to make calls
into rocSPARSE, and the rocSPARSE import library and dynamic link library will become respective link-time and run-time
dependencies for the user application.

Once installed, find rocsparse.h in the HIP SDK `\\include\\rocsparse`
directory. Only use these two installed files when needed in user code.
Find other rocSPARSE included files in HIP SDK `\\include\\rocsparse\\internal`, however,
do not include these files directly into source code.

---------------------------------
Building and Installing rocSPARSE
---------------------------------

Building from source is not necessary, as rocSPARSE can be used after installing the pre-built packages as described above.
If desired, the following instructions can be used to build rocSPARSE from source.

Requirements
^^^^^^^^^^^^

- `git <https://git-scm.com/>`_
- `CMake <https://cmake.org/>`_ 3.5 or later
- `AMD ROCm <https://github.com/RadeonOpenCompute/ROCm>`_
- `rocPRIM <https://github.com/ROCmSoftwarePlatform/rocPRIM>`_
- `vcpkg <https://github.com/Microsoft/vcpkg.git>`_
- `googletest <https://github.com/google/googletest>`_ (optional, for clients)


Download rocSPARSE
^^^^^^^^^^^^^^^^^^

The rocSPARSE source code, which is the same as for the ROCm linux distributions, is available at the `rocSPARSE github page <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_.
The version of the ROCm HIP SDK may be shown in the path of default installation, but
you can run the HIP SDK compiler to report the verison from the bin/ folder with:

::

    hipcc --version

The HIP version has major, minor, and patch fields, possibly followed by a build specific identifier. For example, HIP version could be 5.4.22880-135e1ab4;
this corresponds to major = 5, minor = 4, patch = 22880, build identifier 135e1ab4.
There are GitHub branches at the rocSPARSE site with names release/rocm-rel-major.minor where major and minor are the same as in the HIP version.
For example for you can use the following to download rocSPARSE:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/rocSPARSE.git
   cd rocSPARSE

Replace x.y in the above command with the version of HIP SDK installed on your machine. For example, if you have HIP 5.5 installed, then use -b release/rocm-rel-5.5
You can can add the SDK tools to your path with an entry like:

::

   %HIP_PATH%\bin

Building
^^^^^^^^

Below are steps to build using the `rmake.py` script. The user can build either:

* library

* library + client

You only need (library) if you call rocSPARSE from your code and only want the library built.
The client contains testing and benchmark tools.  rmake.py will print to the screen the full cmake command being used to configure rocSPARSE based on your rmake command line options.
This full cmake command can be used in your own build scripts if you want to bypass the python helper script for a fixed set of build options.


Build Library
^^^^^^^^^^^^^

Common uses of rmake.py to build (library) are
in the table below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+--------------------+--------------------------+
| Command            | Description              |
+====================+==========================+
| ``./rmake.py -h``  | Help information.        |
+--------------------+--------------------------+
| ``./rmake.py``     | Build library.           |
+--------------------+--------------------------+
| ``./rmake.py -i``  | Build library, then      |
|                    | build and install        |
|                    | rocSPARSE package.       |
|                    | If you want to keep      |
|                    | rocSPARSE in your local  |
|                    | tree, you do not         |
|                    | need the -i flag.        |
+--------------------+--------------------------+


Build Library + Client
^^^^^^^^^^^^^^^^^^^^^^

Some client executables (.exe) are listed in the table below:

====================== =================================================
executable name        description
====================== =================================================
rocsparse-test           runs Google Tests to test the library
rocsparse-bench          executable to benchmark or test functions
rocsparse_axpyi          example C code calling rocsparse_axpyi function
====================== =================================================

Common uses of rmake.py to build (library + client) are
in the table below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+------------------------+--------------------------+
| Command                | Description              |
+========================+==========================+
| ``./rmake.py -h``      | Help information.        |
+------------------------+--------------------------+
| ``./rmake.py -c``      | Build library and client |
|                        | in your local directory. |
+------------------------+--------------------------+
| ``./rmake.py -ic``     | Build and install        |
|                        | rocSPARSE package, and   |
|                        | build the client.        |
|                        | If you want to keep      |
|                        | rocSPARSE in your local  |
|                        | directory, you do not    |
|                        | need the -i flag.        |
+------------------------+--------------------------+
