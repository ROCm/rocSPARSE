.. _rocsparse_building:

Building and Installing
=======================

Prerequisites
-------------
rocSPARSE requires a ROCm enabled platform, more information `here <https://rocm.github.io/>`_.

Installing pre-built packages
-----------------------------
rocSPARSE can be installed from `AMD ROCm repository <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`_.
For detailed instructions on how to set up ROCm on different platforms, see the `AMD ROCm Platform Installation Guide for Linux <https://rocm.github.io/ROCmInstall.html>`_.

rocSPARSE can be installed on e.g. Ubuntu using

::

    $ sudo apt-get update
    $ sudo apt-get install rocsparse

Once installed, rocSPARSE can be used just like any other library with a C API.
The header file will need to be included in the user code in order to make calls into rocSPARSE, and the rocSPARSE shared library will become link-time and run-time dependent for the user application.

Building rocSPARSE from source
------------------------------
Building from source is not necessary, as rocSPARSE can be used after installing the pre-built packages as described above.
If desired, the following instructions can be used to build rocSPARSE from source.
Furthermore, the following compile-time dependencies must be met

- `git <https://git-scm.com/>`_
- `CMake <https://cmake.org/>`_ 3.5 or later
- `AMD ROCm <https://github.com/RadeonOpenCompute/ROCm>`_
- `rocPRIM <https://github.com/ROCmSoftwarePlatform/rocPRIM>`_
- `googletest <https://github.com/google/googletest>`_ (optional, for clients)

Download rocSPARSE
``````````````````
The rocSPARSE source code is available at the `rocSPARSE GitHub page <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_.
Download the master branch using:

::

  $ git clone -b master https://github.com/ROCmSoftwarePlatform/rocSPARSE.git
  $ cd rocSPARSE

Below are steps to build different packages of the library, including dependencies and clients.
It is recommended to install rocSPARSE using the `install.sh` script.

Using `install.sh` to build rocSPARSE with dependencies
```````````````````````````````````````````````````````
The following table lists common uses of `install.sh` to build dependencies + library.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

================= ====
Command           Description
================= ====
`./install.sh -h` Print help information.
`./install.sh -d` Build dependencies and library in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh`    Build library in your local directory. It is assumed dependencies are available.
`./install.sh -i` Build library, then build and install rocSPARSE package in `/opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
================= ====

Using `install.sh` to build rocSPARSE with dependencies and clients
```````````````````````````````````````````````````````````````````
The client contains example code, unit tests and benchmarks. Common uses of `install.sh` to build them are listed in the table below.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

=================== ====
Command             Description
=================== ====
`./install.sh -h`   Print help information.
`./install.sh -dc`  Build dependencies, library and client in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh -c`   Build library and client in your local directory. It is assumed dependencies are available.
`./install.sh -idc` Build library, dependencies and client, then build and install rocSPARSE package in `/opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
`./install.sh -ic`  Build library and client, then build and install rocSPARSE package in `opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
=================== ====

Using individual commands to build rocSPARSE
````````````````````````````````````````````
CMake 3.5 or later is required in order to build rocSPARSE.
The rocSPARSE library contains both, host and device code, therefore the HIP compiler must be specified during cmake configuration process.

rocSPARSE can be built using the following commands:

::

  # Create and change to build directory
  $ mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ CXX=/opt/rocm/bin/hipcc cmake ../..

  # Compile rocSPARSE library
  $ make -j$(nproc)

  # Install rocSPARSE to /opt/rocm
  $ make install

GoogleTest is required in order to build rocSPARSE clients.

rocSPARSE with dependencies and clients can be built using the following commands:

::

  # Install googletest
  $ mkdir -p build/release/deps ; cd build/release/deps
  $ cmake ../../../deps
  $ make -j$(nproc) install

  # Change to build directory
  $ cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ CXX=/opt/rocm/bin/hipcc cmake ../.. -DBUILD_CLIENTS_TESTS=ON \
                                        -DBUILD_CLIENTS_BENCHMARKS=ON \
                                        -DBUILD_CLIENTS_SAMPLES=ON

  # Compile rocSPARSE library
  $ make -j$(nproc)

  # Install rocSPARSE to /opt/rocm
  $ make install

Common build problems
`````````````````````
#. **Issue:** Could not find a package configuration file provided by "ROCM" with any of the following names: ROCMConfig.cmake, rocm-config.cmake

   **Solution:** Install `ROCm cmake modules <https://github.com/RadeonOpenCompute/rocm-cmake>`_

Simple Test
```````````
You can test the installation by running one of the rocSPARSE examples, after successfully compiling the library with clients.

::

   # Navigate to clients binary directory
   $ cd rocSPARSE/build/release/clients/staging

   # Execute rocSPARSE example
   $ ./example_csrmv 1000

Supported Targets
-----------------
Currently, rocSPARSE is supported under the following operating systems

- `Ubuntu 16.04 <https://ubuntu.com/>`_
- `Ubuntu 18.04 <https://ubuntu.com/>`_
- `CentOS 7 <https://www.centos.org/>`_
- `SLES 15 <https://www.suse.com/solutions/enterprise-linux/>`_

To compile and run rocSPARSE, `AMD ROCm Platform <https://github.com/RadeonOpenCompute/ROCm>`_ is required.

The following HIP capable devices are currently supported

- gfx803 (e.g. Fiji)
- gfx900 (e.g. Vega10, MI25)
- gfx906 (e.g. Vega20, MI50, MI60)
- gfx908
