.. meta::
  :description: rocSPARSE documentation and API reference library
  :keywords: rocSPARSE, ROCm, API, documentation

.. _linux-install:

********************************************************************
Installation and building for Linux
********************************************************************

Prerequisites
=============

- A ROCm enabled platform. `ROCm Documentation <https://docs.amd.com/>`_ has more information on
  supported GPUs, Linux distributions, and Windows SKUs. It also has information on how to install ROCm.


Installing pre-built packages
=============================

For detailed instructions on how to set up ROCm on different platforms, see the `ROCm Documentation <https://docs.amd.com/>`_.

rocSPARSE can be installed on e.g. Ubuntu using

::

    $ sudo apt-get update
    $ sudo apt-get install rocsparse

Once installed, rocSPARSE can be used just like any other library with a C API.
The header file will need to be included in the user code in order to make calls into rocSPARSE, and the rocSPARSE shared library will become link-time and run-time dependent for the user application.


Building rocSPARSE from source
==============================

Building from source is not necessary, as rocSPARSE can be used after installing the pre-built packages as described above.
If desired, the following instructions can be used to build rocSPARSE from source.

Requirements
------------

- `git <https://git-scm.com/>`_
- `CMake <https://cmake.org/>`_ 3.5 or later
- `AMD ROCm <https://github.com/ROCm/ROCm>`_
- `rocPRIM <https://github.com/ROCm/rocPRIM>`_
- `rocBLAS <https://github.com/ROCm/rocBLAS>`_ (optional, for library)
- `googletest <https://github.com/google/googletest>`_ (optional, for clients)
- `python <https://www.python.org/>`_
- `PyYaml <https://pypi.org/project/PyYAML/>`_

When building rocSPARSE from source, care must be given regarding what versions of the math library
dependencies (in this case rocPRIM and optionally rocBLAS) are supported. Given a version of rocSPARSE,
you must use the same or later version of rocPRIM (and optionally rocBLAS). For example, it should be
possible to re-build rocSPARSE 3.2.0 with any future rocPRIM 3.Y.Z version (within the same major version
and where 3.Y.Z is 3.2.0 or later), but it is not supported to compile rocSPARSE with an older version of
rocPRIM such as 3.1.0.

Download rocSPARSE
------------------
The rocSPARSE source code is available at the `rocSPARSE GitHub page <https://github.com/ROCm/rocSPARSE>`_.
Download the master branch using:

::

  $ git clone -b master https://github.com/ROCm/rocSPARSE.git
  $ cd rocSPARSE

Below are steps to build different packages of the library, including dependencies and clients.
It is recommended to install rocSPARSE using the ``install.sh`` script.

Using ``install.sh`` to build rocSPARSE with dependencies
---------------------------------------------------------
The following table lists common uses of ``install.sh`` to build dependencies + library.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

=========================== ====
Command                     Description
=========================== ====
`./install.sh -h`           Print help information.
`./install.sh -d`           Build dependencies and library in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh`              Build library in your local directory. It is assumed dependencies are available.
`./install.sh -i`           Build library, then build and install rocSPARSE package in `/opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
`./install.sh -i -a gfx908` Build library specifically for architecture gfx908, then build and install rocSPARSE package in `/opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
=========================== ====

Using ``install.sh`` to build rocSPARSE with dependencies and clients
---------------------------------------------------------------------
The client contains example code, unit tests and benchmarks. Common uses of ``install.sh`` to build them are listed in the table below.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

============================= ====
Command                       Description
============================= ====
`./install.sh -h`             Print help information.
`./install.sh -dc`            Build dependencies, library and client in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh -c`             Build library and client in your local directory. It is assumed dependencies are available.
`./install.sh -idc`           Build library, dependencies and client, then build and install rocSPARSE package in `/opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
`./install.sh -ic`            Build library and client, then build and install rocSPARSE package in `opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
`./install.sh -idc -a gfx908` Build library specifically for architecture gfx908, dependencies and client, then build and install rocSPARSE package in `/opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
`./install.sh -ic -a gfx908`  Build library specifically for architecture gfx908 and client, then build and install rocSPARSE package in `opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
============================= ====

Using individual commands to build rocSPARSE
--------------------------------------------
CMake 3.5 or later is required in order to build rocSPARSE.
The rocSPARSE library contains both, host and device code, therefore the HIP compiler must be specified during cmake configuration process.

rocSPARSE can be built using the following commands:

::

  # Create and change to build directory
  $ mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ CXX=/opt/rocm/bin/amdclang++ cmake ../..

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
  $ CXX=/opt/rocm/bin/amdclang++ cmake ../.. -DBUILD_CLIENTS_TESTS=ON \
                                        -DBUILD_CLIENTS_BENCHMARKS=ON \
                                        -DBUILD_CLIENTS_SAMPLES=ON

  # Compile rocSPARSE library
  $ make -j$(nproc)

  # Install rocSPARSE to /opt/rocm
  $ make install

Common build problems
=====================
#. **Issue:** Could not find a package configuration file provided by "ROCM" with any of the following names: ROCMConfig.cmake, rocm-config.cmake

   **Solution:** Install `ROCm cmake modules <https://github.com/ROCm/rocm-cmake>`_

Simple Test
-----------
You can test the installation by running one of the rocSPARSE examples, after successfully compiling the library with clients.

::

   # Navigate to clients binary directory
   $ cd rocSPARSE/build/release/clients/staging

   # Execute rocSPARSE example
   $ ./example_csrmv 1000

