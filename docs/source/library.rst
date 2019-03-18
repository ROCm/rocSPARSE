.. toctree::
   :maxdepth: 4 
   :caption: Contents:

.. |br| raw:: html

  <br />

=========
rocSPARSE
=========

Introduction
------------

rocSPARSE is a library that contains basic linear algebra subroutines for sparse matrices and vectors written in HiP for GPU devices. It is designed to be used from C and C++ code. The functionality of rocSPARSE is organized in the following categories:

* :ref:`rocsparse_auxiliary_functions_` describe available helper functions that are required for subsequent library calls.
* :ref:`rocsparse_level1_functions_` describe operations between a vector in sparse format and a vector in dense format.
* :ref:`rocsparse_level2_functions_` describe operations between a matrix in sparse format and a vector in dense format.
* :ref:`rocsparse_level3_functions_` describe operations between a matrix in sparse format and multiple vectors in dense format.
* :ref:`rocsparse_precond_functions_` describe manipulations on a matrix in sparse format to obtain a preconditioner.
* :ref:`rocsparse_conversion_functions_` describe operations on a matrix in sparse format to obtain a different matrix format.

The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/rocSPARSE

Device and Stream Management
*****************************
*hipSetDevice()* and *hipGetDevice()* are HIP device management APIs. They are NOT part of the rocSPARSE API.

Asynchronous Execution
``````````````````````
All rocSPARSE library functions, unless otherwise stated, are non blocking and executed asynchronously with respect to the host. They may return before the actual computation has finished. To force synchronization, *hipDeviceSynchronize()* or *hipStreamSynchronize()* can be used. This will ensure that all previously executed rocSPARSE functions on the device / this particular stream have completed.

HIP Device Management
``````````````````````
Before a HIP kernel invocation, users need to call *hipSetDevice()* to set a device, e.g. device 1. If users do not explicitly call it, the system by default sets it as device 0. Unless users explicitly call *hipSetDevice()* to set to another device, their HIP kernels are always launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing to do with rocSPARSE. rocSPARSE honors the approach above and assumes users have already set the device before a rocSPARSE routine call.

Once users set the device, they create a handle with :ref:`rocsparse_create_handle_`.

Subsequent rocSPARSE routines take this handle as an input parameter. rocSPARSE ONLY queries (by *hipGetDevice()*) the user's device; rocSPARSE does NOT set the device for users. If rocSPARSE does not see a valid device, it returns an error message. It is the users' responsibility to provide a valid device to rocSPARSE and ensure the device safety.

Users CANNOT switch devices between :ref:`rocsparse_create_handle_` and :ref:`rocsparse_destroy_handle_`. If users want to change device, they must destroy the current handle and create another rocSPARSE handle.

HIP Stream Management
``````````````````````
HIP kernels are always launched in a queue (also known as stream).

If users do not explicitly specify a stream, the system provides a default stream, maintained by the system. Users cannot create or destroy the default stream. However, users can freely create new streams (with *hipStreamCreate()*) and bind it to the rocSPARSE handle using :ref:`rocsparse_set_stream_`. HIP kernels are invoked in rocSPARSE routines. The rocSPARSE handle is always associated with a stream, and rocSPARSE passes its stream to the kernels inside the routine. One rocSPARSE routine only takes one stream in a single invocation. If users create a stream, they are responsible for destroying it.

Multiple Streams and Multiple Devices
``````````````````````````````````````
If the system under test has multiple HIP devices, users can run multiple rocSPARSE handles concurrently, but can NOT run a single rocSPARSE handle on different discrete devices. Each handle is associated with a particular singular device, and a new handle should be created for each additional device.

.. _rocsparse_contributing:

Contributing
*************

Contribution License Agreement
```````````````````````````````

#. The code I am contributing is mine, and I have the right to license it.
#. By submitting a pull request for this project I am granting you a license to distribute said code under the MIT License for the project.

How to contribute
``````````````````
Our code contriubtion guidelines closely follows the model of GitHub pull-requests. This repository follows the git flow workflow, which dictates a /master branch where releases are cut, and a /develop branch which serves as an integration branch for new code.

A `git extention <https://github.com/nvie/gitflow>`_ has been developed to ease the use of the 'git flow' methodology, but requires manual installation by the user. Please refer to the projects wiki.

Pull-request guidelines
````````````````````````
* Target the **develop** branch for integration.
* Ensure code builds successfully.
* Do not break existing test cases
* New functionality will only be merged with new unit tests.

  * New unit tests should integrate within the existing `googletest framework <https://github.com/google/googletest/blob/master/googletest/docs/primer.md>`_.
  * Tests must have good code coverage.
  * Code must also have benchmark tests, and performance must approach the compute bound limit or memory bound limit.

StyleGuide
```````````
This project follows the `CPP Core guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`_, with few modifications or additions noted below. All pull-requests should in good faith attempt to follow the guidelines stated therein, but we recognize that the content is lengthy. Below we list our primary concerns when reviewing pull-requests.

**Interface**

* All public APIs are C89 compatible; all other library code should use C++14.
* Our minimum supported compiler is clang 3.6.
* Avoid CamelCase.
* This rule applies specifically to publicly visible APIs, but is also encouraged (not mandated) for internal code.

**Philosophy**

* `P.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-Cplusplus>`_: Write in ISO Standard C++ (especially to support Windows, Linux and MacOS platforms).
* `P.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-compile-time>`_: Prefer compile-time checking to run-time checking.

**Implementation**

* `SF.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-file-suffix>`_: Use a .cpp suffix for code files and .h for interface files if your project doesn't already follow another convention.
* We modify this rule:

  * .h: C header files.
  * .hpp: C++ header files.

* `SF.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-consistency>`_: A .cpp file must include the .h file(s) that defines its interface.
* `SF.7 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive>`_: Don't put a using-directive in a header file.
* `SF.8 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-guards>`_: Use #include guards for all .h files.
* `SF.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-unnamed>`_: Don't use an unnamed (anonymous) namespace in a header.
* `SL.10 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rsl-arrays>`_: Prefer using STL array or vector instead of a C array.
* `C.9 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-private>`_: Minimize exposure of members.
* `F.3 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-single>`_: Keep functions short and simple.
* `F.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out-multi>`_: To return multiple 'out' values, prefer returning a tuple.
* `R.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rr-raii>`_: Manage resources automatically using RAII (this includes unique_ptr & shared_ptr).
* `ES.11 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-auto>`_:  Use auto to avoid redundant repetition of type names.
* `ES.20 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-always>`_: Always initialize an object.
* `ES.23 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-list>`_: Prefer the {} initializer syntax.
* `ES.49 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-casts-named>`_: If you must use a cast, use a named cast.
* `CP.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#S-concurrency>`_: Assume that your code will run as part of a multi-threaded program.
* `I.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global>`_: Avoid global variables.

**Format**

C and C++ code is formatted using clang-format. To format a file, use

::

  clang-format-3.8 -style=file -i <file>

To format all files, run the following script in rocSPARSE directory:

::

  #!/bin/bash

  find . -iname '*.h' \
  -o -iname '*.hpp' \
  -o -iname '*.cpp' \
  -o -iname '*.h.in' \
  -o -iname '*.hpp.in' \
  -o -iname '*.cpp.in' \
  -o -iname '*.cl' \
  | grep -v 'build' \
  | xargs -n 1 -P 8 -I{} clang-format-3.8 -style=file -i {}

Also, githooks can be installed to format the code per-commit:

::

  ./.githooks/install

Building and Installing
-----------------------

Installing from AMD ROCm repositories
**************************************
rocSPARSE can be installed from `AMD ROCm repositories <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`_ by

::

  sudo apt install rocsparse


Building rocSPARSE from Open-Source repository
***********************************************

Download rocSPARSE
```````````````````
The rocSPARSE source code is available at the `rocSPARSE github page <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_.
Download the master branch using:

::

  git clone -b master https://github.com/ROCmSoftwarePlatform/rocSPARSE.git
  cd rocSPARSE


Note that if you want to contribute to rocSPARSE, you will need to checkout the develop branch instead of the master branch. See :ref:`rocsparse_contributing` for further details.
Below are steps to build different packages of the library, including dependencies and clients.
It is recommended to install rocSPARSE using the *install.sh* script.

Using *install.sh* to build dependencies + library
```````````````````````````````````````````````````
The following table lists common uses of *install.sh* to build dependencies + library.

================= ====
Command           Description
================= ====
`./install.sh -h` Print help information.
`./install.sh -d` Build dependencies and library in your local directory. The `-d` flag only needs to be |br| used once. For subsequent invocations of *install.sh* it is not necessary to rebuild the |br| dependencies.
`./install.sh`    Build library in your local directory. It is assumed dependencies are available.
`./install.sh -i` Build library, then build and install rocSPARSE package in `/opt/rocm/rocsparse`. You will be |br| prompted for sudo access. This will install for all users.
================= ====

Using *install.sh* to build dependencies + library + client
````````````````````````````````````````````````````````````
The client contains example code, unit tests and benchmarks. Common uses of *install.sh* to build them are listed in the table below.

=================== ====
Command             Description
=================== ====
`./install.sh -h`   Print help information.
`./install.sh -dc`  Build dependencies, library and client in your local directory. The `-d` flag only needs to be |br| used once. For subsequent invocations of *install.sh* it is not necessary to rebuild the |br| dependencies.
`./install.sh -c`   Build library and client in your local directory. It is assumed dependencies are available.
`./install.sh -idc` Build library, dependencies and client, then build and install rocSPARSE package in |br| `/opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
`./install.sh -ic`  Build library and client, then build and install rocSPARSE package in `opt/rocm/rocsparse`. |br| You will be prompted for sudo access. This will install for all users.
=================== ====

Using individual commands to build rocSPARSE
`````````````````````````````````````````````
CMake 3.5 or later is required in order to build rocSPARSE.
The rocSPARSE library contains both, host and device code, therefore the HCC compiler must be specified during cmake configuration process.

rocSPARSE can be built using the following commands:

::

  # Create and change to build directory
  mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  CXX=/opt/rocm/bin/hcc cmake ../..

  # Compile rocSPARSE library
  make -j$(nproc)

  # Install rocSPARSE to /opt/rocm
  sudo make install

Boost and GoogleTest is required in order to build rocSPARSE client.

rocSPARSE with dependencies and client can be built using the following commands:

::

  # Install boost on Ubuntu
  sudo apt install libboost-program-options-dev
  # Install boost on Fedora
  sudo dnf install boost-program-options

  # Install googletest
  mkdir -p build/release/deps ; cd build/release/deps
  cmake -DBUILD_BOOST=OFF ../../../deps
  sudo make -j$(nproc) install

  # Change to build directory
  cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  CXX=/opt/rocm/bin/hcc cmake ../.. -DBUILD_CLIENTS_TESTS=ON \
                                    -DBUILD_CLIENTS_BENCHMARKS=ON \
                                    -DBUILD_CLIENTS_SAMPLES=ON

  # Compile rocSPARSE library
  make -j$(nproc)

  # Install rocSPARSE to /opt/rocm
  sudo make install

Common build problems
``````````````````````
#. **Issue:** HIP (/opt/rocm/hip) was built using hcc 1.0.xxx-xxx-xxx-xxx, but you are using /opt/rocm/bin/hcc with version 1.0.yyy-yyy-yyy-yyy from hipcc (version mismatch). Please rebuild HIP including cmake or update HCC_HOME variable.

   **Solution:** Download HIP from github and use hcc to `build from source <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md>`_ and then use the built HIP instead of /opt/rocm/hip.

#. **Issue:** For Carrizo - HCC RUNTIME ERROR: Failed to find compatible kernel

   **Solution:** Add the following to the cmake command when configuring: `-DCMAKE_CXX_FLAGS="--amdgpu-target=gfx801"`

#. **Issue:** For MI25 (Vega10 Server) - HCC RUNTIME ERROR: Failed to find compatible kernel

   **Solution:** `export HCC_AMDGPU_TARGET=gfx900`

#. **Issue:** Could not find a package configuration file provided by "ROCM" with any of the following names:
              ROCMConfig.cmake |br|
              rocm-config.cmake

   **Solution:** Install `ROCm cmake modules <https://github.com/RadeonOpenCompute/rocm-cmake>`_

Storage Formats
---------------

COO storage format
*******************
The Coordinate (COO) storage format represents a :math:`m \times n` matrix by

=========== ==================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
coo_val     array of ``nnz`` elements containing the data (floating point).
coo_row_ind array of ``nnz`` elements containing the row indices (integer).
coo_col_ind array of ``nnz`` elements containing the column indices (integer).
=========== ==================================================================

The COO matrix is expected to be sorted by row indices and column indices per row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding COO structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using zero based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{coo_val}[8] & = \{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0\} \\
    \text{coo_row_ind}[8] & = \{0, 0, 0, 1, 1, 2, 2, 2\} \\
    \text{coo_col_ind}[8] & = \{0, 1, 3, 1, 2, 0, 3, 4\}
  \end{array}

CSR storage format
*******************
The Compressed Sparse Row (CSR) storage format represents a :math:`m \times n` matrix by

=========== =========================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
csr_val     array of ``nnz`` elements containing the data (floating point).
csr_row_ptr array of ``m+1`` elements that point to the start of every row (integer).
csr_col_ind array of ``nnz`` elements containing the column indices (integer).
=========== =========================================================================

The CSR matrix is expected to be sorted by column indices within each row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding CSR structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using one based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{csr_val}[8] & = \{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0\} \\
    \text{csr_row_ptr}[4] & = \{1, 4, 6, 9\} \\
    \text{csr_col_ind}[8] & = \{1, 2, 4, 2, 3, 1, 4, 5\}
  \end{array}

ELL storage format
*******************
The Ellpack-Itpack (ELL) storage format represents a :math:`m \times n` matrix by

=========== ================================================================================
m           number of rows (integer).
n           number of columns (integer).
ell_width   maximum number of non-zero elements per row (integer)
ell_val     array of ``m times ell_width`` elements containing the data (floating point).
ell_col_ind array of ``m times ell_width`` elements containing the column indices (integer).
=========== ================================================================================

The ELL matrix is assumed to be stored in column-major format. Rows with less than ``ell_width`` non-zero elements are padded with zeros (``ell_val``) and :math:`-1` (``ell_col_ind``).
Consider the following :math:`3 \times 5` matrix and the corresponding ELL structures, with :math:`m = 3, n = 5` and :math:`\text{ell_width} = 3` using zero based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{ell_val}[9] & = \{1.0, 4.0, 6.0, 2.0, 5.0, 7.0, 3.0, 0.0, 8.0\} \\
    \text{ell_col_ind}[9] & = \{0, 1, 0, 1, 2, 3, 3, -1, 4\}
  \end{array}

.. _HYB storage format:

HYB storage format
*******************
The Hybrid (HYB) storage format represents a :math:`m \times n` matrix by

=========== =========================================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements of the COO part (integer)
ell_width   maximum number of non-zero elements per row of the ELL part (integer)
ell_val     array of ``m times ell_width`` elements containing the ELL part data (floating point).
ell_col_ind array of ``m times ell_width`` elements containing the ELL part column indices (integer).
coo_val     array of ``nnz`` elements containing the COO part data (floating point).
coo_row_ind array of ``nnz`` elements containing the COO part row indices (integer).
coo_col_ind array of ``nnz`` elements containing the COO part column indices (integer).
=========== =========================================================================================

The HYB format is a combination of the ELL and COO sparse matrix formats. Typically, the regular part of the matrix is stored in ELL storage format, and the irregular part of the matrix is stored in COO storage format. Three different partitioning schemes can be applied when converting a CSR matrix to a matrix in HYB storage format. For further details on the partitioning schemes, see :ref:`rocsparse_hyb_partition_`.

Types
-----

rocsparse_handle
*****************

.. doxygentypedef:: rocsparse_handle

rocsparse_mat_descr
********************

.. doxygentypedef:: rocsparse_mat_descr


.. _rocsparse_mat_info_:

rocsparse_mat_info
*******************

.. doxygentypedef:: rocsparse_mat_info

rocsparse_hyb_mat
******************

.. doxygentypedef:: rocsparse_hyb_mat

For more details on the HYB format, see :ref:`HYB storage format`.

rocsparse_action
*****************

.. doxygenenum:: rocsparse_action

.. _rocsparse_hyb_partition_:

rocsparse_hyb_partition
************************

.. doxygenenum:: rocsparse_hyb_partition

rocsparse_index_base
*********************

.. doxygenenum:: rocsparse_index_base

rocsparse_matrix_type
**********************

.. doxygenenum:: rocsparse_matrix_type

rocsparse_fill_mode
*******************

.. doxygenenum:: rocsparse_fill_mode

rocsparse_diag_type
*******************

.. doxygenenum:: rocsparse_diag_type

rocsparse_operation
********************

.. doxygenenum:: rocsparse_operation

rocsparse_pointer_mode
***********************

.. doxygenenum:: rocsparse_pointer_mode

rocsparse_analysis_policy
*************************

.. doxygenenum:: rocsparse_analysis_policy

rocsparse_solve_policy
**********************

.. doxygenenum:: rocsparse_solve_policy

.. _rocsparse_layer_mode_:

rocsparse_layer_mode
*********************

.. doxygenenum:: rocsparse_layer_mode

For more details on logging, see :ref:`rocsparse_logging`.

rocsparse_status
*****************

.. doxygenenum:: rocsparse_status

.. _rocsparse_logging:

Logging
-------
Three different environment variables can be set to enable logging in rocSPARSE: ``ROCSPARSE_LAYER``, ``ROCSPARSE_LOG_TRACE_PATH`` and ``ROCSPARSE_LOG_BENCH_PATH``.

``ROCSPARSE_LAYER`` is a bit mask, where several logging modes (:ref:`rocsparse_layer_mode_`) can be combined as follows:

================================  ===========================================
``ROCSPARSE_LAYER`` unset         logging is disabled.
``ROCSPARSE_LAYER`` set to ``1``  trace logging is enabled.
``ROCSPARSE_LAYER`` set to ``2``  bench logging is enabled.
``ROCSPARSE_LAYER`` set to ``3``  trace logging and bench logging is enabled.
================================  ===========================================

When logging is enabled, each rocSPARSE function call will write the function name as well as function arguments to the logging stream. The default logging stream is ``stderr``.

If the user sets the environment variable ``ROCSPARSE_LOG_TRACE_PATH`` to the full path name for a file, the file is opened and trace logging is streamed to that file. If the user sets the environment variable ``ROCSPARSE_LOG_BENCH_PATH`` to the full path name for a file, the file is opened and bench logging is streamed to that file. If the file cannot be opened, logging output is stream to ``stderr``.

Note that performance will degrade when logging is enabled. By default, the environment variable ``ROCSPARSE_LAYER`` is unset and logging is disabled.

.. _rocsparse_auxiliary_functions_:

Sparse Auxiliary Functions
--------------------------

This module holds all sparse auxiliary functions.

The functions that are contained in the auxiliary module describe all available helper functions that are required for subsequent library calls.

.. _rocsparse_create_handle_:

rocsparse_create_handle()
**************************

.. doxygenfunction:: rocsparse_create_handle

.. _rocsparse_destroy_handle_:

rocsparse_destroy_handle()
***************************

.. doxygenfunction:: rocsparse_destroy_handle

.. _rocsparse_set_stream_:

rocsparse_set_stream()
***********************

.. doxygenfunction:: rocsparse_set_stream

rocsparse_get_stream()
***********************

.. doxygenfunction:: rocsparse_get_stream

rocsparse_set_pointer_mode()
*****************************

.. doxygenfunction:: rocsparse_set_pointer_mode

rocsparse_get_pointer_mode()
*****************************

.. doxygenfunction:: rocsparse_get_pointer_mode

rocsparse_get_version()
************************

.. doxygenfunction:: rocsparse_get_version

rocsparse_get_git_rev()
************************

.. doxygenfunction:: rocsparse_get_git_rev

rocsparse_create_mat_descr()
*****************************

.. doxygenfunction:: rocsparse_create_mat_descr

rocsparse_destroy_mat_descr()
******************************

.. doxygenfunction:: rocsparse_destroy_mat_descr

rocsparse_copy_mat_descr()
**************************

.. doxygenfunction:: rocsparse_copy_mat_descr

rocsparse_set_mat_index_base()
*******************************

.. doxygenfunction:: rocsparse_set_mat_index_base

rocsparse_get_mat_index_base()
*******************************

.. doxygenfunction:: rocsparse_get_mat_index_base

rocsparse_set_mat_type()
*************************

.. doxygenfunction:: rocsparse_set_mat_type

rocsparse_get_mat_type()
*************************

.. doxygenfunction:: rocsparse_get_mat_type

rocsparse_set_mat_fill_mode()
*****************************

.. doxygenfunction:: rocsparse_set_mat_fill_mode

rocsparse_get_mat_fill_mode()
*****************************

.. doxygenfunction:: rocsparse_get_mat_fill_mode

rocsparse_set_mat_diag_type()
*****************************

.. doxygenfunction:: rocsparse_set_mat_diag_type

rocsparse_get_mat_diag_type()
*****************************

.. doxygenfunction:: rocsparse_get_mat_diag_type

.. _rocsparse_create_hyb_mat_:

rocsparse_create_hyb_mat()
***************************

.. doxygenfunction:: rocsparse_create_hyb_mat

rocsparse_destroy_hyb_mat()
****************************

.. doxygenfunction:: rocsparse_destroy_hyb_mat

rocsparse_create_mat_info()
***************************

.. doxygenfunction:: rocsparse_create_mat_info

.. _rocsparse_destroy_mat_info_:

rocsparse_destroy_mat_info()
*****************************

.. doxygenfunction:: rocsparse_destroy_mat_info

.. _rocsparse_level1_functions_:

Sparse Level 1 Functions
------------------------

The sparse level 1 routines describe operations between a vector in sparse format and a vector in dense format. This section describes all rocSPARSE level 1 sparse linear algebra functions.

rocsparse_axpyi()
*****************

.. doxygenfunction:: rocsparse_saxpyi
  :outline:
.. doxygenfunction:: rocsparse_daxpyi

rocsparse_doti()
*********************

.. doxygenfunction:: rocsparse_sdoti
  :outline:
.. doxygenfunction:: rocsparse_ddoti

rocsparse_gthr()
*********************

.. doxygenfunction:: rocsparse_sgthr
  :outline:
.. doxygenfunction:: rocsparse_dgthr

rocsparse_gthrz()
*********************

.. doxygenfunction:: rocsparse_sgthrz
  :outline:
.. doxygenfunction:: rocsparse_dgthrz

rocsparse_roti()
****************

.. doxygenfunction:: rocsparse_sroti
  :outline:
.. doxygenfunction:: rocsparse_droti

rocsparse_sctr()
****************

.. doxygenfunction:: rocsparse_ssctr
  :outline:
.. doxygenfunction:: rocsparse_dsctr

.. _rocsparse_level2_functions_:

Sparse Level 2 Functions
------------------------

This module holds all sparse level 2 routines.

The sparse level 2 routines describe operations between a matrix in sparse format and a vector in dense format.

rocsparse_coomv()
*****************

.. doxygenfunction:: rocsparse_scoomv
  :outline:
.. doxygenfunction:: rocsparse_dcoomv

rocsparse_csrmv_analysis()
***************************

.. doxygenfunction:: rocsparse_scsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrmv_analysis

rocsparse_csrmv()
*****************

.. doxygenfunction:: rocsparse_scsrmv
  :outline:
.. doxygenfunction:: rocsparse_dcsrmv

rocsparse_csrmv_analysis_clear()
*********************************

.. doxygenfunction:: rocsparse_csrmv_clear

rocsparse_ellmv()
*****************

.. doxygenfunction:: rocsparse_sellmv
  :outline:
.. doxygenfunction:: rocsparse_dellmv

rocsparse_hybmv()
*****************

.. doxygenfunction:: rocsparse_shybmv
  :outline:
.. doxygenfunction:: rocsparse_dhybmv

rocsparse_csrsv_zero_pivot()
****************************

.. doxygenfunction:: rocsparse_csrsv_zero_pivot

rocsparse_csrsv_buffer_size()
*****************************

.. doxygenfunction:: rocsparse_scsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrsv_buffer_size

rocsparse_csrsv_analysis()
**************************

.. doxygenfunction:: rocsparse_scsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrsv_analysis

rocsparse_csrsv_solve()
***********************

.. doxygenfunction:: rocsparse_scsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_dcsrsv_solve

rocsparse_csrsv_clear()
********************************

.. doxygenfunction:: rocsparse_csrsv_clear

.. _rocsparse_level3_functions_:

Sparse Level 3 Functions
------------------------

This module holds all sparse level 3 routines.

The sparse level 3 routines describe operations between a matrix in sparse format and multiple vectors in dense format that can also be seen as a dense matrix.

rocsparse_csrmm()
*********************

.. doxygenfunction:: rocsparse_scsrmm
  :outline:
.. doxygenfunction:: rocsparse_dcsrmm

.. _rocsparse_precond_functions_:

Preconditioner Functions
------------------------

This module holds all sparse preconditioners.

The sparse preconditioners describe manipulations on a matrix in sparse format to obtain a sparse preconditioner matrix.

rocsparse_csrilu0_zero_pivot()
******************************

.. doxygenfunction:: rocsparse_csrilu0_zero_pivot

rocsparse_csrilu0_buffer_size()
*******************************

.. doxygenfunction:: rocsparse_scsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0_buffer_size

rocsparse_csrilu0_analysis()
****************************

.. doxygenfunction:: rocsparse_scsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0_analysis

rocsparse_csrilu0()
*******************

.. doxygenfunction:: rocsparse_scsrilu0
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0

rocsparse_csrilu0_clear()
**********************************

.. doxygenfunction:: rocsparse_csrilu0_clear

.. _rocsparse_conversion_functions_:

Sparse Conversion Functions
---------------------------

This module holds all sparse conversion routines.

The sparse conversion routines describe operations on a matrix in sparse format to obtain a matrix in a different sparse format.

rocsparse_csr2coo()
*******************

.. doxygenfunction:: rocsparse_csr2coo

rocsparse_coo2csr()
*******************

.. doxygenfunction:: rocsparse_coo2csr

rocsparse_csr2csc_buffer_size()
*******************************

.. doxygenfunction:: rocsparse_csr2csc_buffer_size

rocsparse_csr2csc()
*******************

.. doxygenfunction:: rocsparse_scsr2csc
  :outline:
.. doxygenfunction:: rocsparse_dcsr2csc

rocsparse_csr2ell_width()
*************************

.. doxygenfunction:: rocsparse_csr2ell_width

rocsparse_csr2ell()
*******************

.. doxygenfunction:: rocsparse_scsr2ell
  :outline:
.. doxygenfunction:: rocsparse_dcsr2ell

rocsparse_ell2csr_nnz()
***********************

.. doxygenfunction:: rocsparse_ell2csr_nnz

rocsparse_ell2csr()
*******************

.. doxygenfunction:: rocsparse_sell2csr
  :outline:
.. doxygenfunction:: rocsparse_dell2csr

rocsparse_csr2hyb()
*******************

.. doxygenfunction:: rocsparse_scsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_dcsr2hyb

rocsparse_create_identity_permutation()
***************************************

.. doxygenfunction:: rocsparse_create_identity_permutation

rocsparse_csrsort_buffer_size()
*******************************

.. doxygenfunction:: rocsparse_csrsort_buffer_size

rocsparse_csrsort()
*******************

.. doxygenfunction:: rocsparse_csrsort

rocsparse_coosort_buffer_size()
*******************************

.. doxygenfunction:: rocsparse_coosort_buffer_size

rocsparse_coosort_by_row()
**************************

.. doxygenfunction:: rocsparse_coosort_by_row

rocsparse_coosort_by_column()
*****************************

.. doxygenfunction:: rocsparse_coosort_by_column
