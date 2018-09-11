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

rocSPARSE is a library that contains basic linear algebra subroutines for sparse matrices and vectors written in HiP for GPU devices. It is designed to be used from C and C++ code. The functionality of rocSPARSE is organized in in the following categories:

* :ref:`rocsparse_auxiliary_functions` describe available helper functions that are required for subsequent library calls.
* :ref:`rocsparse_level1_functions` describe operations between a vector in sparse format and a vector in dense format.
* :ref:`rocsparse_level2_functions` describe operations between a matrix in sparse format and a vector in dense format.
* :ref:`rocsparse_level3_functions` describe operations between a matrix in sparse format and multiple vectors in dense format.
* :ref:`rocsparse_precond_functions` describe manipulations on a matrix in sparse format to obtain a preconditioner.
* :ref:`rocsparse_conversion_functions` describe operations on a matrix in sparse format to obtain a different matrix format.

The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/rocSPARSE

Device and Stream Management
*****************************
:cpp:func:`hipSetDevice` and :cpp:func:`hipGetDevice` are HIP device management APIs. They are NOT part of the rocSPARSE API.

HIP Device Management
``````````````````````
Before a HIP kernel invocation, users need to call *hipSetDevice()* to set a device, e.g. device 1. If users do not explicitly call it, the system by default sets it as device 0. Unless users explicitly call *hipSetDevice()* to set to another device, their HIP kernels are always launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing to do with rocSPARSE. rocSPARSE honors the approach above and assumes users have already set the device before a rocSPARSE routine call.

Once users set the device, they create a handle with :ref:`rocsparse_create_handle`.

Subsequent rocSPARSE routines take this handle as an input parameter. rocSPARSE ONLY queries (by *hipGetDevice()*) the user's device; rocSPARSE does NOT set the device for users. If rocSPARSE does not see a valid device, it returns an error message. It is the users' responsibility to provide a valid device to rocSPARSE and ensure the device safety.

Users CANNOT switch devices between :ref:`rocsparse_create_handle` and :ref:`rocsparse_destroy_handle`. If users want to change device, they must destroy the current handle and create another rocSPARSE handle.

HIP Stream Management
``````````````````````
HIP kernels are always launched in a queue (also known as stream).

If users do not explicitly specify a stream, the system provides a default stream, maintained by the system. Users cannot create or destroy the default stream. However, users can freely create new streams (with *hipStreamCreate()*) and bind it to the rocSPARSE handle using :ref:`rocsparse_set_stream`. HIP kernels are invoked in rocSPARSE routines. The rocSPARSE handle is always associated with a stream, and rocSPARSE passes its stream to the kernels inside the routine. One rocSPARSE routine only takes one stream in a single invocation. If users create a stream, they are responsible for destroying it.

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
TODO, not yet available

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

The HYB format is a combination of the ELL and COO sparse matrix formats. Typically, the regular part of the matrix is stored in ELL storage format, and the irregular part of the matrix is stored in COO storage format. Three different partitioning schemes can be applied when converting a CSR matrix to a matrix in HYB storage format. For further details on the partitioning schemes, see :ref:`rocsparse_hyb_partition`.

Types
-----

.. _rocsparse_handle:

rocsparse_handle
*****************
The rocSPARSE handle is a structure holding the rocSPARSE library context. It must be initialized using :ref:`rocsparse_create_handle` and the returned handle must be passed to all subsequent library function calls. It should be destroyed at the end using :ref:`rocsparse_destroy_handle`.

.. _rocsparse_mat_descr:

rocsparse_mat_descr
********************
The rocSPARSE matrix descriptor is a structure holding all properties of a matrix. It must be initialized using :ref:`rocsparse_create_mat_descr` and the returned descriptor must be passed to all subsequent library calls that involve the matrix. It should be destroyed at the end using :ref:`rocsparse_destroy_mat_descr`.

.. _rocsparse_mat_info:

rocsparse_mat_info
*******************
The rocSPARSE matrix info is a structure holding all matrix information that is gathered during analysis routines. It must be initialized using :ref:`rocsparse_create_mat_info` and the returned info structure must be passed to all subsequent library calls that require additional matrix information. It should be destroyed at the end using :ref:`rocsparse_destroy_mat_info`.

.. _rocsparse_hyb_mat:

rocsparse_hyb_mat
******************
The rocSPARSE HYB matrix structure holds the HYB matrix. It must be initialized using :ref:`rocsparse_create_hyb_mat` and the returned HYB matrix must be passed to all subsequent library calls that involve the matrix. It should be destroyed at the end using :ref:`rocsparse_destroy_hyb_mat`. For more details on the HYB format, see :ref:`HYB storage format`.

.. _rocsparse_action:

rocsparse_action
*****************
The rocSPARSE action indicates whether the operation is performed on the full matrix, or only on the sparsity pattern of the matrix.

========================= ====
rocsparse_action_numeric  operate on data and indices.
rocsparse_action_symbolic operate only on indices.
========================= ====

.. _rocsparse_hyb_partition:

rocsparse_hyb_partition
************************
The rocSPARSE hyb partition type indicates how the hybrid format partitioning between COO and ELL storage formats is performed.

============================ ====
rocsparse_hyb_partition_auto automatically decide on ELL non-zero elements per row (recommended).
rocsparse_hyb_partition_user user given ELL non-zero elements per row.
rocsparse_hyb_partition_max  maximum ELL non-zero elements per row, COO part is empty.
============================ ====

.. _rocsparse_index_base:

rocsparse_index_base
*********************
The rocSPARSE index base indicates the index base of the given indices.

========================= ===========================================
rocsparse_index_base_zero zero index base (e.g. indices start with 0)
rocsparse_index_base_one  one index base (e.g. indices start with 1)
========================= ===========================================

.. _rocsparse_layer_mode:

rocsparse_layer_mode
*********************
The rocSPARSE layer mode bit mask indicates the logging characteristics. See :ref:`rocsparse_logging` for more informations.

============================== ==============================
rocsparse_layer_mode_none      layer is not active.
rocsparse_layer_mode_log_trace layer is in logging mode.
rocsparse_layer_mode_log_bench layer is in benchmarking mode.
============================== ==============================

.. _rocsparse_matrix_type:

rocsparse_matrix_type
**********************
The rocSPARSE matrix type indices the type of the given matrix.

================================ ======================
rocsparse_matrix_type_general    general matrix type
rocsparse_matrix_type_symmetric  symmetric matrix type
rocsparse_matrix_type_hermitian  hermitian matrix type
rocsparse_matrix_type_triangular triangular matrix type
================================ ======================

.. _rocsparse_operation:

rocsparse_operation
********************
The rocSPARSE operation indicates the operation performed with the given matrix.

======================================= ==================================
rocsparse_operation_none                non transposed operation mode
rocsparse_operation_transpose           transpose operation mode
rocsparse_operation_conjugate_transpose conjugate transpose operation mode
======================================= ==================================

.. _rocsparse_pointer_mode:

rocsparse_pointer_mode
***********************
The rocSPARSE pointer mode indicates whether scalar values are passed by reference on the host or device. The pointer mode can be changed by :ref:`rocsparse_set_pointer_mode`. The currently used pointer mode can be obtained by :ref:`rocsparse_get_pointer_mode`.

The following table lists the available pointer modes.

============================= ====================================================
rocsparse_pointer_mode_host   scalar values are passed by reference on the host.
rocsparse_pointer_mode_device scalar values are passed by reference on the device.
============================= ====================================================

.. _rocsparse_status:

rocsparse_status
*****************
This is a list of the status types that are used by the rocSPARSE library.

================================ ============================================
rocsparse_status_success         success.
rocsparse_status_invalid_handle  handle not initialized, invalid or ``NULL``.
rocsparse_status_not_implemented function is not implemented.
rocsparse_status_invalid_pointer invalid pointer parameter.
rocsparse_status_invalid_size    invalid size parameter.
rocsparse_status_memory_error    failed memory allocation, copy or dealloc.
rocsparse_status_internal_error  other internal library failure.
rocsparse_status_invalid_value   invalid value parameter.
rocsparse_status_arch_mismatch   device is not supported.
================================ ============================================

.. _rocsparse_logging:

Logging
-------
Three different environment variables can be set to enable logging in rocSPARSE: ``ROCSPARSE_LAYER``, ``ROCSPARSE_LOG_TRACE_PATH`` and ``ROCSPARSE_LOG_BENCH_PATH``.

``ROCSPARSE_LAYER`` is a bit mask, where several logging modes (:ref:`rocsparse_layer_mode`) can be combined as follows:

================================  ===========================================
``ROCSPARSE_LAYER`` unset         logging is disabled.
``ROCSPARSE_LAYER`` set to ``1``  trace logging is enabled.
``ROCSPARSE_LAYER`` set to ``2``  bench logging is enabled.
``ROCSPARSE_LAYER`` set to ``3``  trace logging and bench logging is enabled.
================================  ===========================================

When logging is enabled, each rocSPARSE function call will write the function name as well as function arguments to the logging stream. The default logging stream is ``stderr``.

If the user sets the environment variable ``ROCSPARSE_LOG_TRACE_PATH`` to the full path name for a file, the file is opened and trace logging is streamed to that file. If the user sets the environment variable ``ROCSPARSE_LOG_BENCH_PATH`` to the full path name for a file, the file is opened and bench logging is streamed to that file. If the file cannot be opened, logging output is stream to ``stderr``.

Note that performance will degrade when logging is enabled. By default, the environment variable ``ROCSPARSE_LAYER`` is unset and logging is disabled.

.. _rocsparse_auxiliary_functions:

Sparse Auxiliary Functions
--------------------------
This section describes all rocSPARSE auxiliary functions.

.. _rocsparse_create_handle:

rocsparse_create_handle()
**************************
.. code-block:: c

  rocsparse_status
  rocsparse_create_handle(rocsparse_handle* handle);

*rocsparse_create_handle()* creates the rocSPARSE library context. It must be initialized before any other rocSPARSE API function is invoked and must be passed to all subsequent library function calls. The handle should be destroyed at the end using :ref:`rocsparse_destroy_handle`.

====== ===========================================================
Output
====== ===========================================================
handle the pointer to the handle to the rocSPARSE library context.
====== ===========================================================

=============================== ===============================================
Status Returned
=============================== ===============================================
rocsparse_status_success        the initialization succeeded.
rocsparse_status_invalid_handle ``handle`` pointer is invalid.
rocsparse_status_internal_error an internal error occurred.
=============================== ===============================================

.. _rocsparse_destroy_handle:

rocsparse_destroy_handle()
***************************
.. code-block:: c

  rocsparse_status
  rocsparse_destroy_handle(rocsparse_handle handle);

*rocsparse_destroy_handle()* destroys the rocSPARSE library context.

====== ============================================
Input
====== ============================================
handle the handle to the rocSPARSE library context.
====== ============================================

=============================== ===============================================
Status Returned
=============================== ===============================================
rocsparse_status_success        the initialization succeeded.
rocsparse_status_internal_error an internal error occurred.
=============================== ===============================================

.. _rocsparse_set_stream:

rocsparse_set_stream()
***********************
.. code-block:: c

  rocsparse_status
  rocsparse_set_stream(rocsparse_handle handle,
                       hipStream_t stream);

*rocsparse_set_stream()* specifies the stream to be used by the rocSPARSE library context and all subsequent function calls.

====== =======================================================
Input
====== =======================================================
handle the handle to the rocSPARSE library context.
stream the stream to be used by the rocSPARSE library context.
====== =======================================================

================================ =============================
Status Returned
================================ =============================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_handle  the library context was not initialized.
================================ =============================

rocsparse_get_stream()
***********************
.. code-block:: c

  rocsparse_status
  rocsparse_get_stream(rocsparse_handle handle,
                       hipStream_t* stream);

*rocsparse_get_stream()* gets the rocSPARSE library context stream which is currently used for all subsequent function calls.

====== ============================================
Input
====== ============================================
handle the handle to the rocSPARSE library context.
====== ============================================

====== ===========================================================
Output
====== ===========================================================
stream the stream currently used by the rocSPARSE library context.
====== ===========================================================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_handle  the library context was not initialized.
================================ ===============================================

.. _rocsparse_set_pointer_mode:

rocsparse_set_pointer_mode()
*****************************
.. code-block:: c

  rocsparse_status
  rocsparse_set_pointer_mode(rocsparse_handle handle,
                             rocsparse_pointer_mode pointer_mode);

*rocsparse_set_pointer_mode()* specifies the pointer mode to be used by the rocSPARSE library context and all subsequent function calls. By default, all values are passed by reference on the host. Valid pointer modes are ``rocsparse_pointer_mode_host`` or ``rocsparse_pointer_mode_device``.

============ =============================================================
Input
============ =============================================================
handle       the handle to the rocSPARSE library context.
pointer_mode the pointer mode to be used by the rocSPARSE library context.
============ =============================================================

================================ =============================
Status Returned
================================ =============================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_handle  the library context was not initialized.
================================ =============================

.. _rocsparse_get_pointer_mode:

rocsparse_get_pointer_mode()
*****************************
.. code-block:: c

  rocsparse_status
  rocsparse_get_pointer_mode(rocsparse_handle handle,
                             rocsparse_pointer_mode* pointer_mode);

*rocsparse_get_pointer_mode()* gets the rocSPARSE library context pointer mode which is currently used for all subsequent function calls.

====== ============================================
Input
====== ============================================
handle the handle to the rocSPARSE library context.
====== ============================================

============ =========================================================================
Output
============ =========================================================================
pointer_mode the pointer mode that is currently used by the rocSPARSE library context.
============ =========================================================================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_handle  the library context was not initialized.
================================ ===============================================

rocsparse_get_version()
************************
.. code-block:: c

  rocsparse_status
  rocsparse_get_version(rocsparse_handle handle,
                        rocsparse_int* version);

*rocsparse_get_version()* gets the rocSPARSE library version number.

====== ============================================
Input
====== ============================================
handle the handle to the rocSPARSE library context.
====== ============================================

======= ============================================
Output
======= ============================================
version the version number of the rocSPARSE library.
======= ============================================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_handle  the library context was not initialized.
================================ ===============================================

.. _rocsparse_create_mat_descr:

rocsparse_create_mat_descr()
*****************************
.. code-block:: c

  rocsparse_status
  rocsparse_create_mat_descr(rocsparse_mat_descr* descr);

*rocsparse_create_mat_descr()* creates the matrix descriptor. It initializes ``rocsparse_matrix_type`` to ``rocsparse_matrix_type_general`` and ``rocsparse_index_base`` to ``rocsparse_index_base_zero``. It should be destroyed at the end using :ref:`rocsparse_destroy_mat_descr`.

====== =====================================
Output
====== =====================================
descr  the pointer to the matrix descriptor.
====== =====================================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_pointer ``descr`` pointer is invalid.
================================ ===============================================

.. _rocsparse_destroy_mat_descr:

rocsparse_destroy_mat_descr()
******************************
.. code-block:: c

  rocsparse_status
  rocsparse_destroy_mat_descr(rocsparse_mat_descr descr);

*rocsparse_destroy_mat_descr()* destroys the matrix descriptor.

===== ======================
Input
===== ======================
descr the matrix descriptor.
===== ======================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
================================ ===============================================

rocsparse_set_mat_index_base()
*******************************
.. code-block:: c

  rocsparse_status
  rocsparse_set_mat_index_base(rocsparse_mat_descr descr,
                               rocsparse_index_base base);

*rocsparse_set_mat_index_base()* sets the index base of the matrix descriptor. Valid index bases are ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.

===== ============================================================================================
Input
===== ============================================================================================
descr the matrix descriptor.
base  the matrix index base, can be ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.
===== ============================================================================================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_pointer ``descr`` pointer is invalid.
rocsparse_status_invalid_value   ``base`` is invalid.
================================ ===============================================

rocsparse_get_mat_index_base()
*******************************
.. code-block:: c

  rocsparse_index_base
  rocsparse_get_mat_index_base(const rocsparse_mat_descr descr);

*rocsparse_get_mat_index_base()* returns the index base of the matrix descriptor.

===== ======================
Input
===== ======================
descr the matrix descriptor.
===== ======================

rocsparse_set_mat_type()
*************************
.. code-block:: c

  rocsparse_status
  rocsparse_set_mat_type(rocsparse_mat_descr descr,
                         rocsparse_matrix_type type);

*rocsparse_set_mat_type()* sets the matrix type of the matrix descriptor. Valid matrix types are ``rocsparse_matrix_type_general``, ``rocsparse_matrix_type_symmetric``, ``rocsparse_matrix_type_hermitian`` or ``rocsparse_matrix_type_triangular``.

===== ==========================================================================
Input
===== ==========================================================================
descr the matrix descriptor.
type  the matrix type, can be ``rocsparse_matrix_type_general``, |br| ``rocsparse_matrix_type_symmetric``, ``rocsparse_matrix_type_hermitian`` or |br| ``rocsparse_matrix_type_triangular``.
===== ==========================================================================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_pointer ``descr`` pointer is invalid.
rocsparse_status_invalid_value   ``type`` is invalid.
================================ ===============================================

rocsparse_get_mat_type()
*************************
.. code-block:: c

  rocsparse_matrix_type
  rocsparse_get_mat_type(const rocsparse_mat_descr descr);

*rocsparse_get_mat_type()* returns the matrix type of the matrix descriptor.

===== ====================================
Input
===== ====================================
descr the matrix descriptor.
===== ====================================

.. _rocsparse_create_mat_info:

rocsparse_create_mat_info()
****************************
.. code-block:: c

  rocsparse_status
  rocsparse_create_mat_info(rocsparse_mat_info* info);

*rocsparse_create_mat_info()* creates a structure that holds the matrix info data that is gathered during the analysis routines available. It should be destroyed at the end using :ref:`rocsparse_destroy_mat_info`.

====== =================================
Output
====== =================================
info   the pointer to the info structure.
====== =================================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_pointer ``info`` pointer is invalid.
================================ ===============================================

.. _rocsparse_destroy_mat_info:

rocsparse_destroy_mat_info()
*****************************
.. code-block:: c

  rocsparse_status
  rocsparse_destroy_mat_info(rocsparse_mat_info info);

*rocsparse_destroy_mat_info()* destroys the info structure.

===== ===================
Input
===== ===================
info  the info structure.
===== ===================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_pointer ``info`` pointer is invalid.
rocsparse_status_internal_error  an internal error occurred.
================================ ===============================================

.. _rocsparse_create_hyb_mat:

rocsparse_create_hyb_mat()
***************************
.. code-block:: c

  rocsparse_status
  rocsparse_create_hyb_mat(rocsparse_hyb_mat* hyb);

*rocsparse_create_hyb_mat()* creates a structure that holds the matrix in HYB storage format. It should be destroyed at the end using :ref:`rocsparse_destroy_hyb_mat`.

====== =================================
Output
====== =================================
hyb    the pointer to the hybrid matrix.
====== =================================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_pointer ``hyb`` pointer is invalid.
================================ ===============================================

.. _rocsparse_destroy_hyb_mat:

rocsparse_destroy_hyb_mat()
****************************
.. code-block:: c

  rocsparse_status
  rocsparse_destroy_hyb_mat(rocsparse_hyb_mat hyb);

*rocsparse_destroy_hyb_mat()* destroys the HYB structure.

===== ==================
Input
===== ==================
hyb   the hybrid matrix.
===== ==================

================================ ===============================================
Status Returned
================================ ===============================================
rocsparse_status_success         the initialization succeeded.
rocsparse_status_invalid_pointer ``hyb`` pointer is invalid.
rocsparse_status_internal_error  an internal error occurred.
================================ ===============================================

.. _rocsparse_level1_functions:

Sparse Level 1 Functions
------------------------

This section describes all rocSPARSE level 1 sparse linear algebra functions.

rocsparse_axpyi()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_saxpyi(rocsparse_handle handle,
                   rocsparse_int nnz,
                   const float* alpha,
                   const float* x_val,
                   const rocsparse_int* x_ind,
                   float* y,
                   rocsparse_index_base idx_base);

  rocsparse_status
  rocsparse_daxpyi(rocsparse_handle handle,
                   rocsparse_int nnz,
                   const double* alpha,
                   const double* x_val,
                   const rocsparse_int* x_ind,
                   double* y,
                   rocsparse_index_base idx_base);

*rocsparse_axpyi()* multiplies the sparse vector :math:`x` with scalar :math:`\alpha` and adds the result to the dense vector :math:`y`, such that :math:`y := y + \alpha \cdot x`.

.. code-block:: c

  for(i = 0; i < nnz; ++i)
    y[x_ind[i] - idx_base] = y[x_ind[i] - idx_base] + alpha * x_val[i];

======== =============================================================================
Input
======== =============================================================================
handle   handle to the rocSPARSE library context queue.
nnz      number of non-zero entries of vector :math:`x`.
alpha    scalar :math:`\alpha`.
x_val    array of ``nnz`` elements containing the values of :math:`x`.
x_ind    array of ``nnz`` elements containing the indices of the non-zero values of :math:`x`.
y        array of values in dense format.
idx_base ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.
======== =============================================================================

====== ================================
Output
====== ================================
y      array of values in dense format.
====== ================================

================================ ====
Returned rocsparse_status
================================ ====
rocsparse_status_success         the operation completed successfully.
rocsparse_status_invalid_handle  the library context was not initialized.
rocsparse_status_invalid_value   ``idx_base`` is invalid.
rocsparse_status_invalid_size    ``nnz`` is invalid.
rocsparse_status_invalid_pointer ``alpha``, ``x_val``, ``x_ind`` or ``y`` pointer is invalid.
================================ ====

rocsparse_doti()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_sdoti(rocsparse_handle handle,
                  rocsparse_int nnz,
                  const float* x_val,
                  const rocsparse_int* x_ind,
                  const float* y,
                  float* result,
                  rocsparse_index_base idx_base);

  rocsparse_status
  rocsparse_ddoti(rocsparse_handle handle,
                  rocsparse_int nnz,
                  const double* x_val,
                  const rocsparse_int* x_ind,
                  const double* y,
                  double* result,
                  rocsparse_index_base idx_base);

*rocsparse_doti()* computes the dot product of the sparse vector :math:`x` with the dense vector :math:`y`, such that :math:`\text{result} := y^T x`.

.. code-block:: c

  for(i = 0; i < nnz; ++i)
    result = result + x_val[i] * y[x_ind[i] - idx_base];

======== =============================================================================
Input
======== =============================================================================
handle   handle to the rocSPARSE library context queue.
nnz      number of non-zero entries of vector :math:`x`.
x_val    array of ``nnz`` elements, containing the values of :math:`x`.
x_ind    array of ``nnz`` elements, containing the indices of the non-zero values of :math:`x`.
y        array of values in dense format.
idx_base ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.
======== =============================================================================

====== ====================================================
Output
====== ====================================================
result pointer to the ``result``, can be host or device memory.
====== ====================================================

================================ ====
Returned rocsparse_status
================================ ====
rocsparse_status_success         the operation completed successfully.
rocsparse_status_invalid_handle  the library context was not initialized.
rocsparse_status_invalid_value   ``idx_base`` is invalid.
rocsparse_status_invalid_size    ``nnz`` is invalid.
rocsparse_status_invalid_pointer ``x_val``, ``x_ind``, ``y`` or ``result`` pointer is invalid.
rocsparse_status_memory_error    the buffer for the dot product reduction could not be allocated.
rocsparse_status_internal_error  an internal error occurred.
================================ ====

rocsparse_gthr()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_sgthr(rocsparse_handle handle,
                  rocsparse_int nnz,
                  const float* y,
                  float* x_val,
                  const rocsparse_int* x_ind,
                  rocsparse_index_base idx_base);

  rocsparse_status
  rocsparse_dgthr(rocsparse_handle handle,
                  rocsparse_int nnz,
                  const double* y,
                  double* x_val,
                  const rocsparse_int* x_ind,
                  rocsparse_index_base idx_base);

*rocsparse_gthr()* gathers the elements that are listed in ``x_ind`` from the dense vector :math:`y` and stores them in the sparse vector :math:`x`.

.. code-block:: c

  for(i = 0; i < nnz; ++i)
    x_val[i] = y[x_ind[i] - idx_base];

======== =============================================================================
Input
======== =============================================================================
handle   handle to the rocSPARSE library context queue.
nnz      number of non-zero entries of vector :math:`x`.
y        array of values in dense format.
x_ind    array of ``nnz`` elements, containing the indices of the non-zero values of :math:`x`.
idx_base ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.
======== =============================================================================

====== =====================================================
Output
====== =====================================================
x_val  array of ``nnz`` elements containing the values of :math:`x`.
====== =====================================================

================================ ====
Returned rocsparse_status
================================ ====
rocsparse_status_success         the operation completed successfully.
rocsparse_status_invalid_handle  the library context was not initialized.
rocsparse_status_invalid_value   ``idx_base`` is invalid.
rocsparse_status_invalid_size    ``nnz`` is invalid.
rocsparse_status_invalid_pointer ``y``, ``x_val`` or ``x_ind`` pointer is invalid.
================================ ====

rocsparse_gthrz()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_sgthrz(rocsparse_handle handle,
                   rocsparse_int nnz,
                   const float* y,
                   float* x_val,
                   const rocsparse_int* x_ind,
                   rocsparse_index_base idx_base);

  rocsparse_status
  rocsparse_dgthrz(rocsparse_handle handle,
                   rocsparse_int nnz,
                   const double* y,
                   double* x_val,
                   const rocsparse_int* x_ind,
                   rocsparse_index_base idx_base);

*rocsparse_gthrz()* gathers the elements that are listed in ``x_ind`` from the dense vector :math:`y` and stores them in the sparse vector :math:`x`. The gathered elements in :math:`y` are replaced by zero.

.. code-block:: c

  for(i = 0; i < nnz; ++i)
  {
    x_val[i] = y[x_ind[i] - idx_base];
    y[x_ind[i] - idx_base] = 0;
  }

======== =============================================================================
Input
======== =============================================================================
handle   handle to the rocSPARSE library context queue.
nnz      number of non-zero entries of vector :math:`x`.
y        array of values in dense format.
x_ind    array of ``nnz`` elements, containing the indices of the non-zero values of :math:`x`.
idx_base ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.
======== =============================================================================

====== =====================================================
Output
====== =====================================================
x_val  array of ``nnz`` elements containing the values of :math:`x`.
====== =====================================================

================================ ====
Returned rocsparse_status
================================ ====
rocsparse_status_success         the operation completed successfully.
rocsparse_status_invalid_handle  the library context was not initialized.
rocsparse_status_invalid_value   ``idx_base`` is invalid.
rocsparse_status_invalid_size    ``nnz`` is invalid.
rocsparse_status_invalid_pointer ``y``, ``x_val`` or ``x_ind`` pointer is invalid.
================================ ====

rocsparse_roti()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_sroti(rocsparse_handle handle,
                  rocsparse_int nnz,
                  float* x_val,
                  const rocsparse_int* x_ind,
                  float* y,
                  const float* c,
                  const float* s,
                  rocsparse_index_base idx_base);

  rocsparse_status
  rocsparse_droti(rocsparse_handle handle,
                  rocsparse_int nnz,
                  double* x_val,
                  const rocsparse_int* x_ind,
                  double* y,
                  const double* c,
                  const double* s,
                  rocsparse_index_base idx_base);

*rocsparse_roti()* applies the Givens rotation matrix :math:`G` to the sparse vector :math:`x` and the dense vector :math:`y`, where

.. math::

  G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}.

.. code-block:: c

  for(i = 0; i < nnz; ++i)
  {
    double x = x_val[i];
    double y = y[x_ind[i] - idx_base];

    x_val[i]               = c * x + s * y;
    y[x_ind[i] - idx_base] = c * y - s * x;
  }

======== =============================================================================
Input
======== =============================================================================
handle   handle to the rocSPARSE library context queue.
nnz      number of non-zero entries of vector :math:`x`.
x_val    array of ``nnz`` elements, containing the values of :math:`x`.
x_ind    array of ``nnz`` elements, containing the indices of the non-zero values of :math:`x`.
y        array of values in dense format.
c        pointer to the cosine element of :math:`G`, can be on host or device.
s        pointer to the sine element of :math:`G`, can be on host or device.
idx_base ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.
======== =============================================================================

====== =====================================================
Output
====== =====================================================
x_val  array of ``nnz`` elements, containing the values of :math:`x`.
y      array of values in dense format.
====== =====================================================

================================ ====
Returned rocsparse_status
================================ ====
rocsparse_status_success         the operation completed successfully.
rocsparse_status_invalid_handle  the library context was not initialized.
rocsparse_status_invalid_value   ``idx_base`` is invalid.
rocsparse_status_invalid_size    ``nnz`` is invalid.
rocsparse_status_invalid_pointer ``c``, ``s``, ``x_val``, ``x_ind`` or ``y`` pointer is invalid.
================================ ====

rocsparse_sctr()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_ssctr(rocsparse_handle handle,
                  rocsparse_int nnz,
                  const float* x_val,
                  const rocsparse_int* x_ind,
                  float* y,
                  rocsparse_index_base idx_base);

  rocsparse_status
  rocsparse_dsctr(rocsparse_handle handle,
                  rocsparse_int nnz,
                  const double* x_val,
                  const rocsparse_int* x_ind,
                  double* y,
                  rocsparse_index_base idx_base);

*rocsparse_sctr()* scatters the elements that are listed in ``x_ind`` from the sparse vector :math:`x` into the dense vector :math:`y`. Entries of :math:`y` that are not listed in ``x_ind`` remain unchanged.

.. code-block:: c

  for(i = 0; i < nnz; ++i)
    y[x_ind[i] - idx_base] = x_val[i];

======== =============================================================================
Input
======== =============================================================================
handle   handle to the rocSPARSE library context queue.
nnz      number of non-zero entries of vector :math:`x`.
x_val    array of ``nnz`` elements, containing the values of :math:`x`.
x_ind    array of ``nnz`` elements, containing the indices of the non-zero values of :math:`x`.
y        array of values in dense format.
idx_base ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.
======== =============================================================================

====== =====================================================
Output
====== =====================================================
y      array of values in dense format.
====== =====================================================

================================ ====
Returned rocsparse_status
================================ ====
rocsparse_status_success         the operation completed successfully.
rocsparse_status_invalid_handle  the library context was not initialized.
rocsparse_status_invalid_value   ``idx_base`` is invalid.
rocsparse_status_invalid_size    ``nnz`` is invalid.
rocsparse_status_invalid_pointer ``x_val``, ``x_ind`` or ``y`` pointer is invalid.
================================ ====

.. _rocsparse_level2_functions:

Sparse Level 2 Functions
------------------------

This section describes all rocSPARSE level 2 sparse linear algebra functions.

rocsparse_coomv()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_scoomv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   rocsparse_int m,
                   rocsparse_int n,
                   rocsparse_int nnz,
                   const float* alpha,
                   const rocsparse_mat_descr descr,
                   const float* coo_val,
                   const rocsparse_int* coo_row_ind,
                   const rocsparse_int* coo_col_ind,
                   const float* x,
                   const float* beta,
                   float* y);

  rocsparse_status
  rocsparse_dcoomv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   rocsparse_int m,
                   rocsparse_int n,
                   rocsparse_int nnz,
                   const double* alpha,
                   const rocsparse_mat_descr descr,
                   const double* coo_val,
                   const rocsparse_int* coo_row_ind,
                   const rocsparse_int* coo_col_ind,
                   const double* x,
                   const double* beta,
                   double* y);

*rocsparse_coomv()* multiplies the scalar :math:`\alpha` with a sparse :math:`m \times n` matrix, defined in COO storage format, and the dense vector :math:`x` and adds the result to the dense vector :math:`y` that is multiplied by the scalar :math:`\beta`, such that :math:`y := \alpha \cdot op(A) \cdot x + \beta \cdot y` with

.. math::

  op(A) = \Bigg\{
            \begin{array}{ll}
              A, & \text{if trans == rocsparse_operation_none} \\
              A^T, & \text{if trans == rocsparse_operation_transpose} \\
              A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
            \end{array}.

Currently, only ``trans == rocsparse_operation_none`` is supported.

The COO matrix has to be sorted by row indices. This can be achieved by using :ref:`rocsparse_coosort`.

.. code-block:: c

  for(i = 0; i < m; ++i)
    y[i] = beta * y[i];

  for(i = 0; i < nnz; ++i)
    y[coo_row_ind[i]] += alpha * coo_val[i] * x[coo_col_ind[i]];

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
trans       matrix operation type.
m           number of rows of the sparse COO matrix.
n           number of columns of the sparse COO matrix.
nnz         number of non-zero entries of the sparse COO matrix.
alpha       scalar :math:`\alpha`.
descr       descriptor of the sparse COO matrix. |br| Currently, *rocsparse_coomv()* supports only ``rocsparse_matrix_type_general``.
coo_val     array of ``nnz`` elements of the sparse COO matrix.
coo_row_ind array of ``nnz`` elements containing the row indices of the sparse COO matrix.
coo_col_ind array of ``nnz`` elements containing the column indices of the sparse COO matrix.
x           array of ``n`` elements (:math:`op(A) == A`) or |br| ``m`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
beta        scalar :math:`\beta`.
y           array of ``m`` elements (:math:`op(A) == A`) or |br| ``n`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
=========== =============================================================================

====== ================================
Output
====== ================================
y      array of ``m`` elements (:math:`op(A) == A`) or |br| ``n`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
====== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``alpha``, ``coo_val``, ``coo_row_ind``, ``coo_col_ind``, |br| ``x``, ``beta`` or ``y`` pointer is invalid.
rocsparse_status_arch_mismatch    the device is not supported by *rocsparse_coomv()*.
rocsparse_status_not_implemented  ``trans != rocsparse_operation_none`` or |br| ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_csrmv_analysis:

rocsparse_csrmv_analysis()
***************************
.. code-block:: c

  rocsparse_status
  rocsparse_csrmv_analysis(rocsparse_handle handle,
                           rocsparse_operation trans,
                           rocsparse_int m,
                           rocsparse_int n,
                           rocsparse_int nnz,
                           const rocsparse_mat_descr descr,
                           const rocsparse_int* csr_row_ptr,
                           const rocsparse_int* csr_col_ind,
                           rocsparse_mat_info info);

*rocsparse_csrmv_analysis()* performs the analysis step for :ref:`rocsparse_csrmv`. It is expected that this function will be executed only once for a given matrix and particular operation type. Note that if the matrix sparsity pattern changes, the gathered information will become invalid.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
trans       matrix operation type.
m           number of rows of the sparse CSR matrix.
n           number of columns of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
descr       descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_csrmv_analysis()* supports only ``rocsparse_matrix_type_general``.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
=========== =============================================================================

====== ================================
Output
====== ================================
info   structure that holds the information collected during analysis step.
====== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``csr_row_ptr``, ``csr_col_ind`` or ``info`` pointer is invalid.
rocsparse_status_memory_error     the buffer for the gathered information could not be allocated.
rocsparse_status_internal_error   an internal error occurred.
rocsparse_status_not_implemented  ``trans != rocsparse_operation_none`` or |br| ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_csrmv_analysis_clear:

rocsparse_csrmv_analysis_clear()
*********************************
.. code-block:: c

  rocsparse_status
  rocsparse_csrmv_analysis_clear(rocsparse_handle handle,
                                 rocsparse_mat_info info);

*rocsparse_csrmv_analysis_clear()* deallocates all memory that was allocated by :ref:`rocsparse_csrmv_analysis`. This is especially useful, if memory is an issue and the analysis data is not required anymore for further computation, e.g. when switching to another sparse matrix format. Calling *rocsparse_csrmv_analysis_clear()* is optional. All allocated resources will be cleared when the opaque :ref:`rocsparse_mat_info` struct is destroyed using :ref:`rocsparse_destroy_mat_info`.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
info        structure that holds the information collected during analysis step.
=========== =============================================================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_pointer  ``info`` pointer is invalid.
rocsparse_status_memory_error     the buffer for the information could not be deallocated.
rocsparse_status_internal_error   an internal error occurred.
================================  ====

.. _rocsparse_csrmv:

rocsparse_csrmv()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_scsrmv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   rocsparse_int m,
                   rocsparse_int n,
                   rocsparse_int nnz,
                   const float* alpha,
                   const rocsparse_mat_descr descr,
                   const float* csr_val,
                   const rocsparse_int* csr_row_ptr,
                   const rocsparse_int* csr_col_ind,
                   const float* x,
                   const float* beta,
                   float* y,
                   const rocsparse_mat_info info);

  rocsparse_status
  rocsparse_dcsrmv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   rocsparse_int m,
                   rocsparse_int n,
                   rocsparse_int nnz,
                   const double* alpha,
                   const rocsparse_mat_descr descr,
                   const double* csr_val,
                   const rocsparse_int* csr_row_ptr,
                   const rocsparse_int* csr_col_ind,
                   const double* x,
                   const double* beta,
                   double* y,
                   const rocsparse_mat_info info);

*rocsparse_csrmv()* multiplies the scalar :math:`\alpha` with a sparse :math:`m \times n` matrix, defined in CSR storage format, and the dense vector :math:`x` and adds the result to the dense vector :math:`y` that is multiplied by the scalar :math:`\beta`, such that :math:`y := \alpha \cdot op(A) \cdot x + \beta \cdot y` with

.. math::

  op(A) = \Bigg\{
            \begin{array}{ll}
              A, & \text{if trans == rocsparse_operation_none} \\
              A^T, & \text{if trans == rocsparse_operation_transpose} \\
              A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
            \end{array}.

The ``info`` parameter is optional and contains information collected by :ref:`rocsparse_csrmv_analysis`. If present, the information will be used to speed up the *csrmv* computation. If ``info == NULL``, general *csrmv* routine will be used instead.

Currently, only ``trans == rocsparse_operation_none`` is supported.

.. code-block:: c

  for(i = 0; i < m; ++i)
  {
    y[i] = beta * y[i];
    for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
      y[i] = y[i] + alpha * csr_val[j] * x[csr_col_ind[j]];
  }

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
trans       matrix operation type.
m           number of rows of the sparse CSR matrix.
n           number of columns of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
alpha       scalar :math:`\alpha`.
descr       descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_csrmv()* supports only ``rocsparse_matrix_type_general``.
csr_val     array of ``nnz`` elements of the sparse CSR matrix.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
x           array of ``n`` elements (:math:`op(A) == A`) or |br| ``m`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
beta        scalar :math:`\beta`.
y           array of ``m`` elements (:math:`op(A) == A`) or |br| ``n`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
info        information collected by :ref:`rocsparse_csrmv_analysis`, |br| can be ``NULL`` if no information is available.
=========== =============================================================================

====== ================================
Output
====== ================================
y      array of ``m`` elements (:math:`op(A) == A`) or |br| ``n`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
====== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``alpha``, ``csr_val``, ``csr_row_ptr``, ``csr_col_ind``, |br| ``x``, ``beta`` or ``y`` pointer is invalid.
rocsparse_status_arch_mismatch    the device is not supported by *rocsparse_csrmv()*.
rocsparse_status_memory_error     the buffer for the segmented reduction could not be allocated.
rocsparse_status_internal_error   an internal error occurred.
rocsparse_status_not_implemented  ``trans != rocsparse_operation_none`` or |br| ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

rocsparse_ellmv()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_sellmv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   rocsparse_int m,
                   rocsparse_int n,
                   const float* alpha,
                   const rocsparse_mat_descr descr,
                   const float* ell_val,
                   const rocsparse_int* ell_col_ind,
                   rocsparse_int ell_width,
                   const float* x,
                   const float* beta,
                   float* y);

  rocsparse_status
  rocsparse_dellmv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   rocsparse_int m,
                   rocsparse_int n,
                   const double* alpha,
                   const rocsparse_mat_descr descr,
                   const double* ell_val,
                   const rocsparse_int* ell_col_ind,
                   rocsparse_int ell_width,
                   const double* x,
                   const double* beta,
                   double* y);

*rocsparse_ellmv()* multiplies the scalar :math:`\alpha` with a sparse :math:`m \times n` matrix, defined in ELL storage format, and the dense vector :math:`x` and adds the result to the dense vector :math:`y` that is multiplied by the scalar :math:`\beta`, such that :math:`y := \alpha \cdot op(A) \cdot x + \beta \cdot y` with

.. math::

  op(A) = \Bigg\{
            \begin{array}{ll}
              A, & \text{if trans == rocsparse_operation_none} \\
              A^T, & \text{if trans == rocsparse_operation_transpose} \\
              A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
            \end{array}.

Currently, only ``trans == rocsparse_operation_none`` is supported.

.. code-block:: c

  for(i = 0; i < m; ++i)
  {
    y[i] = beta * y[i];
    for(p = 0; p < ell_width; ++p)
      if((ell_col_ind[p * m + i] >= 0) && (ell_col_ind[p * m + i] < n))
        y[i] = y[i] + alpha * ell_val[p * m + i] * x[ell_col_ind[p * m + i]];
  }

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
trans       matrix operation type.
m           number of rows of the sparse ELL matrix.
n           number of columns of the sparse ELL matrix.
alpha       scalar :math:`\alpha`.
descr       descriptor of the sparse ELL matrix. |br| Currently, *rocsparse_ellmv()* supports only ``rocsparse_matrix_type_general``.
ell_val     array that contains the elements of the sparse ELL matrix. |br| Padded elements should be zero.
ell_col_ind array that contains the column indices of the sparse ELL matrix. |br| Padded column indices should be -1.
ell_width   number of non-zero elements per row of the sparse ELL matrix.
x           array of ``n`` elements (:math:`op(A) == A`) or |br| ``m`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
beta        scalar :math:`\beta`.
y           array of ``m`` elements (:math:`op(A) == A`) or |br| ``n`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
=========== =============================================================================

====== ================================
Output
====== ================================
y      array of ``m`` elements (:math:`op(A) == A`) or |br| ``n`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
====== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``ell_width`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``alpha``, ``ell_val``, ``ell_col_ind``, |br| ``x``, ``beta`` or ``y`` pointer is invalid.
rocsparse_status_not_implemented  ``trans != rocsparse_operation_none`` or |br| ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

rocsparse_hybmv()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_shybmv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   const float* alpha,
                   const rocsparse_mat_descr descr,
                   const rocsparse_hyb_mat hyb,
                   const float* x,
                   const float* beta,
                   float* y);

  rocsparse_status
  rocsparse_dhybmv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   const double* alpha,
                   const rocsparse_mat_descr descr,
                   const rocsparse_hyb_mat hyb,
                   const double* x,
                   const double* beta,
                   double* y);


*rocsparse_hybmv()* multiplies the scalar :math:`\alpha` with a sparse :math:`m \times n` matrix, defined in HYB storage format, and the dense vector :math:`x` and adds the result to the dense vector :math:`y` that is multiplied by the scalar :math:`\beta`, such that :math:`y := \alpha \cdot op(A) \cdot x + \beta \cdot y` with

.. math::

  op(A) = \Bigg\{
            \begin{array}{ll}
              A, & \text{if trans == rocsparse_operation_none} \\
              A^T, & \text{if trans == rocsparse_operation_transpose} \\
              A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
            \end{array}.

Currently, only ``trans == rocsparse_operation_none`` is supported.

.. code-block:: c

  // ellmv on the ELL matrix part
  ellmv(hyb->ell);
  // coomv on the COO matrix part
  coomv(hyb->coo);

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
trans       matrix operation type.
alpha       scalar :math:`\alpha`.
descr       descriptor of the sparse HYB matrix. |br| Currently, *rocsparse_hybmv()* supports only ``rocsparse_matrix_type_general``.
hyb         matrix in HYB storage format.
x           array of ``n`` elements (:math:`op(A) == A`) or |br| ``m`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
beta        scalar :math:`\beta`.
y           array of ``m`` elements (:math:`op(A) == A`) or |br| ``n`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
=========== =============================================================================

====== ================================
Output
====== ================================
y      array of ``m`` elements (:math:`op(A) == A`) or |br| ``n`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
====== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``hyb`` structure was not initialized with valid matrix sizes.
rocsparse_status_invalid_pointer  ``descr``, ``alpha``, ``hyb``, ``x``, ``beta`` or ``y`` pointer is invalid.
rocsparse_status_invalid_value    ``hyb`` structure was not initialized with a valid partitioning type.
rocsparse_status_arch_mismatch    the device is not supported by *rocsparse_hybmv()*.
rocsparse_status_memory_error     the buffer for *rocsparse_hybmv()* could not be allocated.
rocsparse_status_internal_error   an internal error occurred.
rocsparse_status_not_implemented  ``trans != rocsparse_operation_none`` or |br| ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_csrsv_buffer_size:

rocsparse_csrsv_buffer_size()
******************************
.. code-block:: c

  rocsparse_status
  rocsparse_csrsv_buffer_size(rocsparse_handle handle,
                              rocsparse_operation trans,
                              rocsparse_int m,
                              rocsparse_int nnz,
                              const rocsparse_mat_descr descr,
                              const rocsparse_int* csr_row_ptr,
                              const rocsparse_int* csr_col_ind,
                              rocsparse_mat_info info,
                              size_t* buffer_size);

*rocsparse_csrsv_buffer_size()* returns the size of the temporary storage buffer required by :ref:`rocsparse_csrsv_analysis`. The temporary storage buffer must be allocated by the user.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
trans       matrix operation type.
m           number of rows of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
descr       descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_csrsv_buffer_size()* supports only ``rocsparse_matrix_type_general``.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
info        structure that holds the information collected during analysis step.
=========== =============================================================================

=========== ================================
Output
=========== ================================
buffer_size number of bytes of the temporary storage buffer required by :ref:`rocsparse_csrsv_analysis`.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``csr_row_ptr``, ``csr_col_ind``, ``info`` |br| or ``buffer_size`` pointer is invalid.
rocsparse_status_internal_error   an internal error occurred.
rocsparse_status_not_implemented  ``trans != rocsparse_operation_none`` or |br| ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_csrsv_analysis:

rocsparse_csrsv_analysis()
***************************
.. code-block:: c

  rocsparse_status
  rocsparse_csrsv_analysis(rocsparse_handle handle,
                           rocsparse_operation trans,
                           rocsparse_int m,
                           rocsparse_int nnz,
                           const rocsparse_mat_descr descr,
                           const rocsparse_int* csr_row_ptr,
                           const rocsparse_int* csr_col_ind,
                           rocsparse_mat_info info,
                           void* temp_buffer);

*rocsparse_csrsv_analysis()* performs the analysis step for :ref:`rocsparse_csrsv`, :ref:`rocsparse_csric0` and/or :ref:`rocsparse_csrilu0`. It is expected that this function will be executed only once for a given matrix and particular operation type. Note that if the matrix sparsity pattern changes, the gathered information will become invalid.
*rocsparse_csrsv_analysis()* requires extra temporary storage buffer that has to be allocated by the user. Storage buffer size can be determined by :ref:`rocsparse_csrsv_buffer_size`.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
trans       matrix operation type.
m           number of rows of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
descr       descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_csrsv_analysis()* supports only ``rocsparse_matrix_type_general``.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
temp_buffer temporary storage buffer allocated by the user, |br| size is returned by :ref:`rocsparse_csrsv_buffer_size`.
=========== =============================================================================

====== ================================
Output
====== ================================
info   structure that holds the information collected during analysis step.
====== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``csr_row_ptr``, ``csr_col_ind``, ``info`` |br| or ``temp_buffer`` pointer is invalid.
rocsparse_status_internal_error   an internal error occurred.
rocsparse_status_not_implemented  ``trans != rocsparse_operation_none`` or |br| ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_csrsv_analysis_clear:

rocsparse_csrsv_analysis_clear()
*********************************
.. code-block:: c

  rocsparse_status
  rocsparse_csrsv_analysis_clear(rocsparse_handle handle,
                                 rocsparse_mat_info info);

*rocsparse_csrsv_analysis_clear()* deallocates all memory that was allocated by :ref:`rocsparse_csrsv_analysis`. This is especially useful, if memory is an issue and the analysis data is not required anymore for further computation, e.g. when switching to another sparse matrix format. Calling *rocsparse_csrmv_analysis_clear()* is optional. All allocated resources will be cleared when the opaque :ref:`rocsparse_mat_info` struct is destroyed using :ref:`rocsparse_destroy_mat_info`.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
info        structure that holds the information collected during analysis step.
=========== =============================================================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_pointer  ``info`` pointer is invalid.
rocsparse_status_memory_error     the buffer for the information could not be deallocated.
rocsparse_status_internal_error   an internal error occurred.
================================  ====

.. _rocsparse_csrsv:

rocsparse_csrsv()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_scsrsv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   rocsparse_int m,
                   rocsparse_int nnz,
                   const float* alpha,
                   const rocsparse_mat_descr descr,
                   const float* csr_val,
                   const rocsparse_int* csr_row_ptr,
                   const rocsparse_int* csr_col_ind,
                   const rocsparse_mat_info info,
                   const float* x,
                   float* y,
                   rocsparse_solve_policy policy,
                   void* temp_buffer);

  rocsparse_status
  rocsparse_dcsrsv(rocsparse_handle handle,
                   rocsparse_operation trans,
                   rocsparse_int m,
                   rocsparse_int nnz,
                   const double* alpha,
                   const rocsparse_mat_descr descr,
                   const double* csr_val,
                   const rocsparse_int* csr_row_ptr,
                   const rocsparse_int* csr_col_ind,
                   const rocsparse_mat_info info,
                   const float* x,
                   float* y,
                   rocsparse_solve_policy policy,
                   void* temp_buffer);

*rocsparse_csrsv()* solves a sparse triangular linear system with a sparse :math:`m \times n` matrix, defined in CSR storage format, the dense vector :math:`y`, the scalar :math:`\alpha` and the right-hand side vector :math:`x` such that :math:`op(A) \cdot y = \alpha \cdot x` with

.. math::

  op(A) = \Bigg\{
            \begin{array}{ll}
              A, & \text{if trans == rocsparse_operation_none} \\
              A^T, & \text{if trans == rocsparse_operation_transpose} \\
              A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
            \end{array}.

Currently, only ``trans == rocsparse_operation_none`` is supported.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
trans       matrix operation type.
m           number of rows of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
alpha       scalar :math:`\alpha`.
descr       descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_csrmv()* supports only ``rocsparse_matrix_type_general``.
csr_val     array of ``nnz`` elements of the sparse CSR matrix.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
info        information collected by :ref:`rocsparse_csrsv_analysis`.
x           array of ``n`` elements (:math:`op(A) == A`) or |br| ``m`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
policy      ``rocsparse_solve_policy_no_level`` or ``rocsparse_solve_policy_use_level``.
temp_buffer temporary storage buffer allocated by the user, |br| size is returned by :ref:`rocsparse_csrsv_buffer_size`.
=========== =============================================================================

====== ================================
Output
====== ================================
y      array of ``m`` elements (:math:`op(A) == A`) or |br| ``n`` elements (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
====== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``alpha``, ``csr_val``, ``csr_row_ptr``, ``csr_col_ind``, |br| ``x`` or ``y`` pointer is invalid.
rocsparse_status_arch_mismatch    the device is not supported by *rocsparse_csrsv()*.
rocsparse_status_internal_error   an internal error occurred.
rocsparse_status_not_implemented  ``trans != rocsparse_operation_none`` or |br| ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_level3_functions:

Sparse Level 3 Functions
------------------------

This section describes all rocSPARSE level 3 sparse linear algebra functions.

rocsparse_csrmm()
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_scsrmm(rocsparse_handle handle,
                   rocsparse_operation trans_A,
                   rocsparse_operation trans_B,
                   rocsparse_int m,
                   rocsparse_int n,
                   rocsparse_int k,
                   rocsparse_int nnz,
                   const float* alpha,
                   const rocsparse_mat_descr descr,
                   const float* csr_val,
                   const rocsparse_int* csr_row_ptr,
                   const rocsparse_int* csr_col_ind,
                   const float* B,
                   rocsparse_int ldb,
                   const float* beta,
                   float* C,
                   rocsparse_int ldc);

  rocsparse_status
  rocsparse_dcsrmm(rocsparse_handle handle,
                   rocsparse_operation trans_A,
                   rocsparse_operation trans_B,
                   rocsparse_int m,
                   rocsparse_int n,
                   rocsparse_int k,
                   rocsparse_int nnz,
                   const double* alpha,
                   const rocsparse_mat_descr descr,
                   const double* csr_val,
                   const rocsparse_int* csr_row_ptr,
                   const rocsparse_int* csr_col_ind,
                   const double* B,
                   rocsparse_int ldb,
                   const double* beta,
                   double* C,
                   rocsparse_int ldc);

*rocsparse_csrmm()* multiplies the scalar :math:`\alpha` with a sparse :math:`m \times k` matrix :math:`A`, defined in CSR storage format, and the dense :math:`k \times n` matrix :math:`B` and adds the result to the dense :math:`m \times n` matrix :math:`C` that is multiplied by the scalar :math:`\beta`, such that :math:`C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C` with

.. math::

  op(A) = \Bigg\{
            \begin{array}{ll}
              A, & \text{if trans_A == rocsparse_operation_none} \\
              A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
              A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
            \end{array}

and 

.. math::

  op(B) = \Bigg\{
            \begin{array}{ll}
              B, & \text{if trans_B == rocsparse_operation_none} \\
              B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
              B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
            \end{array}.

Currently, only ``trans_A == rocsparse_operation_none`` is supported.

.. code-block:: c

  for(i = 0; i < ldc; ++i)
    for(j = 0; j < n; ++j)
    {
      C[i][j] = beta * C[i][j];
      for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
        C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
    }

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
trans_A     matrix :math:`A` operation type.
trans_B     matrix :math:`B` operation type.
m           number of rows of the sparse CSR matrix :math:`A`.
n           number of columns of the dense matrix :math:`op(B)` and :math:`C`.
k           number of columns of the sparse CSR matrix :math:`A`.
nnz         number of non-zero entries of the sparse CSR matrix :math:`A`.
alpha       scalar :math:`\alpha`.
descr       descriptor of the sparse CSR matrix :math:`A`. |br| Currently, *rocsparse_csrmm()* supports only ``rocsparse_matrix_type_general``.
csr_val     array of ``nnz`` elements of the sparse CSR matrix :math:`A`.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix :math:`A`.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix :math:`A`.
B           array of dimension :math:`ldb \times n` (:math:`op(B) == B`) or |br| :math:`ldb \times k` (:math:`op(B) == B^T` or :math:`op(B) == B^H`).
ldb         leading dimension of :math:`B`, must be at least :math:`\max{(1, k)}` (:math:`op(A) == A`) or |br| :math:`\max{(1, m)}` (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
beta        scalar :math:`\beta`.
C           array of dimension :math:`ldc \times n`.
ldc         leading dimension of :math:`C`, must be at least :math:`\max{(1, m)}` (:math:`op(A) == A`) or |br| :math:`\max{(1, k)}` (:math:`op(A) == A^T` or :math:`op(A) == A^H`).
=========== =============================================================================

====== ================================
Output
====== ================================
C      array of dimension :math:`ldc \times n`.
====== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n``, ``k``, ``nnz``, ``ldb`` or ``ldc`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``alpha``, ``csr_val``, ``csr_row_ptr``, ``csr_col_ind``, |br| ``B``, ``beta`` or ``C`` pointer is invalid.
rocsparse_status_arch_mismatch    the device is not supported by *rocsparse_csrmm()*.
rocsparse_status_not_implemented  ``trans_A != rocsparse_operation_none`` or |br| ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_precond_functions:

Preconditioner Functions
------------------------

.. _rocsparse_csric0:

rocsparse_csric0
*****************

.. _rocsparse_csrilu0:

rocsparse_csrilu0
******************

.. _rocsparse_conversion_functions:

Sparse Conversion Functions
---------------------------

This section describes all rocSPARSE conversion functions.

rocsparse_csr2coo
******************
.. code-block:: c

  rocsparse_status
  rocsparse_csr2coo(rocsparse_handle handle,
                    const rocsparse_int* csr_row_ptr,
                    rocsparse_int nnz,
                    rocsparse_int m,
                    rocsparse_int* coo_row_ind,
                    rocsparse_index_base idx_base);

*rocsparse_csr2coo()* converts the CSR array containing the row offsets, that point to the start of every row, into a COO array of row indices. It can also be used, to convert a CSC array containing the column offsets into a COO array of column indices.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
m           number of rows of the sparse CSR matrix.
idx_base    ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.
=========== =============================================================================

=========== ================================
Output
=========== ================================
coo_row_ind array of ``nnz`` elements containing the row indices of the sparse COO matrix.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``csr_row_ptr`` or ``coo_row_ind`` pointer is invalid.
rocsparse_status_arch_mismatch    the device is not supported by *rocsparse_csr2coo()*.
================================  ====

rocsparse_coo2csr
******************
.. code-block:: c

  rocsparse_status
  rocsparse_coo2csr(rocsparse_handle handle,
                    const rocsparse_int* coo_row_ind,
                    rocsparse_int nnz,
                    rocsparse_int m,
                    rocsparse_int* csr_row_ptr,
                    rocsparse_index_base idx_base);

*rocsparse_coo2csr()* converts the COO array containing the row indices into a CSR array of row offsets, that point to the start of every row. It can also be used, to convert a COO array containing the column indices into a CSC array of column offsets, that point to the start of every column.

It is assumed that the COO row index array is sorted.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
coo_row_ind array of ``nnz`` elements containing the row indices of the sparse COO matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
m           number of rows of the sparse CSR matrix.
idx_base    ``rocsparse_index_base_zero`` or ``rocsparse_index_base_one``.
=========== =============================================================================

=========== ================================
Output
=========== ================================
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``coo_row_ind`` or ``csr_row_ptr`` pointer is invalid.
================================  ====

.. _rocsparse_csr2csc_buffer_size:

rocsparse_csr2csc_buffer_size
******************************
.. code-block:: c

  rocsparse_status
  rocsparse_csr2csc_buffer_size(rocsparse_handle handle,
                                rocsparse_int m,
                                rocsparse_int n,
                                rocsparse_int nnz,
                                const rocsparse_int* csr_row_ptr,
                                const rocsparse_int* csr_col_ind,
                                rocsparse_action copy_values,
                                size_t* buffer_size);

*rocsparse_csr2csc_buffer_size()* returns the size of the temporary storage buffer required by :ref:`rocsparse_csr2csc`. The temporary storage buffer must be allocated by the user.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse CSR matrix.
n           number of columns of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
copy_values ``rocsparse_action_numeric`` or ``rocsparse_action_symbolic``.
=========== =============================================================================

=========== ================================
Output
=========== ================================
buffer_size number of bytes of the temporary storage buffer required by :ref:`rocsparse_csr2csc`.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``csr_row_ptr``, ``csr_col_ind`` or ``buffer_size`` pointer is invalid.
rocsparse_status_internal_error   an internal error occurred.
================================  ====

.. _rocsparse_csr2csc:

rocsparse_csr2csc
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_scsr2csc(rocsparse_handle handle,
                     rocsparse_int m,
                     rocsparse_int n,
                     rocsparse_int nnz,
                     const float* csr_val,
                     const rocsparse_int* csr_row_ptr,
                     const rocsparse_int* csr_col_ind,
                     float* csc_val,
                     rocsparse_int* csc_row_ind,
                     rocsparse_int* csc_col_ptr,
                     rocsparse_action copy_values,
                     void* temp_buffer);

  rocsparse_status
  rocsparse_dcsr2csc(rocsparse_handle handle,
                     rocsparse_int m,
                     rocsparse_int n,
                     rocsparse_int nnz,
                     const double* csr_val,
                     const rocsparse_int* csr_row_ptr,
                     const rocsparse_int* csr_col_ind,
                     double* csc_val,
                     rocsparse_int* csc_row_ind,
                     rocsparse_int* csc_col_ptr,
                     rocsparse_action copy_values,
                     void* temp_buffer);

*rocsparse_csr2csc()* converts a CSR matrix info a CSC matrix. The resulting matrix can also be seen as the transpose of the input matrix. *rocsparse_csr2csc()* can also be used to convert a CSC matrix into a CSR matrix. ``copy_values`` decides whether ``csc_val`` is being filled during conversion (``rocsparse_action_numeric``) or not (``rocsparse_action_symbolic``).

*rocsparse_csr2csc()* requires extra temporary storage buffer that has to be allocated by the user. Storage buffer size can be determined by :ref:`rocsparse_csr2csc_buffer_size`.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse CSR matrix.
n           number of columns of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
csr_val     array of ``nnz`` elements of the sparse CSR matrix.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
copy_values ``rocsparse_action_numeric`` or ``rocsparse_action_symbolic``.
temp_buffer temporary storage buffer allocated by the user, |br| size is returned by :ref:`rocsparse_csr2csc_buffer_size`.
=========== =============================================================================

=========== ================================
Output
=========== ================================
csc_val     array of ``nnz`` elements of the sparse CSC matrix.
csc_row_ind array of ``nnz`` elements containing the row indices of the sparse CSC matrix.
csc_col_ptr array of ``n+1`` elements that point to the start of every column of the sparse CSC matrix.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``csr_val``, ``csr_row_ptr``, ``csr_col_ind``, ``csc_val``, |br| ``csc_row_ind``, ``csc_col_ptr`` or ``temp_buffer`` pointer is invalid.
rocsparse_status_arch_mismatch    the device is not supported by *rocsparse_csr2csc()*.
rocsparse_status_internal_error   an internal error occurred.
================================  ====

.. _rocsparse_csr2ell_width:

rocsparse_csr2ell_width
************************
.. code-block:: c

  rocsparse_status
  rocsparse_csr2ell_width(rocsparse_handle handle,
                          rocsparse_int m,
                          const rocsparse_mat_descr csr_descr,
                          const rocsparse_int* csr_row_ptr,
                          const rocsparse_mat_descr ell_descr,
                          rocsparse_int* ell_width);

*rocsparse_csr2ell_width()* computes the maximum of the per row non-zero elements over all rows, the ELL width, for a given CSR matrix.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse CSR matrix.
csr_descr   descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_csr2ell_width()* supports only ``rocsparse_matrix_type_general``.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
ell_descr   descriptor of the sparse ELL matrix. |br| Currently, *rocsparse_csr2ell_width()* supports only ``rocsparse_matrix_type_general``.
=========== =============================================================================

=========== ================================
Output
=========== ================================
ell_width   pointer to the number of non-zero elements per row in ELL storage format.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m`` is invalid.
rocsparse_status_invalid_pointer  ``csr_descr``, ``csr_row_ptr``, |br| ``ell_descr`` or ``ell_width`` pointer is invalid.
rocsparse_status_internal_error   an internal error occurred.
rocsparse_status_not_implemented  ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_csr2ell:

rocsparse_csr2ell
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_scsr2ell(rocsparse_handle handle,
                     rocsparse_int m,
                     const rocsparse_mat_descr csr_descr,
                     const float* csr_val,
                     const rocsparse_int* csr_row_ptr,
                     const rocsparse_int* csr_col_ind,
                     const rocsparse_mat_descr ell_descr,
                     rocsparse_int ell_width,
                     float* ell_val,
                     rocsparse_int* ell_col_ind);

  rocsparse_status
  rocsparse_dcsr2ell(rocsparse_handle handle,
                     rocsparse_int m,
                     const rocsparse_mat_descr csr_descr,
                     const double* csr_val,
                     const rocsparse_int* csr_row_ptr,
                     const rocsparse_int* csr_col_ind,
                     const rocsparse_mat_descr ell_descr,
                     rocsparse_int ell_width,
                     double* ell_val,
                     rocsparse_int* ell_col_ind);

*rocsparse_csr2ell()* converts a CSR matrix into an ELL matrix. It is assumed, that ``ell_val`` and ``ell_col_ind`` are allocated. Allocation size is computed by the number of rows times the number of ELL non-zero elements per row, such that :math:`\text{nnz}_{\text{ELL}} = m \cdot \text{ell_width}`. The number of ELL non-zero elements per row is obtained by :ref:`rocsparse_csr2ell_width`.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse CSR matrix.
csr_descr   descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_csr2ell()* supports only ``rocsparse_matrix_type_general``.
csr_val     array containing the values of the sparse CSR matrix.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array containing the column indices of the sparse CSR matrix.
ell_descr   descriptor of the sparse ELL matrix. |br| Currently, *rocsparse_csr2ell()* supports only ``rocsparse_matrix_type_general``.
ell_width   number of non-zero elements per row in ELL storage format.
=========== =============================================================================

=========== ================================
Output
=========== ================================
ell_val     array of ``m`` times ``ell_width`` elements of the sparse ELL matrix.
ell_col_ind array of ``m`` times ``ell_width`` elements containing the column indices |br| of the sparse ELL matrix.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m`` or ``ell_width`` is invalid.
rocsparse_status_invalid_pointer  ``csr_descr``, ``csr_val``, ``csr_row_ptr``, ``csr_col_ind``, |br| ``ell_descr``, ``ell_val`` or ``ell_col_ind`` pointer is invalid.
rocsparse_status_not_implemented  ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_ell2csr_nnz:

rocsparse_ell2csr_nnz
**********************
.. code-block:: c

  rocsparse_status
  rocsparse_ell2csr_nnz(rocsparse_handle handle,
                        rocsparse_int m,
                        rocsparse_int n,
                        const rocsparse_mat_descr ell_descr,
                        rocsparse_int ell_width,
                        const rocsparse_int* ell_col_ind,
                        const rocsparse_mat_descr csr_descr,
                        rocsparse_int* csr_row_ptr,
                        rocsparse_int* csr_nnz);

*rocsparse_ell2csr_nnz()* computes the total CSR non-zero elements and the CSR row offsets, that point to the start of every row of the sparse CSR matrix, for a given ELL matrix. It is assumed that ``csr_row_ptr`` has been allocated with size ``m + 1``.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse ELL matrix.
n           number of columns of the sparse ELL matrix.
ell_descr   descriptor of the sparse ELL matrix. |br| Currently, *rocsparse_ell2csr_nnz()* supports only ``rocsparse_matrix_type_general``.
ell_width   number of non-zero elements per row in ELL storage format.
ell_col_ind array of ``m`` times ``ell_width`` elements containing the column indices |br| of the sparse ELL matrix.
csr_descr   descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_ell2csr_nnz()* supports only ``rocsparse_matrix_type_general``.
=========== =============================================================================

=========== ================================
Output
=========== ================================
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_nnz     pointer to the total number of non-zero elements in CSR storage format.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``ell_width`` is invalid.
rocsparse_status_invalid_pointer  ``ell_descr``, ``ell_col_ind``, |br| ``csr_descr``, ``csr_row_ptr`` or ``csr_nnz`` pointer is invalid.
rocsparse_status_not_implemented  ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_ell2csr:

rocsparse_ell2csr
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_sell2csr(rocsparse_handle handle,
                     rocsparse_int m,
                     rocsparse_int n,
                     const rocsparse_mat_descr ell_descr,
                     rocsparse_int ell_width,
                     const float* ell_val,
                     const rocsparse_int* ell_col_ind,
                     const rocsparse_mat_descr csr_descr,
                     float* csr_val,
                     const rocsparse_int* csr_row_ptr,
                     rocsparse_int* csr_col_ind);

  rocsparse_status
  rocsparse_dell2csr(rocsparse_handle handle,
                     rocsparse_int m,
                     rocsparse_int n,
                     const rocsparse_mat_descr ell_descr,
                     rocsparse_int ell_width,
                     const double* ell_val,
                     const rocsparse_int* ell_col_ind,
                     const rocsparse_mat_descr csr_descr,
                     double* csr_val,
                     const rocsparse_int* csr_row_ptr,
                     rocsparse_int* csr_col_ind);

*rocsparse_ell2csr()* converts an ELL matrix into a CSR matrix. It is assumed that ``csr_row_ptr`` has already been filled and that ``csr_val`` and ``csr_col_ind`` are allocated by the user. ``csr_row_ptr`` and allocation size of ``csr_col_ind`` and ``csr_val`` is defined by the number of CSR non-zero elements. Both can be obtained by :ref:`rocsparse_ell2csr_nnz`.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse ELL matrix.
n           number of columns of the sparse ELL matrix.
ell_descr   descriptor of the sparse ELL matrix. |br| Currently, *rocsparse_ell2csr()* supports only ``rocsparse_matrix_type_general``.
ell_width   number of non-zero elements per row in ELL storage format.
ell_val     array of ``m`` times ``ell_width`` elements of the sparse ELL matrix.
ell_col_ind array of ``m`` times ``ell_width`` elements containing the column indices |br| of the sparse ELL matrix.
csr_descr   descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_ell2csr()* supports only ``rocsparse_matrix_type_general``.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
=========== =============================================================================

=========== ================================
Output
=========== ================================
csr_val     array containing the values of the sparse CSR matrix.
csr_col_ind array containing the column indices of the sparse CSR matrix.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``ell_width`` is invalid.
rocsparse_status_invalid_pointer  ``csr_descr``, ``csr_val``, ``csr_row_ptr``, ``csr_col_ind``, |br| ``ell_descr``, ``ell_val`` or ``ell_col_ind`` pointer is invalid.
rocsparse_status_not_implemented  ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

rocsparse_csr2hyb
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_scsr2hyb(rocsparse_handle handle,
                     rocsparse_int m,
                     rocsparse_int n,
                     const rocsparse_mat_descr descr,
                     const float* csr_val,
                     const rocsparse_int* csr_row_ptr,
                     const rocsparse_int* csr_col_ind,
                     rocsparse_hyb_mat hyb,
                     rocsparse_int user_ell_width,
                     rocsparse_hyb_partition partition_type);

  rocsparse_status
  rocsparse_dcsr2hyb(rocsparse_handle handle,
                     rocsparse_int m,
                     rocsparse_int n,
                     const rocsparse_mat_descr descr,
                     const double* csr_val,
                     const rocsparse_int* csr_row_ptr,
                     const rocsparse_int* csr_col_ind,
                     rocsparse_hyb_mat hyb,
                     rocsparse_int user_ell_width,
                     rocsparse_hyb_partition partition_type);

*rocsparse_csr2hyb()* converts a CSR matrix into a HYB matrix. It is assumed that ``hyb`` has been initialized with :ref:`rocsparse_create_hyb_mat`.

This function requires a significant amount of storage for the HYB matrix, depending on the matrix structure.

============== =============================================================================
Input
============== =============================================================================
handle         handle to the rocSPARSE library context queue.
m              number of rows of the sparse ELL matrix.
n              number of columns of the sparse ELL matrix.
descr          descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_csr2hyb()* supports only ``rocsparse_matrix_type_general``.
csr_val        array containing the values of the sparse CSR matrix.
csr_row_ptr    array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind    array containing the column indices of the sparse CSR matrix.
user_ell_width width of the ELL part of the HYB matrix (only required if |br| ``partition_type == rocsparse_hyb_partition_user``).
partition_type ``rocsparse_hyb_partition_auto`` (recommended), |br| ``rocsparse_hyb_partition_user`` or ``rocsparse_hyb_partition_max``.
============== =============================================================================

====== ================================
Output
====== ================================
hyb    sparse matrix in HYB format.
====== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``user_ell_width`` is invalid.
rocsparse_status_invalid_value    ``partition_type`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``hyb``, ``csr_val``, ``csr_row_ptr`` |br| or ``csr_col_ind`` pointer is invalid.
rocsparse_status_memory_error     the buffer for the HYB matrix could not be allocated.
rocsparse_status_internal_error   an internal error occurred.
rocsparse_status_not_implemented  ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_create_identity_permutation:

rocsparse_create_identity_permutation
**************************************
.. code-block:: c

  rocsparse_status
  rocsparse_create_identity_permutation(rocsparse_handle handle,
                                        rocsparse_int n,
                                        rocsparse_int* p);

*rocsparse_create_identity_permutation()* stores the identity map in ``p``, such that :math:`p = 0:1:(n-1)`.

.. code-block:: c

  for(i = 0; i < n; ++i)
    p[i] = i;

====== ==============================================
Input
====== ==============================================
handle handle to the rocSPARSE library context queue.
n      size of the map ``p``.
====== ==============================================

====== ===========================================
Output
====== ===========================================
p      array of ``n`` integers containing the map.
====== ===========================================

================================  ========================================
Returned rocsparse_status
================================  ========================================
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``n`` is invalid.
rocsparse_status_invalid_pointer  ``p`` pointer is invalid.
================================  ========================================

.. _rocsparse_csrsort_buffer_size:

rocsparse_csrsort_buffer_size
******************************
.. code-block:: c

  rocsparse_status
  rocsparse_csrsort_buffer_size(rocsparse_handle handle,
                                rocsparse_int m,
                                rocsparse_int n,
                                rocsparse_int nnz,
                                const rocsparse_int* csr_row_ptr,
                                const rocsparse_int* csr_col_ind,
                                size_t* buffer_size);

*rocsparse_csrsort_buffer_size()* returns the size of the temporary storage buffer required by :ref:`rocsparse_csrsort`. The temporary storage buffer must be allocated by the user.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse CSR matrix.
n           number of columns of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
=========== =============================================================================

=========== ================================
Output
=========== ================================
buffer_size number of bytes of the temporary storage buffer required by :ref:`rocsparse_csrsort`.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``csr_row_ptr``, ``csr_col_ind`` or ``buffer_size`` pointer is invalid.
rocsparse_status_internal_error   an internal error occurred.
================================  ====

.. _rocsparse_csrsort:

rocsparse_csrsort
*********************
.. code-block:: c

  rocsparse_status
  rocsparse_csrsort(rocsparse_handle handle,
                    rocsparse_int m,
                    rocsparse_int n,
                    rocsparse_int nnz,
                    const rocsparse_mat_descr descr,
                    const rocsparse_int* csr_row_ptr,
                    rocsparse_int* csr_col_ind,
                    rocsparse_int* perm,
                    void* temp_buffer);

*rocsparse_csrsort()* sorts a matrix in CSR format. The sorted permutation vector ``perm`` can be used to obtain sorted ``csr_val`` array. In this case, ``perm`` must be initialized as the identity permutation, see :ref:`rocsparse_create_identity_permutation`.

*rocsparse_csrsort()* requires extra temporary storage buffer that has to be allocated by the user. Storage buffer size can be determined by :ref:`rocsparse_csrsort_buffer_size`.

Example application:

.. code-block:: c

  //     1 2 3
  // A = 4 5 6
  //     7 8 9
  rocsparse_int m   = 3;
  rocsparse_int n   = 3;
  rocsparse_int nnz = 9;

  csr_row_ptr[m + 1] = {0, 3, 6, 9}                 // device memory
  csr_col_ind[nnz]   = {2, 0, 1, 0, 1, 2, 0, 2, 1}; // device memory
  csr_val[nnz]       = {3, 1, 2, 4, 5, 6, 7, 9, 8}; // device memory

  // Allocate temporary buffer
  size_t buffer_size = 0;
  void* temp_buffer  = NULL;
  rocsparse_csrsort_buffer_size(handle, m, n, nnz, csr_row_ptr, csr_col_ind, &buffer_size);
  hipMalloc(&temp_buffer, sizeof(char) * buffer_size);

  // Create permutation vector perm as the identity map
  rocsparse_int* perm = NULL;
  hipMalloc((void**)&perm, sizeof(rocsparse_int) * nnz);
  rocsparse_create_identity_permutation(handle, nnz, perm);

  // Sort the CSR matrix
  rocsparse_csrsort(handle, m, n, nnz, descr, csr_row_ptr, csr_col_ind, perm, temp_buffer);

  // Gather sorted csr_val array
  float* csr_val_sorted = NULL;
  hipMalloc((void**)&csr_val_sorted, sizeof(float) * nnz);
  rocsparse_sgthr(handle, nnz, csr_val, csr_val_sorted, perm, rocsparse_index_base_zero);

  // Clean up
  hipFree(temp_buffer);
  hipFree(perm);
  hipFree(csr_val);

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse CSR matrix.
n           number of columns of the sparse CSR matrix.
nnz         number of non-zero entries of the sparse CSR matrix.
descr       descriptor of the sparse CSR matrix. |br| Currently, *rocsparse_csrsort()* supports only ``rocsparse_matrix_type_general``.
csr_row_ptr array of ``m+1`` elements that point to the start of every row of the sparse CSR matrix.
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
perm        array of ``nnz`` integers containing the unsorted map indices, can be ``NULL``.
temp_buffer temporary storage buffer allocated by the user, |br| size is returned by :ref:`rocsparse_csrsort_buffer_size`.
=========== =============================================================================

=========== ================================
Output
=========== ================================
csr_col_ind array of ``nnz`` elements containing the column indices of the sparse CSR matrix.
perm        array of ``nnz`` integers containing the unsorted map indices, if not ``NULL``.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``descr``, ``csr_row_ptr``, ``csr_col_ind`` or |br| ``temp_buffer`` pointer is invalid.
rocsparse_status_internal_error   an internal error occurred.
rocsparse_status_not_implemented  ``rocsparse_matrix_type != rocsparse_matrix_type_general``.
================================  ====

.. _rocsparse_coosort_buffer_size:

rocsparse_coosort_buffer_size
*****************************
.. code-block:: c

  rocsparse_status
  rocsparse_coosort_buffer_size(rocsparse_handle handle,
                                rocsparse_int m,
                                rocsparse_int n,
                                rocsparse_int nnz,
                                const rocsparse_int* coo_row_ind,
                                const rocsparse_int* coo_col_ind,
                                size_t* buffer_size);

*rocsparse_coosort_buffer_size()* returns the size of the temporary storage buffer required by :ref:`rocsparse_coosort`. The temporary storage buffer must be allocated by the user.

=========== =============================================================================
Input
=========== =============================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse COO matrix.
n           number of columns of the sparse COO matrix.
nnz         number of non-zero entries of the sparse COO matrix.
coo_row_ind array of ``nnz`` elements containing the row indices of the sparse COO matrix.
coo_col_ind array of ``nnz`` elements containing the column indices of the sparse COO matrix.
=========== =============================================================================

=========== ================================
Output
=========== ================================
buffer_size number of bytes of the temporary storage buffer required by :ref:`rocsparse_coosort`.
=========== ================================

================================  ====
Returned rocsparse_status
================================  ====
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``coo_row_ind``, ``coo_col_ind`` or ``buffer_size`` pointer is invalid.
rocsparse_status_internal_error   an internal error occurred.
================================  ====

.. _rocsparse_coosort:

rocsparse_coosort
*****************
.. code-block:: c

  rocsparse_status
  rocsparse_coosort_by_row(rocsparse_handle handle,
                           rocsparse_int m,
                           rocsparse_int n,
                           rocsparse_int nnz,
                           rocsparse_int* coo_row_ind,
                           rocsparse_int* coo_col_ind,
                           rocsparse_int* perm,
                           void* temp_buffer);

  rocsparse_status
  rocsparse_coosort_by_column(rocsparse_handle handle,
                              rocsparse_int m,
                              rocsparse_int n,
                              rocsparse_int nnz,
                              rocsparse_int* coo_row_ind,
                              rocsparse_int* coo_col_ind,
                              rocsparse_int* perm,
                              void* temp_buffer);

*rocsparse_coosort_by_row/column()* sorts a matrix in COO format by row/column. The sorted permutation vector ``perm`` can be used to obtain sorted ``coo_val`` array. In this case, ``perm`` must be initialized as the identity permutation, see :ref:`rocsparse_create_identity_permutation`.

*rocsparse_coosort_by_row/column()* requires extra temporary storage buffer that has to be allocated by the user. Storage buffer size can be determined by :ref:`rocsparse_coosort_buffer_size`.

=========== =================================================================================================
Input
=========== =================================================================================================
handle      handle to the rocSPARSE library context queue.
m           number of rows of the sparse COO matrix.
n           number of columns of the sparse COO matrix.
nnz         number of non-zero entries of the sparse COO matrix.
coo_row_ind array of ``nnz`` elements containing the row indices of the sparse COO matrix.
coo_col_ind array of ``nnz`` elements containing the column indices of the sparse COO matrix.
perm        array of ``nnz`` integers containing the unsorted map indices, can be ``NULL``.
temp_buffer temporary storage buffer allocated by the user, |br| size is returned by :ref:`rocsparse_coosort_buffer_size`.
=========== =================================================================================================

=========== =================================================================================
Output
=========== =================================================================================
coo_row_ind array of ``nnz`` elements containing the row indices of the sparse COO matrix.
coo_col_ind array of ``nnz`` elements containing the column indices of the sparse COO matrix.
perm        array of ``nnz`` integers containing the unsorted map indices, if not ``NULL``.
=========== =================================================================================

================================  =======================================================================
Returned rocsparse_status
================================  =======================================================================
rocsparse_status_success          the operation completed successfully.
rocsparse_status_invalid_handle   the library context was not initialized.
rocsparse_status_invalid_size     ``m``, ``n`` or ``nnz`` is invalid.
rocsparse_status_invalid_pointer  ``coo_row_ind``, ``coo_col_ind`` or ``temp_buffer`` pointer is invalid.
rocsparse_status_internal_error   an internal error occurred.
================================  =======================================================================
