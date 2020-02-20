.. _user_manual:

***********
User Manual
***********

.. toctree::
   :maxdepth: 3
   :caption: Contents:

Introduction
============

rocSPARSE is a library that contains basic linear algebra subroutines for sparse matrices and vectors written in HiP for GPU devices. It is designed to be used from C and C++ code. The functionality of rocSPARSE is organized in the following categories:

* :ref:`rocsparse_auxiliary_functions_` describe available helper functions that are required for subsequent library calls.
* :ref:`rocsparse_level1_functions_` describe operations between a vector in sparse format and a vector in dense format.
* :ref:`rocsparse_level2_functions_` describe operations between a matrix in sparse format and a vector in dense format.
* :ref:`rocsparse_level3_functions_` describe operations between a matrix in sparse format and multiple vectors in dense format.
* :ref:`rocsparse_extra_functions_` describe operations that manipulate sparse matrices.
* :ref:`rocsparse_precond_functions_` describe manipulations on a matrix in sparse format to obtain a preconditioner.
* :ref:`rocsparse_conversion_functions_` describe operations on a matrix in sparse format to obtain a different matrix format.

The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/rocSPARSE

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
- `libboost-program-options <https://www.boost.org/>`_ (optional, for clients)

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
  $ CXX=/opt/rocm/bin/hcc cmake ../..

  # Compile rocSPARSE library
  $ make -j$(nproc)

  # Install rocSPARSE to /opt/rocm
  $ make install

Boost and GoogleTest is required in order to build rocSPARSE clients.

rocSPARSE with dependencies and clients can be built using the following commands:

::

  # Install boost on e.g. Ubuntu
  $ apt install libboost-program-options-dev

  # Install googletest
  $ mkdir -p build/release/deps ; cd build/release/deps
  $ cmake ../../../deps
  $ make -j$(nproc) install

  # Change to build directory
  $ cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ CXX=/opt/rocm/bin/hcc cmake ../.. -DBUILD_CLIENTS_TESTS=ON \
                                      -DBUILD_CLIENTS_BENCHMARKS=ON \
                                      -DBUILD_CLIENTS_SAMPLES=ON

  # Compile rocSPARSE library
  $ make -j$(nproc)

  # Install rocSPARSE to /opt/rocm
  $ make install

Common build problems
`````````````````````
#. **Issue:** HIP (`/opt/rocm/hip`) was built using `hcc` 1.0.xxx-xxx-xxx-xxx, but you are using `/opt/rocm/bin/hcc` with version 1.0.yyy-yyy-yyy-yyy from `hipcc` (version mismatch). Please rebuild HIP including cmake or update HCC_HOME variable.

   **Solution:** Download HIP from GitHub and use `hcc` to `build from source <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md>`_ and then use the built HIP instead of `/opt/rocm/hip`.

#. **Issue:** HCC RUNTIME ERROR: Failed to find compatible kernel

   **Solution:** Add the following to the cmake command when configuring: `-DCMAKE_CXX_FLAGS="--amdgpu-target=gfx803,gfx900,gfx906,gfx908"`

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

Device and Stream Management
============================
:cpp:func:`hipSetDevice` and :cpp:func:`hipGetDevice` are HIP device management APIs.
They are NOT part of the rocSPARSE API.

Asynchronous Execution
----------------------
All rocSPARSE library functions, unless otherwise stated, are non blocking and executed asynchronously with respect to the host. They may return before the actual computation has finished. To force synchronization, :cpp:func:`hipDeviceSynchronize` or :cpp:func:`hipStreamSynchronize` can be used. This will ensure that all previously executed rocSPARSE functions on the device / this particular stream have completed.

HIP Device Management
---------------------
Before a HIP kernel invocation, users need to call :cpp:func:`hipSetDevice` to set a device, e.g. device 1. If users do not explicitly call it, the system by default sets it as device 0. Unless users explicitly call :cpp:func:`hipSetDevice` to set to another device, their HIP kernels are always launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing to do with rocSPARSE. rocSPARSE honors the approach above and assumes users have already set the device before a rocSPARSE routine call.

Once users set the device, they create a handle with :ref:`rocsparse_create_handle_`.

Subsequent rocSPARSE routines take this handle as an input parameter. rocSPARSE ONLY queries (by :cpp:func:`hipGetDevice`) the user's device; rocSPARSE does NOT set the device for users. If rocSPARSE does not see a valid device, it returns an error message. It is the users' responsibility to provide a valid device to rocSPARSE and ensure the device safety.

Users CANNOT switch devices between :ref:`rocsparse_create_handle_` and :ref:`rocsparse_destroy_handle_`. If users want to change device, they must destroy the current handle and create another rocSPARSE handle.

HIP Stream Management
---------------------
HIP kernels are always launched in a queue (also known as stream).

If users do not explicitly specify a stream, the system provides a default stream, maintained by the system. Users cannot create or destroy the default stream. However, users can freely create new streams (with :cpp:func:`hipStreamCreate`) and bind it to the rocSPARSE handle using :ref:`rocsparse_set_stream_`. HIP kernels are invoked in rocSPARSE routines. The rocSPARSE handle is always associated with a stream, and rocSPARSE passes its stream to the kernels inside the routine. One rocSPARSE routine only takes one stream in a single invocation. If users create a stream, they are responsible for destroying it.

Multiple Streams and Multiple Devices
-------------------------------------
If the system under test has multiple HIP devices, users can run multiple rocSPARSE handles concurrently, but can NOT run a single rocSPARSE handle on different discrete devices. Each handle is associated with a particular singular device, and a new handle should be created for each additional device.

Storage Formats
===============

COO storage format
------------------
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
------------------
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
------------------
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
------------------
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
=====

rocsparse_handle
----------------

.. doxygentypedef:: rocsparse_handle

rocsparse_mat_descr
-------------------

.. doxygentypedef:: rocsparse_mat_descr


.. _rocsparse_mat_info_:

rocsparse_mat_info
------------------

.. doxygentypedef:: rocsparse_mat_info

rocsparse_hyb_mat
-----------------

.. doxygentypedef:: rocsparse_hyb_mat

For more details on the HYB format, see :ref:`HYB storage format`.

.. _rocsparse_action_:

rocsparse_action
----------------

.. doxygenenum:: rocsparse_action

.. _rocsparse_hyb_partition_:

rocsparse_hyb_partition
-----------------------

.. doxygenenum:: rocsparse_hyb_partition

.. _rocsparse_index_base_:

rocsparse_index_base
--------------------

.. doxygenenum:: rocsparse_index_base

rocsparse_matrix_type
---------------------

.. doxygenenum:: rocsparse_matrix_type

.. _rocsparse_fill_mode_:

rocsparse_fill_mode
-------------------

.. doxygenenum:: rocsparse_fill_mode

.. _rocsparse_diag_type_:

rocsparse_diag_type
-------------------

.. doxygenenum:: rocsparse_diag_type

.. _rocsparse_operation_:

rocsparse_operation
-------------------

.. doxygenenum:: rocsparse_operation

rocsparse_pointer_mode
----------------------

.. doxygenenum:: rocsparse_pointer_mode

.. _rocsparse_analysis_policy_:

rocsparse_analysis_policy
-------------------------

.. doxygenenum:: rocsparse_analysis_policy

rocsparse_solve_policy
----------------------

.. doxygenenum:: rocsparse_solve_policy

.. _rocsparse_layer_mode_:

rocsparse_layer_mode
--------------------

.. doxygenenum:: rocsparse_layer_mode

For more details on logging, see :ref:`rocsparse_logging`.

rocsparse_status
----------------

.. doxygenenum:: rocsparse_status

.. _rocsparse_logging:

Logging
=======
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

.. _api:

Exported Sparse Functions
=========================

Auxiliary Functions
-------------------

+----------------------------------------+
|Function name                           |
+----------------------------------------+
|:cpp:func:`rocsparse_create_handle`     |
+----------------------------------------+
|:cpp:func:`rocsparse_destroy_handle`    |
+----------------------------------------+
|:cpp:func:`rocsparse_set_stream`        |
+----------------------------------------+
|:cpp:func:`rocsparse_get_stream`        |
+----------------------------------------+
|:cpp:func:`rocsparse_set_pointer_mode`  |
+----------------------------------------+
|:cpp:func:`rocsparse_get_pointer_mode`  |
+----------------------------------------+
|:cpp:func:`rocsparse_get_version`       |
+----------------------------------------+
|:cpp:func:`rocsparse_get_git_rev`       |
+----------------------------------------+
|:cpp:func:`rocsparse_create_mat_descr`  |
+----------------------------------------+
|:cpp:func:`rocsparse_destroy_mat_descr` |
+----------------------------------------+
|:cpp:func:`rocsparse_copy_mat_descr`    |
+----------------------------------------+
|:cpp:func:`rocsparse_set_mat_index_base`|
+----------------------------------------+
|:cpp:func:`rocsparse_get_mat_index_base`|
+----------------------------------------+
|:cpp:func:`rocsparse_set_mat_type`      |
+----------------------------------------+
|:cpp:func:`rocsparse_get_mat_type`      |
+----------------------------------------+
|:cpp:func:`rocsparse_set_mat_fill_mode` |
+----------------------------------------+
|:cpp:func:`rocsparse_get_mat_fill_mode` |
+----------------------------------------+
|:cpp:func:`rocsparse_set_mat_diag_type` |
+----------------------------------------+
|:cpp:func:`rocsparse_get_mat_diag_type` |
+----------------------------------------+
|:cpp:func:`rocsparse_create_hyb_mat`    |
+----------------------------------------+
|:cpp:func:`rocsparse_destroy_hyb_mat`   |
+----------------------------------------+
|:cpp:func:`rocsparse_create_mat_info`   |
+----------------------------------------+
|:cpp:func:`rocsparse_destroy_mat_info`  |
+----------------------------------------+

Sparse Level 1 Functions
------------------------

================================================= ====== ====== ============== ==============
Function name                                     single double single complex double complex
================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xaxpyi() <rocsparse_saxpyi>` x      x      x              x
:cpp:func:`rocsparse_Xdoti() <rocsparse_sdoti>`   x      x      x              x
:cpp:func:`rocsparse_Xdotci() <rocsparse_cdotci>`               x              x
:cpp:func:`rocsparse_Xgthr() <rocsparse_sgthr>`   x      x      x              x
:cpp:func:`rocsparse_Xgthrz() <rocsparse_sgthrz>` x      x      x              x
:cpp:func:`rocsparse_Xroti() <rocsparse_sroti>`   x      x
:cpp:func:`rocsparse_Xsctr() <rocsparse_ssctr>`   x      x      x              x
================================================= ====== ====== ============== ==============

Sparse Level 2 Functions
------------------------

========================================================================= ====== ====== ============== ==============
Function name                                                             single double single complex double complex
========================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xcoomv() <rocsparse_scoomv>`                         x      x      x              x
:cpp:func:`rocsparse_Xcsrmv_analysis() <rocsparse_scsrmv_analysis>`       x      x      x              x
:cpp:func:`rocsparse_csrmv_clear`
:cpp:func:`rocsparse_Xcsrmv() <rocsparse_scsrmv>`                         x      x      x              x
:cpp:func:`rocsparse_Xcsrsv_buffer_size() <rocsparse_scsrsv_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcsrsv_analysis() <rocsparse_scsrsv_analysis>`       x      x      x              x
:cpp:func:`rocsparse_csrsv_zero_pivot`
:cpp:func:`rocsparse_csrsv_clear`
:cpp:func:`rocsparse_Xcsrsv_solve() <rocsparse_scsrsv_solve>`             x      x      x              x
:cpp:func:`rocsparse_Xellmv() <rocsparse_sellmv>`                         x      x      x              x
:cpp:func:`rocsparse_Xhybmv() <rocsparse_shybmv>`                         x      x      x              x
========================================================================= ====== ====== ============== ==============

Sparse Level 3 Functions
------------------------

========================================================================= ====== ====== ============== ==============
Function name                                                             single double single complex double complex
========================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xcsrmm() <rocsparse_scsrmm>`                         x      x      x              x
:cpp:func:`rocsparse_Xcsrsm_buffer_size() <rocsparse_scsrsm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcsrsm_analysis() <rocsparse_scsrsm_analysis>`       x      x      x              x
:cpp:func:`rocsparse_csrsm_zero_pivot`
:cpp:func:`rocsparse_csrsm_clear`
:cpp:func:`rocsparse_Xcsrsm_solve() <rocsparse_scsrsm_solve>`             x      x      x              x
========================================================================= ====== ====== ============== ==============

Sparse Extra Functions
----------------------

============================================================================= ====== ====== ============== ==============
Function name                                                                 single double single complex double complex
============================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xcsrgemm_buffer_size() <rocsparse_scsrgemm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_csrgemm_nnz`
:cpp:func:`rocsparse_Xcsrgemm() <rocsparse_scsrgemm>`                         x      x      x              x
============================================================================= ====== ====== ============== ==============

Preconditioner Functions
------------------------

============================================================================= ====== ====== ============== ==============
Function name                                                                 single double single complex double complex
============================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xcsric0_buffer_size() <rocsparse_scsric0_buffer_size>`   x      x      x              x
:cpp:func:`rocsparse_Xcsric0_analysis() <rocsparse_scsric0_analysis>`         x      x      x              x
:cpp:func:`rocsparse_csric0_zero_pivot`
:cpp:func:`rocsparse_csric0_clear`
:cpp:func:`rocsparse_Xcsric0() <rocsparse_scsric0>`                           x      x      x              x
:cpp:func:`rocsparse_Xcsrilu0_buffer_size() <rocsparse_scsrilu0_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcsrilu0_analysis() <rocsparse_scsrilu0_analysis>`       x      x      x              x
:cpp:func:`rocsparse_csrilu0_zero_pivot`
:cpp:func:`rocsparse_csrilu0_clear`
:cpp:func:`rocsparse_Xcsrilu0() <rocsparse_scsrilu0>`                         x      x      x              x
============================================================================= ====== ====== ============== ==============

Conversion Functions
--------------------

===================================================== ====== ====== ============== ==============
Function name                                         single double single complex double complex
===================================================== ====== ====== ============== ==============
:cpp:func:`rocsparse_csr2coo`
:cpp:func:`rocsparse_csr2csc_buffer_size`
:cpp:func:`rocsparse_Xcsr2csc() <rocsparse_scsr2csc>` x      x      x              x
:cpp:func:`rocsparse_csr2ell_width`
:cpp:func:`rocsparse_Xcsr2ell() <rocsparse_scsr2ell>` x      x      x              x
:cpp:func:`rocsparse_Xcsr2hyb() <rocsparse_scsr2hyb>` x      x      x              x
:cpp:func:`rocsparse_coo2csr`
:cpp:func:`rocsparse_ell2csr_nnz`
:cpp:func:`rocsparse_Xell2csr() <rocsparse_sell2csr>` x      x      x              x
:cpp:func:`rocsparse_hyb2csr_buffer_size`
:cpp:func:`rocsparse_Xhyb2csr() <rocsparse_shyb2csr>` x      x      x              x
:cpp:func:`rocsparse_create_identity_permutation`
:cpp:func:`rocsparse_cscsort_buffer_size`
:cpp:func:`rocsparse_cscsort`
:cpp:func:`rocsparse_csrsort_buffer_size`
:cpp:func:`rocsparse_csrsort`
:cpp:func:`rocsparse_coosort_buffer_size`
:cpp:func:`rocsparse_coosort_by_row`
:cpp:func:`rocsparse_coosort_by_column`
:cpp:func:`rocsparse_Xnnz`                            x      x      x              x
===================================================== ====== ====== ============== ==============

Storage schemes and indexing base
---------------------------------
rocSPARSE supports 0 and 1 based indexing.
The index base is selected by the :cpp:enum:`rocsparse_index_base` type which is either passed as standalone parameter or as part of the :cpp:type:`rocsparse_mat_descr` type.

Furthermore, dense vectors are represented with a 1D array, stored linearly in memory.
Sparse vectors are represented by a 1D data array stored linearly in memory that hold all non-zero elements and a 1D indexing array stored linearly in memory that hold the positions of the corresponding non-zero elements.

Pointer mode
------------
The auxiliary functions :cpp:func:`rocsparse_set_pointer_mode` and :cpp:func:`rocsparse_get_pointer_mode` are used to set and get the value of the state variable :cpp:enum:`rocsparse_pointer_mode`.
If :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_host`, then scalar parameters must be allocated on the host.
If :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_device`, then scalar parameters must be allocated on the device.

There are two types of scalar parameter:

  1. Scaling parameters, such as `alpha` and `beta` used in e.g. :cpp:func:`rocsparse_scsrmv`, :cpp:func:`rocsparse_scoomv`, ...
  2. Scalar results from functions such as :cpp:func:`rocsparse_sdoti`, :cpp:func:`rocsparse_cdotci`, ...

For scalar parameters such as alpha and beta, memory can be allocated on the host heap or stack, when :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_host`.
The kernel launch is asynchronous, and if the scalar parameter is on the heap, it can be freed after the return from the kernel launch.
When :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_device`, the scalar parameter must not be changed till the kernel completes.

For scalar results, when :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_host`, the function blocks the CPU till the GPU has copied the result back to the host.
Using :cpp:enum:`rocsparse_pointer_mode` equal to :cpp:enumerator:`rocsparse_pointer_mode_device`, the function will return after the asynchronous launch.
Similarly to vector and matrix results, the scalar result is only available when the kernel has completed execution.

Asynchronous API
----------------
Except a functions having memory allocation inside preventing asynchronicity, all rocSPARSE functions are configured to operate in non-blocking fashion with respect to CPU, meaning these library functions return immediately.

hipSPARSE
---------
hipSPARSE is a SPARSE marshalling library, with multiple supported backends.
It sits between the application and a `worker` SPARSE library, marshalling inputs into the backend library and marshalling results back to the application.
hipSPARSE exports an interface that does not require the client to change, regardless of the chosen backend.
Currently, hipSPARSE supports rocSPARSE and cuSPARSE as backends.
hipSPARSE focuses on convenience and portability.
If performance outweighs these factors, then using rocSPARSE itself is highly recommended.
hipSPARSE can be found on `GitHub <https://github.com/ROCmSoftwarePlatform/hipSPARSE/>`_.

.. _rocsparse_auxiliary_functions_:

Sparse Auxiliary Functions
==========================

This module holds all sparse auxiliary functions.

The functions that are contained in the auxiliary module describe all available helper functions that are required for subsequent library calls.

.. _rocsparse_create_handle_:

rocsparse_create_handle()
-------------------------

.. doxygenfunction:: rocsparse_create_handle

.. _rocsparse_destroy_handle_:

rocsparse_destroy_handle()
--------------------------

.. doxygenfunction:: rocsparse_destroy_handle

.. _rocsparse_set_stream_:

rocsparse_set_stream()
----------------------

.. doxygenfunction:: rocsparse_set_stream

rocsparse_get_stream()
----------------------

.. doxygenfunction:: rocsparse_get_stream

rocsparse_set_pointer_mode()
----------------------------

.. doxygenfunction:: rocsparse_set_pointer_mode

rocsparse_get_pointer_mode()
----------------------------

.. doxygenfunction:: rocsparse_get_pointer_mode

rocsparse_get_version()
-----------------------

.. doxygenfunction:: rocsparse_get_version

rocsparse_get_git_rev()
-----------------------

.. doxygenfunction:: rocsparse_get_git_rev

rocsparse_create_mat_descr()
----------------------------

.. doxygenfunction:: rocsparse_create_mat_descr

rocsparse_destroy_mat_descr()
-----------------------------

.. doxygenfunction:: rocsparse_destroy_mat_descr

rocsparse_copy_mat_descr()
--------------------------

.. doxygenfunction:: rocsparse_copy_mat_descr

rocsparse_set_mat_index_base()
------------------------------

.. doxygenfunction:: rocsparse_set_mat_index_base

rocsparse_get_mat_index_base()
------------------------------

.. doxygenfunction:: rocsparse_get_mat_index_base

rocsparse_set_mat_type()
------------------------

.. doxygenfunction:: rocsparse_set_mat_type

rocsparse_get_mat_type()
------------------------

.. doxygenfunction:: rocsparse_get_mat_type

rocsparse_set_mat_fill_mode()
-----------------------------

.. doxygenfunction:: rocsparse_set_mat_fill_mode

rocsparse_get_mat_fill_mode()
-----------------------------

.. doxygenfunction:: rocsparse_get_mat_fill_mode

rocsparse_set_mat_diag_type()
-----------------------------

.. doxygenfunction:: rocsparse_set_mat_diag_type

rocsparse_get_mat_diag_type()
-----------------------------

.. doxygenfunction:: rocsparse_get_mat_diag_type

.. _rocsparse_create_hyb_mat_:

rocsparse_create_hyb_mat()
--------------------------

.. doxygenfunction:: rocsparse_create_hyb_mat

rocsparse_destroy_hyb_mat()
---------------------------

.. doxygenfunction:: rocsparse_destroy_hyb_mat

rocsparse_create_mat_info()
---------------------------

.. doxygenfunction:: rocsparse_create_mat_info

.. _rocsparse_destroy_mat_info_:

rocsparse_destroy_mat_info()
----------------------------

.. doxygenfunction:: rocsparse_destroy_mat_info

.. _rocsparse_level1_functions_:

Sparse Level 1 Functions
========================

The sparse level 1 routines describe operations between a vector in sparse format and a vector in dense format. This section describes all rocSPARSE level 1 sparse linear algebra functions.

rocsparse_axpyi()
-----------------

.. doxygenfunction:: rocsparse_saxpyi
  :outline:
.. doxygenfunction:: rocsparse_daxpyi
  :outline:
.. doxygenfunction:: rocsparse_caxpyi
  :outline:
.. doxygenfunction:: rocsparse_zaxpyi

rocsparse_doti()
----------------

.. doxygenfunction:: rocsparse_sdoti
  :outline:
.. doxygenfunction:: rocsparse_ddoti
  :outline:
.. doxygenfunction:: rocsparse_cdoti
  :outline:
.. doxygenfunction:: rocsparse_zdoti

rocsparse_dotci()
-----------------

.. doxygenfunction:: rocsparse_cdotci
  :outline:
.. doxygenfunction:: rocsparse_zdotci

rocsparse_gthr()
----------------

.. doxygenfunction:: rocsparse_sgthr
  :outline:
.. doxygenfunction:: rocsparse_dgthr
  :outline:
.. doxygenfunction:: rocsparse_cgthr
  :outline:
.. doxygenfunction:: rocsparse_zgthr

rocsparse_gthrz()
-----------------

.. doxygenfunction:: rocsparse_sgthrz
  :outline:
.. doxygenfunction:: rocsparse_dgthrz
  :outline:
.. doxygenfunction:: rocsparse_cgthrz
  :outline:
.. doxygenfunction:: rocsparse_zgthrz

rocsparse_roti()
----------------

.. doxygenfunction:: rocsparse_sroti
  :outline:
.. doxygenfunction:: rocsparse_droti

rocsparse_sctr()
----------------

.. doxygenfunction:: rocsparse_ssctr
  :outline:
.. doxygenfunction:: rocsparse_dsctr
  :outline:
.. doxygenfunction:: rocsparse_csctr
  :outline:
.. doxygenfunction:: rocsparse_zsctr

.. _rocsparse_level2_functions_:

Sparse Level 2 Functions
========================

This module holds all sparse level 2 routines.

The sparse level 2 routines describe operations between a matrix in sparse format and a vector in dense format.

rocsparse_coomv()
-----------------

.. doxygenfunction:: rocsparse_scoomv
  :outline:
.. doxygenfunction:: rocsparse_dcoomv
  :outline:
.. doxygenfunction:: rocsparse_ccoomv
  :outline:
.. doxygenfunction:: rocsparse_zcoomv

rocsparse_csrmv_analysis()
--------------------------

.. doxygenfunction:: rocsparse_scsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsrmv_analysis

rocsparse_csrmv()
-----------------

.. doxygenfunction:: rocsparse_scsrmv
  :outline:
.. doxygenfunction:: rocsparse_dcsrmv
  :outline:
.. doxygenfunction:: rocsparse_ccsrmv
  :outline:
.. doxygenfunction:: rocsparse_zcsrmv

rocsparse_csrmv_analysis_clear()
--------------------------------

.. doxygenfunction:: rocsparse_csrmv_clear

rocsparse_ellmv()
-----------------

.. doxygenfunction:: rocsparse_sellmv
  :outline:
.. doxygenfunction:: rocsparse_dellmv
  :outline:
.. doxygenfunction:: rocsparse_cellmv
  :outline:
.. doxygenfunction:: rocsparse_zellmv

rocsparse_hybmv()
-----------------

.. doxygenfunction:: rocsparse_shybmv
  :outline:
.. doxygenfunction:: rocsparse_dhybmv
  :outline:
.. doxygenfunction:: rocsparse_chybmv
  :outline:
.. doxygenfunction:: rocsparse_zhybmv

rocsparse_csrsv_zero_pivot()
----------------------------

.. doxygenfunction:: rocsparse_csrsv_zero_pivot

rocsparse_csrsv_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_scsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsrsv_buffer_size

rocsparse_csrsv_analysis()
--------------------------

.. doxygenfunction:: rocsparse_scsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsrsv_analysis

rocsparse_csrsv_solve()
-----------------------

.. doxygenfunction:: rocsparse_scsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_dcsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_ccsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_zcsrsv_solve

rocsparse_csrsv_clear()
-----------------------

.. doxygenfunction:: rocsparse_csrsv_clear

.. _rocsparse_level3_functions_:

Sparse Level 3 Functions
========================

This module holds all sparse level 3 routines.

The sparse level 3 routines describe operations between a matrix in sparse format and multiple vectors in dense format that can also be seen as a dense matrix.

rocsparse_csrmm()
-----------------

.. doxygenfunction:: rocsparse_scsrmm
  :outline:
.. doxygenfunction:: rocsparse_dcsrmm
  :outline:
.. doxygenfunction:: rocsparse_ccsrmm
  :outline:
.. doxygenfunction:: rocsparse_zcsrmm

rocsparse_csrsm_zero_pivot()
----------------------------

.. doxygenfunction:: rocsparse_csrsm_zero_pivot

rocsparse_csrsm_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_scsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsrsm_buffer_size

rocsparse_csrsm_analysis()
--------------------------

.. doxygenfunction:: rocsparse_scsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsrsm_analysis

rocsparse_csrsm_solve()
-----------------------

.. doxygenfunction:: rocsparse_scsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_dcsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_ccsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_zcsrsm_solve

rocsparse_csrsm_clear()
-----------------------

.. doxygenfunction:: rocsparse_csrsm_clear

.. _rocsparse_extra_functions_:

Sparse Extra Functions
======================

This module holds all sparse extra routines.

The sparse extra routines describe operations that manipulate sparse matrices.

rocsparse_csrgemm_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_scsrgemm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrgemm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsrgemm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsrgemm_buffer_size

rocsparse_csrgemm_nnz()
-----------------------

.. doxygenfunction:: rocsparse_csrgemm_nnz

rocsparse_csrgemm()
-------------------

.. doxygenfunction:: rocsparse_scsrgemm
  :outline:
.. doxygenfunction:: rocsparse_dcsrgemm
  :outline:
.. doxygenfunction:: rocsparse_ccsrgemm
  :outline:
.. doxygenfunction:: rocsparse_zcsrgemm

.. _rocsparse_precond_functions_:

Preconditioner Functions
========================

This module holds all sparse preconditioners.

The sparse preconditioners describe manipulations on a matrix in sparse format to obtain a sparse preconditioner matrix.

rocsparse_csric0_zero_pivot()
-----------------------------

.. doxygenfunction:: rocsparse_csric0_zero_pivot

rocsparse_csric0_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_scsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsric0_buffer_size

rocsparse_csric0_analysis()
---------------------------

.. doxygenfunction:: rocsparse_scsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsric0_analysis

rocsparse_csric0()
------------------

.. doxygenfunction:: rocsparse_scsric0
  :outline:
.. doxygenfunction:: rocsparse_dcsric0
  :outline:
.. doxygenfunction:: rocsparse_ccsric0
  :outline:
.. doxygenfunction:: rocsparse_zcsric0

rocsparse_csric0_clear()
------------------------

.. doxygenfunction:: rocsparse_csric0_clear

rocsparse_csrilu0_zero_pivot()
------------------------------

.. doxygenfunction:: rocsparse_csrilu0_zero_pivot

rocsparse_csrilu0_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_scsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsrilu0_buffer_size

rocsparse_csrilu0_analysis()
----------------------------

.. doxygenfunction:: rocsparse_scsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsrilu0_analysis

rocsparse_csrilu0()
-------------------

.. doxygenfunction:: rocsparse_scsrilu0
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0
  :outline:
.. doxygenfunction:: rocsparse_ccsrilu0
  :outline:
.. doxygenfunction:: rocsparse_zcsrilu0

rocsparse_csrilu0_clear()
-------------------------

.. doxygenfunction:: rocsparse_csrilu0_clear

.. _rocsparse_conversion_functions_:

Sparse Conversion Functions
===========================

This module holds all sparse conversion routines.

The sparse conversion routines describe operations on a matrix in sparse format to obtain a matrix in a different sparse format.

rocsparse_csr2coo()
-------------------

.. doxygenfunction:: rocsparse_csr2coo

rocsparse_coo2csr()
-------------------

.. doxygenfunction:: rocsparse_coo2csr

rocsparse_csr2csc_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_csr2csc_buffer_size

rocsparse_csr2csc()
-------------------

.. doxygenfunction:: rocsparse_scsr2csc
  :outline:
.. doxygenfunction:: rocsparse_dcsr2csc
  :outline:
.. doxygenfunction:: rocsparse_ccsr2csc
  :outline:
.. doxygenfunction:: rocsparse_zcsr2csc

rocsparse_csr2ell_width()
-------------------------

.. doxygenfunction:: rocsparse_csr2ell_width

rocsparse_csr2ell()
-------------------

.. doxygenfunction:: rocsparse_scsr2ell
  :outline:
.. doxygenfunction:: rocsparse_dcsr2ell
  :outline:
.. doxygenfunction:: rocsparse_ccsr2ell
  :outline:
.. doxygenfunction:: rocsparse_zcsr2ell

rocsparse_ell2csr_nnz()
-----------------------

.. doxygenfunction:: rocsparse_ell2csr_nnz

rocsparse_ell2csr()
-------------------

.. doxygenfunction:: rocsparse_sell2csr
  :outline:
.. doxygenfunction:: rocsparse_dell2csr
  :outline:
.. doxygenfunction:: rocsparse_cell2csr
  :outline:
.. doxygenfunction:: rocsparse_zell2csr

rocsparse_csr2hyb()
-------------------

.. doxygenfunction:: rocsparse_scsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_dcsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_ccsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_zcsr2hyb

rocsparse_hyb2csr_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_hyb2csr_buffer_size

rocsparse_hyb2csr()
-------------------

.. doxygenfunction:: rocsparse_shyb2csr
  :outline:
.. doxygenfunction:: rocsparse_dhyb2csr
  :outline:
.. doxygenfunction:: rocsparse_chyb2csr
  :outline:
.. doxygenfunction:: rocsparse_zhyb2csr

rocsparse_create_identity_permutation()
---------------------------------------

.. doxygenfunction:: rocsparse_create_identity_permutation

rocsparse_csrsort_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_csrsort_buffer_size

rocsparse_csrsort()
-------------------

.. doxygenfunction:: rocsparse_csrsort

rocsparse_cscsort_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_cscsort_buffer_size

rocsparse_cscsort()
-------------------

.. doxygenfunction:: rocsparse_cscsort

rocsparse_coosort_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_coosort_buffer_size

rocsparse_coosort_by_row()
--------------------------

.. doxygenfunction:: rocsparse_coosort_by_row

rocsparse_coosort_by_column()
-----------------------------

.. doxygenfunction:: rocsparse_coosort_by_column

rocsparse_nnz()
-------------------

.. doxygenfunction:: rocsparse_snnz
  :outline:
.. doxygenfunction:: rocsparse_dnnz
  :outline:
.. doxygenfunction:: rocsparse_cnnz
  :outline:
.. doxygenfunction:: rocsparse_znnz
