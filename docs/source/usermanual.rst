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
* :ref:`rocsparse_reordering_functions_` describe operations on a matrix in sparse format to obtain a reordering.

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

COO (AoS) storage format
------------------------
The Coordinate (COO) Array of Structure (AoS) storage format represents a :math:`m \times n` matrix by

======= ==========================================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
coo_val     array of ``nnz`` elements containing the data (floating point).
coo_ind     array of ``2 * nnz`` elements containing alternating row and column indices (integer).
======= ==========================================================================================

The COO (AoS) matrix is expected to be sorted by row indices and column indices per row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding COO (AoS) structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using zero based indexing:

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
    \text{coo_ind}[16] & = \{0, 0, 0, 1, 0, 3, 1, 1, 1, 2, 2, 0, 2, 3, 2, 4\} \\
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

CSC storage format
------------------
The Compressed Sparse Column (CSC) storage format represents a :math:`m \times n` matrix by

=========== =========================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
csc_val     array of ``nnz`` elements containing the data (floating point).
csc_col_ptr array of ``n+1`` elements that point to the start of every column (integer).
csc_row_ind array of ``nnz`` elements containing the row indices (integer).
=========== =========================================================================

The CSC matrix is expected to be sorted by row indices within each column. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding CSC structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using one based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{csc_val}[8] & = \{1.0, 6.0, 2.0, 4.0, 5.0, 3.0, 7.0, 8.0\} \\
    \text{csc_col_ptr}[6] & = \{1, 3, 5, 6, 8, 9\} \\
    \text{csc_row_ind}[8] & = \{1, 3, 1, 2, 2, 1, 3, 3\}
  \end{array}

BSR storage format
------------------
The Block Compressed Sparse Row (BSR) storage format represents a :math:`(mb \cdot \text{bsr_dim}) \times (nb \cdot \text{bsr_dim})` matrix by

=========== ====================================================================================================================================
mb          number of block rows (integer)
nb          number of block columns (integer)
nnzb        number of non-zero blocks (integer)
bsr_val     array of ``nnzb * bsr_dim * bsr_dim`` elements containing the data (floating point). Blocks can be stored column-major or row-major.
bsr_row_ptr array of ``mb+1`` elements that point to the start of every block row (integer).
bsr_col_ind array of ``nnzb`` elements containing the block column indices (integer).
bsr_dim     dimension of each block (integer).
=========== ====================================================================================================================================

The BSR matrix is expected to be sorted by column indices within each row. If :math:`m` or :math:`n` are not evenly divisible by the block dimension, then zeros are padded to the matrix, such that :math:`mb = (m + \text{bsr_dim} - 1) / \text{bsr_dim}` and :math:`nb = (n + \text{bsr_dim} - 1) / \text{bsr_dim}`.
Consider the following :math:`4 \times 3` matrix and the corresponding BSR structures, with :math:`\text{bsr_dim} = 2, mb = 2, nb = 2` and :math:`\text{nnzb} = 4` using zero based indexing and column-major storage:

.. math::

  A = \begin{pmatrix}
        1.0 & 0.0 & 2.0 \\
        3.0 & 0.0 & 4.0 \\
        5.0 & 6.0 & 0.0 \\
        7.0 & 0.0 & 8.0 \\
      \end{pmatrix}

with the blocks :math:`A_{ij}`

.. math::

  A_{00} = \begin{pmatrix}
             1.0 & 0.0 \\
             3.0 & 0.0 \\
           \end{pmatrix},
  A_{01} = \begin{pmatrix}
             2.0 & 0.0 \\
             4.0 & 0.0 \\
           \end{pmatrix},
  A_{10} = \begin{pmatrix}
             5.0 & 6.0 \\
             7.0 & 0.0 \\
           \end{pmatrix},
  A_{11} = \begin{pmatrix}
             0.0 & 0.0 \\
             8.0 & 0.0 \\
           \end{pmatrix}

such that

.. math::

  A = \begin{pmatrix}
        A_{00} & A_{01} \\
        A_{10} & A_{11} \\
      \end{pmatrix}

with arrays representation

.. math::

  \begin{array}{ll}
    \text{bsr_val}[16] & = \{1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0, 0.0, 0.0\} \\
    \text{bsr_row_ptr}[3] & = \{0, 2, 4\} \\
    \text{bsr_col_ind}[4] & = \{0, 1, 0, 1\}
  \end{array}

GEBSR storage format
--------------------
The General Block Compressed Sparse Row (GEBSR) storage format represents a :math:`(mb \cdot \text{bsr_row_dim}) \times (nb \cdot \text{bsr_col_dim})` matrix by

=========== ====================================================================================================================================
mb          number of block rows (integer)
nb          number of block columns (integer)
nnzb        number of non-zero blocks (integer)
bsr_val     array of ``nnzb * bsr_row_dim * bsr_col_dim`` elements containing the data (floating point). Blocks can be stored column-major or row-major.
bsr_row_ptr array of ``mb+1`` elements that point to the start of every block row (integer).
bsr_col_ind array of ``nnzb`` elements containing the block column indices (integer).
bsr_row_dim row dimension of each block (integer).
bsr_col_dim column dimension of each block (integer).
=========== ====================================================================================================================================

The GEBSR matrix is expected to be sorted by column indices within each row. If :math:`m` is not evenly divisible by the row block dimension or :math:`n` is not evenly
divisible by the column block dimension, then zeros are padded to the matrix, such that :math:`mb = (m + \text{bsr_row_dim} - 1) / \text{bsr_row_dim}` and
:math:`nb = (n + \text{bsr_col_dim} - 1) / \text{bsr_col_dim}`. Consider the following :math:`4 \times 5` matrix and the corresponding GEBSR structures,
with :math:`\text{bsr_row_dim} = 2`, :math:`\text{bsr_col_dim} = 3`, mb = 2, nb = 2` and :math:`\text{nnzb} = 4` using zero based indexing and column-major storage:

.. math::

  A = \begin{pmatrix}
        1.0 & 0.0 & 0.0 & 2.0 & 0.0 \\
        3.0 & 0.0 & 4.0 & 0.0 & 0.0 \\
        5.0 & 6.0 & 0.0 & 7.0 & 0.0 \\
        0.0 & 0.0 & 8.0 & 0.0 & 9.0 \\
      \end{pmatrix}

with the blocks :math:`A_{ij}`

.. math::

  A_{00} = \begin{pmatrix}
             1.0 & 0.0 & 0.0 \\
             3.0 & 0.0 & 4.0 \\
           \end{pmatrix},
  A_{01} = \begin{pmatrix}
             2.0 & 0.0 & 0.0 \\
             0.0 & 0.0 & 0.0 \\
           \end{pmatrix},
  A_{10} = \begin{pmatrix}
             5.0 & 6.0 & 0.0 \\
             0.0 & 0.0 & 8.0 \\
           \end{pmatrix},
  A_{11} = \begin{pmatrix}
             7.0 & 0.0 & 0.0 \\
             0.0 & 9.0 & 0.0 \\
           \end{pmatrix}

such that

.. math::

  A = \begin{pmatrix}
        A_{00} & A_{01} \\
        A_{10} & A_{11} \\
      \end{pmatrix}

with arrays representation

.. math::

  \begin{array}{ll}
    \text{bsr_val}[24] & = \{1.0, 3.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 8.0, 7.0, 0.0, 0.0, 9.0, 0.0, 0.0\} \\
    \text{bsr_row_ptr}[3] & = \{0, 2, 4\} \\
    \text{bsr_col_ind}[4] & = \{0, 1, 0, 1\}
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

.. _rocsparse_direction_:

rocsparse_direction
-------------------

.. doxygenenum:: rocsparse_direction

.. _rocsparse_hyb_partition_:

rocsparse_hyb_partition
-----------------------

.. doxygenenum:: rocsparse_hyb_partition

.. _rocsparse_index_base_:

rocsparse_index_base
--------------------

.. doxygenenum:: rocsparse_index_base

.. _rocsparse_matrix_type_:

rocsparse_matrix_type
---------------------

.. doxygenenum:: rocsparse_matrix_type

.. _rocsparse_fill_mode_:

rocsparse_fill_mode
-------------------

.. doxygenenum:: rocsparse_fill_mode

.. _rocsparse_storage_mode_:

rocsparse_storage_mode
----------------------

.. doxygenenum:: rocsparse_storage_mode

.. _rocsparse_diag_type_:

rocsparse_diag_type
-------------------

.. doxygenenum:: rocsparse_diag_type

.. _rocsparse_operation_:

rocsparse_operation
-------------------

.. doxygenenum:: rocsparse_operation

.. _rocsparse_pointer_mode_:

rocsparse_pointer_mode
----------------------

.. doxygenenum:: rocsparse_pointer_mode

.. _rocsparse_analysis_policy_:

rocsparse_analysis_policy
-------------------------

.. doxygenenum:: rocsparse_analysis_policy

.. _rocsparse_solve_policy_:

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

rocsparse_indextype
-------------------

.. doxygenenum:: rocsparse_indextype

rocsparse_datatype
------------------

.. doxygenenum:: rocsparse_datatype

rocsparse_format
----------------

.. doxygenenum:: rocsparse_format

rocsparse_order
---------------

.. doxygenenum:: rocsparse_order

rocsparse_spmv_alg
------------------

.. doxygenenum:: rocsparse_spmv_alg

rocsparse_spmv_stage
--------------------

.. doxygenenum:: rocsparse_spmv_stage


rocsparse_spsv_alg
------------------

.. doxygenenum:: rocsparse_spsv_alg

rocsparse_spsv_stage
--------------------

.. doxygenenum:: rocsparse_spsv_stage

rocsparse_spsm_alg
------------------

.. doxygenenum:: rocsparse_spsm_alg

rocsparse_spsm_stage
--------------------

.. doxygenenum:: rocsparse_spsm_stage

rocsparse_spmm_alg
------------------

.. doxygenenum:: rocsparse_spmm_alg


rocsparse_spmm_stage
--------------------

.. doxygenenum:: rocsparse_spmm_stage


rocsparse_sddmm_alg
-------------------

.. doxygenenum:: rocsparse_sddmm_alg

rocsparse_spgemm_stage
----------------------

.. doxygenenum:: rocsparse_spgemm_stage

rocsparse_spgemm_alg
--------------------

.. doxygenenum:: rocsparse_spgemm_alg


rocsparse_sparse_to_dense_alg
-----------------------------

.. doxygenenum:: rocsparse_sparse_to_dense_alg

rocsparse_dense_to_sparse_alg
-----------------------------

.. doxygenenum:: rocsparse_dense_to_sparse_alg

rocsparse_gtsv_interleaved_alg
------------------------------

.. doxygenenum:: rocsparse_gtsv_interleaved_alg

.. _rocsparse_logging:

Logging
=======
Three different environment variables can be set to enable logging in rocSPARSE: ``ROCSPARSE_LAYER``, ``ROCSPARSE_LOG_TRACE_PATH``, ``ROCSPARSE_LOG_BENCH_PATH`` and ``ROCSPARSE_LOG_DEBUG_PATH``.

``ROCSPARSE_LAYER`` is a bit mask, where several logging modes (:ref:`rocsparse_layer_mode_`) can be combined as follows:

================================  =============================================================
``ROCSPARSE_LAYER`` unset         logging is disabled.
``ROCSPARSE_LAYER`` set to ``1``  trace logging is enabled.
``ROCSPARSE_LAYER`` set to ``2``  bench logging is enabled.
``ROCSPARSE_LAYER`` set to ``3``  trace logging and bench logging is enabled.
``ROCSPARSE_LAYER`` set to ``4``  debug logging is enabled.
``ROCSPARSE_LAYER`` set to ``5``  trace logging and debug logging is enabled.
``ROCSPARSE_LAYER`` set to ``6``  bench logging and debug logging is enabled.
``ROCSPARSE_LAYER`` set to ``7``  trace logging and bench logging and debug logging is enabled.
================================  =============================================================

When logging is enabled, each rocSPARSE function call will write the function name as well as function arguments to the logging stream. The default logging stream is ``stderr``.

If the user sets the environment variable ``ROCSPARSE_LOG_TRACE_PATH`` to the full path name for a file, the file is opened and trace logging is streamed to that file. If the user sets the environment variable ``ROCSPARSE_LOG_BENCH_PATH`` to the full path name for a file, the file is opened and bench logging is streamed to that file. If the file cannot be opened, logging output is stream to ``stderr``.

Note that performance will degrade when logging is enabled. By default, the environment variable ``ROCSPARSE_LAYER`` is unset and logging is disabled.

.. _api:

Exported Sparse Functions
=========================

Auxiliary Functions
-------------------

+---------------------------------------------+
|Function name                                |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_handle`          |
+---------------------------------------------+
|:cpp:func:`rocsparse_destroy_handle`         |
+---------------------------------------------+
|:cpp:func:`rocsparse_set_stream`             |
+---------------------------------------------+
|:cpp:func:`rocsparse_get_stream`             |
+---------------------------------------------+
|:cpp:func:`rocsparse_set_pointer_mode`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_get_pointer_mode`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_get_version`            |
+---------------------------------------------+
|:cpp:func:`rocsparse_get_git_rev`            |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_mat_descr`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_destroy_mat_descr`      |
+---------------------------------------------+
|:cpp:func:`rocsparse_copy_mat_descr`         |
+---------------------------------------------+
|:cpp:func:`rocsparse_set_mat_index_base`     |
+---------------------------------------------+
|:cpp:func:`rocsparse_get_mat_index_base`     |
+---------------------------------------------+
|:cpp:func:`rocsparse_set_mat_type`           |
+---------------------------------------------+
|:cpp:func:`rocsparse_get_mat_type`           |
+---------------------------------------------+
|:cpp:func:`rocsparse_set_mat_fill_mode`      |
+---------------------------------------------+
|:cpp:func:`rocsparse_get_mat_fill_mode`      |
+---------------------------------------------+
|:cpp:func:`rocsparse_set_mat_diag_type`      |
+---------------------------------------------+
|:cpp:func:`rocsparse_get_mat_diag_type`      |
+---------------------------------------------+
|:cpp:func:`rocsparse_set_mat_storage_mode`   |
+---------------------------------------------+
|:cpp:func:`rocsparse_get_mat_storage_mode`   |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_hyb_mat`         |
+---------------------------------------------+
|:cpp:func:`rocsparse_destroy_hyb_mat`        |
+---------------------------------------------+
|:cpp:func:`rocsparse_copy_hyb_mat`           |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_mat_info`        |
+---------------------------------------------+
|:cpp:func:`rocsparse_copy_mat_info`          |
+---------------------------------------------+
|:cpp:func:`rocsparse_destroy_mat_info`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_color_info`      |
+---------------------------------------------+
|:cpp:func:`rocsparse_destroy_color_info`     |
+---------------------------------------------+
|:cpp:func:`rocsparse_copy_color_info`        |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_spvec_descr`     |
+---------------------------------------------+
|:cpp:func:`rocsparse_destroy_spvec_descr`    |
+---------------------------------------------+
|:cpp:func:`rocsparse_spvec_get`              |
+---------------------------------------------+
|:cpp:func:`rocsparse_spvec_get_index_base`   |
+---------------------------------------------+
|:cpp:func:`rocsparse_spvec_get_values`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_spvec_set_values`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_coo_descr`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_coo_aos_descr`   |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_csr_descr`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_csc_descr`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_ell_descr`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_bell_descr`      |
+---------------------------------------------+
|:cpp:func:`rocsparse_destroy_spmat_descr`    |
+---------------------------------------------+
|:cpp:func:`rocsparse_coo_get`                |
+---------------------------------------------+
|:cpp:func:`rocsparse_coo_aos_get`            |
+---------------------------------------------+
|:cpp:func:`rocsparse_csr_get`                |
+---------------------------------------------+
|:cpp:func:`rocsparse_ell_get`                |
+---------------------------------------------+
|:cpp:func:`rocsparse_bell_get`               |
+---------------------------------------------+
|:cpp:func:`rocsparse_coo_set_pointers`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_coo_aos_set_pointers`   |
+---------------------------------------------+
|:cpp:func:`rocsparse_csr_set_pointers`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_csc_set_pointers`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_ell_set_pointers`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_size`         |
+---------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_format`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_index_base`   |
+---------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_values`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_spmat_set_values`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_strided_batch`|
+---------------------------------------------+
|:cpp:func:`rocsparse_spmat_set_strided_batch`|
+---------------------------------------------+
|:cpp:func:`rocsparse_coo_set_strided_batch`  |
+---------------------------------------------+
|:cpp:func:`rocsparse_csr_set_strided_batch`  |
+---------------------------------------------+
|:cpp:func:`rocsparse_csc_set_strided_batch`  |
+---------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_attribute`    |
+---------------------------------------------+
|:cpp:func:`rocsparse_spmat_set_attribute`    |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_dnvec_descr`     |
+---------------------------------------------+
|:cpp:func:`rocsparse_destroy_dnvec_descr`    |
+---------------------------------------------+
|:cpp:func:`rocsparse_dnvec_get`              |
+---------------------------------------------+
|:cpp:func:`rocsparse_dnvec_get_values`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_dnvec_set_values`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_create_dnmat_descr`     |
+---------------------------------------------+
|:cpp:func:`rocsparse_destroy_dnmat_descr`    |
+---------------------------------------------+
|:cpp:func:`rocsparse_dnmat_get`              |
+---------------------------------------------+
|:cpp:func:`rocsparse_dnmat_get_values`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_dnmat_set_values`       |
+---------------------------------------------+
|:cpp:func:`rocsparse_dnmat_get_strided_batch`|
+---------------------------------------------+
|:cpp:func:`rocsparse_dnmat_set_strided_batch`|
+---------------------------------------------+

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
:cpp:func:`rocsparse_Xbsrmv_ex_analysis() <rocsparse_sbsrmv_ex_analysis>` x      x      x              x
:cpp:func:`rocsparse_bsrmv_ex_clear`
:cpp:func:`rocsparse_Xbsrmv_ex() <rocsparse_sbsrmv_ex>`                   x      x      x              x
:cpp:func:`rocsparse_Xbsrmv() <rocsparse_sbsrmv>`                         x      x      x              x
:cpp:func:`rocsparse_Xbsrxmv() <rocsparse_sbsrxmv>`                       x      x      x              x
:cpp:func:`rocsparse_Xbsrsv_buffer_size() <rocsparse_sbsrsv_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xbsrsv_analysis() <rocsparse_sbsrsv_analysis>`       x      x      x              x
:cpp:func:`rocsparse_bsrsv_zero_pivot`
:cpp:func:`rocsparse_bsrsv_clear`
:cpp:func:`rocsparse_Xbsrsv_solve() <rocsparse_sbsrsv_solve>`             x      x      x              x
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
:cpp:func:`rocsparse_Xgebsrmv() <rocsparse_sgebsrmv>`                     x      x      x              x
:cpp:func:`rocsparse_Xgemvi_buffer_size() <rocsparse_sgemvi_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xgemvi() <rocsparse_sgemvi>`                         x      x      x              x
========================================================================= ====== ====== ============== ==============

Sparse Level 3 Functions
------------------------

========================================================================= ====== ====== ============== ==============
Function name                                                             single double single complex double complex
========================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xbsrmm() <rocsparse_sbsrmm>`                         x      x      x              x
:cpp:func:`rocsparse_Xgebsrmm() <rocsparse_sgebsrmm>`                     x      x      x              x
:cpp:func:`rocsparse_Xcsrmm() <rocsparse_scsrmm>`                         x      x      x              x
:cpp:func:`rocsparse_Xcsrsm_buffer_size() <rocsparse_scsrsm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcsrsm_analysis() <rocsparse_scsrsm_analysis>`       x      x      x              x
:cpp:func:`rocsparse_csrsm_zero_pivot`
:cpp:func:`rocsparse_csrsm_clear`
:cpp:func:`rocsparse_Xcsrsm_solve() <rocsparse_scsrsm_solve>`             x      x      x              x
:cpp:func:`rocsparse_Xbsrsm_buffer_size() <rocsparse_sbsrsm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xbsrsm_analysis() <rocsparse_sbsrsm_analysis>`       x      x      x              x
:cpp:func:`rocsparse_bsrsm_zero_pivot`
:cpp:func:`rocsparse_bsrsm_clear`
:cpp:func:`rocsparse_Xbsrsm_solve() <rocsparse_sbsrsm_solve>`             x      x      x              x
:cpp:func:`rocsparse_Xgemmi() <rocsparse_sgemmi>`                         x      x      x              x
========================================================================= ====== ====== ============== ==============

Sparse Extra Functions
----------------------

============================================================================= ====== ====== ============== ==============
Function name                                                                 single double single complex double complex
============================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_bsrgeam_nnzb`
:cpp:func:`rocsparse_Xbsrgeam() <rocsparse_sbsrgeam>`                         x      x      x              x
:cpp:func:`rocsparse_csrgeam_nnz`
:cpp:func:`rocsparse_Xcsrgeam() <rocsparse_scsrgeam>`                         x      x      x              x
:cpp:func:`rocsparse_Xcsrgemm_buffer_size() <rocsparse_scsrgemm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_csrgemm_nnz`
:cpp:func:`rocsparse_csrgemm_symbolic`
:cpp:func:`rocsparse_Xcsrgemm() <rocsparse_scsrgemm>`                         x      x      x              x
:cpp:func:`rocsparse_Xcsrgemm_numeric() <rocsparse_scsrgemm_numeric>`         x      x      x              x
============================================================================= ====== ====== ============== ==============

Preconditioner Functions
------------------------

===================================================================================================================== ====== ====== ============== ==============
Function name                                                                                                         single double single complex double complex
===================================================================================================================== ====== ====== ============== ==============
:cpp:func:`rocsparse_Xbsric0_buffer_size() <rocsparse_sbsric0_buffer_size>`                                           x      x      x              x
:cpp:func:`rocsparse_Xbsric0_analysis() <rocsparse_sbsric0_analysis>`                                                 x      x      x              x
:cpp:func:`rocsparse_bsric0_zero_pivot`
:cpp:func:`rocsparse_bsric0_clear`
:cpp:func:`rocsparse_Xbsric0() <rocsparse_sbsric0>`                                                                   x      x      x              x
:cpp:func:`rocsparse_Xbsrilu0_buffer_size() <rocsparse_sbsrilu0_buffer_size>`                                         x      x      x              x
:cpp:func:`rocsparse_Xbsrilu0_analysis() <rocsparse_sbsrilu0_analysis>`                                               x      x      x              x
:cpp:func:`rocsparse_bsrilu0_zero_pivot`
:cpp:func:`rocsparse_Xbsrilu0_numeric_boost() <rocsparse_sbsrilu0_numeric_boost>`                                     x      x      x              x
:cpp:func:`rocsparse_bsrilu0_clear`
:cpp:func:`rocsparse_Xbsrilu0() <rocsparse_sbsrilu0>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xcsric0_buffer_size() <rocsparse_scsric0_buffer_size>`                                           x      x      x              x
:cpp:func:`rocsparse_Xcsric0_analysis() <rocsparse_scsric0_analysis>`                                                 x      x      x              x
:cpp:func:`rocsparse_csric0_zero_pivot`
:cpp:func:`rocsparse_csric0_clear`
:cpp:func:`rocsparse_Xcsric0() <rocsparse_scsric0>`                                                                   x      x      x              x
:cpp:func:`rocsparse_Xcsrilu0_buffer_size() <rocsparse_scsrilu0_buffer_size>`                                         x      x      x              x
:cpp:func:`rocsparse_Xcsrilu0_numeric_boost() <rocsparse_scsrilu0_numeric_boost>`                                     x      x      x              x
:cpp:func:`rocsparse_Xcsrilu0_analysis() <rocsparse_scsrilu0_analysis>`                                               x      x      x              x
:cpp:func:`rocsparse_csrilu0_zero_pivot`
:cpp:func:`rocsparse_csrilu0_clear`
:cpp:func:`rocsparse_Xcsrilu0() <rocsparse_scsrilu0>`                                                                 x      x      x              x
:cpp:func:`rocsparse_csritilu0_buffer_size`
:cpp:func:`rocsparse_csritilu0_preprocess`
:cpp:func:`rocsparse_Xcsritilu0_compute() <rocsparse_scsritilu0_compute>`                                             x      x      x              x
:cpp:func:`rocsparse_Xcsritilu0_history() <rocsparse_scsritilu0_history>`                                             x      x      x              x
:cpp:func:`rocsparse_Xgtsv_buffer_size() <rocsparse_sgtsv_buffer_size>`                                               x      x      x              x
:cpp:func:`rocsparse_Xgtsv() <rocsparse_sgtsv>`                                                                       x      x      x              x
:cpp:func:`rocsparse_Xgtsv_no_pivot_buffer_size() <rocsparse_sgtsv_no_pivot_buffer_size>`                             x      x      x              x
:cpp:func:`rocsparse_Xgtsv_no_pivot() <rocsparse_sgtsv_no_pivot>`                                                     x      x      x              x
:cpp:func:`rocsparse_Xgtsv_no_pivot_strided_batch_buffer_size() <rocsparse_sgtsv_no_pivot_strided_batch_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xgtsv_no_pivot_strided_batch() <rocsparse_sgtsv_no_pivot_strided_batch>`                         x      x      x              x
:cpp:func:`rocsparse_Xgtsv_interleaved_batch_buffer_size() <rocsparse_sgtsv_interleaved_batch_buffer_size>`           x      x      x              x
:cpp:func:`rocsparse_Xgtsv_interleaved_batch() <rocsparse_sgtsv_interleaved_batch>`                                   x      x      x              x
:cpp:func:`rocsparse_Xgpsv_interleaved_batch_buffer_size() <rocsparse_sgpsv_interleaved_batch_buffer_size>`           x      x      x              x
:cpp:func:`rocsparse_Xgpsv_interleaved_batch() <rocsparse_sgpsv_interleaved_batch>`                                   x      x      x              x
===================================================================================================================== ====== ====== ============== ==============

Conversion Functions
--------------------

========================================================================================================================= ====== ====== ============== ==============
Function name                                                                                                             single double single complex double complex
========================================================================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_csr2coo`
:cpp:func:`rocsparse_csr2csc_buffer_size`
:cpp:func:`rocsparse_Xcsr2csc() <rocsparse_scsr2csc>`                                                                     x      x      x              x
:cpp:func:`rocsparse_Xgebsr2gebsc_buffer_size`                                                                            x      x      x              x
:cpp:func:`rocsparse_Xgebsr2gebsc() <rocsparse_sgebsr2gebsc>`                                                             x      x      x              x
:cpp:func:`rocsparse_csr2ell_width`
:cpp:func:`rocsparse_Xcsr2ell() <rocsparse_scsr2ell>`                                                                     x      x      x              x
:cpp:func:`rocsparse_Xcsr2hyb() <rocsparse_scsr2hyb>`                                                                     x      x      x              x
:cpp:func:`rocsparse_csr2bsr_nnz`
:cpp:func:`rocsparse_Xcsr2bsr() <rocsparse_scsr2bsr>`                                                                     x      x      x              x
:cpp:func:`rocsparse_csr2gebsr_nnz`
:cpp:func:`rocsparse_Xcsr2gebsr_buffer_size() <rocsparse_scsr2gebsr_buffer_size>`                                         x      x      x              x
:cpp:func:`rocsparse_Xcsr2gebsr() <rocsparse_scsr2gebsr>`                                                                 x      x      x              x
:cpp:func:`rocsparse_coo2csr`
:cpp:func:`rocsparse_ell2csr_nnz`
:cpp:func:`rocsparse_Xell2csr() <rocsparse_sell2csr>`                                                                     x      x      x              x
:cpp:func:`rocsparse_hyb2csr_buffer_size`
:cpp:func:`rocsparse_Xhyb2csr() <rocsparse_shyb2csr>`                                                                     x      x      x              x
:cpp:func:`rocsparse_Xbsr2csr() <rocsparse_sbsr2csr>`                                                                     x      x      x              x
:cpp:func:`rocsparse_Xgebsr2csr() <rocsparse_sgebsr2csr>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xgebsr2gebsr_buffer_size() <rocsparse_sgebsr2gebsr_buffer_size>`                                     x      x      x              x
:cpp:func:`rocsparse_gebsr2gebsr_nnz()`
:cpp:func:`rocsparse_Xgebsr2gebsr() <rocsparse_sgebsr2gebsr>`                                                             x      x      x              x
:cpp:func:`rocsparse_Xcsr2csr_compress() <rocsparse_scsr2csr_compress>`                                                   x      x      x              x
:cpp:func:`rocsparse_create_identity_permutation`
:cpp:func:`rocsparse_create_inverse_permutation`
:cpp:func:`rocsparse_cscsort_buffer_size`
:cpp:func:`rocsparse_cscsort`
:cpp:func:`rocsparse_csrsort_buffer_size`
:cpp:func:`rocsparse_csrsort`
:cpp:func:`rocsparse_coosort_buffer_size`
:cpp:func:`rocsparse_coosort_by_row`
:cpp:func:`rocsparse_coosort_by_column`
:cpp:func:`rocsparse_Xdense2csr() <rocsparse_sdense2csr>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xdense2csc() <rocsparse_sdense2csc>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xdense2coo() <rocsparse_sdense2coo>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xcsr2dense() <rocsparse_scsr2dense>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xcsc2dense() <rocsparse_scsc2dense>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xcoo2dense() <rocsparse_scoo2dense>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xnnz_compress() <rocsparse_snnz_compress>`                                                           x      x      x              x
:cpp:func:`rocsparse_Xnnz() <rocsparse_snnz>`                                                                             x      x      x              x
:cpp:func:`rocsparse_Xprune_dense2csr_buffer_size() <rocsparse_sprune_dense2csr_buffer_size>`                             x      x
:cpp:func:`rocsparse_Xprune_dense2csr_nnz() <rocsparse_sprune_dense2csr_nnz>`                                             x      x
:cpp:func:`rocsparse_Xprune_dense2csr() <rocsparse_sprune_dense2csr>`                                                     x      x
:cpp:func:`rocsparse_Xprune_csr2csr_buffer_size() <rocsparse_sprune_csr2csr_buffer_size>`                                 x      x
:cpp:func:`rocsparse_Xprune_csr2csr_nnz() <rocsparse_sprune_csr2csr_nnz>`                                                 x      x
:cpp:func:`rocsparse_Xprune_csr2csr() <rocsparse_sprune_csr2csr>`                                                         x      x
:cpp:func:`rocsparse_Xprune_dense2csr_by_percentage_buffer_size() <rocsparse_sprune_dense2csr_by_percentage_buffer_size>` x      x
:cpp:func:`rocsparse_Xprune_dense2csr_nnz_by_percentage() <rocsparse_sprune_dense2csr_nnz_by_percentage>`                 x      x
:cpp:func:`rocsparse_Xprune_dense2csr_by_percentage() <rocsparse_sprune_dense2csr_by_percentage>`                         x      x
:cpp:func:`rocsparse_Xprune_csr2csr_by_percentage_buffer_size() <rocsparse_sprune_csr2csr_by_percentage_buffer_size>`     x      x
:cpp:func:`rocsparse_Xprune_csr2csr_nnz_by_percentage() <rocsparse_sprune_csr2csr_nnz_by_percentage>`                     x      x
:cpp:func:`rocsparse_Xprune_csr2csr_by_percentage() <rocsparse_sprune_csr2csr_by_percentage>`                             x      x
:cpp:func:`rocsparse_Xbsrpad_value() <rocsparse_sbsrpad_value>`                                                           x      x      x              x
========================================================================================================================= ====== ====== ============== ==============

Reordering Functions
--------------------

======================================================= ====== ====== ============== ==============
Function name                                           single double single complex double complex
======================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xcsrcolor() <rocsparse_scsrcolor>` x      x      x              x
======================================================= ====== ====== ============== ==============

Utility Functions
-----------------

=================================================================================================== ====== ====== ============== ==============
Function name                                                                                       single double single complex double complex
=================================================================================================== ====== ====== ============== ==============
:cpp:func:`rocsparse_Xcheck_matrix_csr_buffer_size() <rocsparse_scheck_matrix_csr_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_csr() <rocsparse_scheck_matrix_csr>`                             x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_csc_buffer_size() <rocsparse_scheck_matrix_csc_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_csc() <rocsparse_scheck_matrix_csc>`                             x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_coo_buffer_size() <rocsparse_scheck_matrix_coo_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_coo() <rocsparse_scheck_matrix_coo>`                             x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_gebsr_buffer_size() <rocsparse_scheck_matrix_gebsr_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_gebsr() <rocsparse_scheck_matrix_gebsr>`                         x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_gebsc_buffer_size() <rocsparse_scheck_matrix_gebsc_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_gebsc() <rocsparse_scheck_matrix_gebsc>`                         x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_ell_buffer_size() <rocsparse_scheck_matrix_ell_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_ell() <rocsparse_scheck_matrix_ell>`                             x      x      x              x
:cpp:func:`rocsparse_check_matrix_hyb_buffer_size() <rocsparse_check_matrix_hyb_buffer_size>`       x      x      x              x
:cpp:func:`rocsparse_check_matrix_hyb() <rocsparse_check_matrix_hyb>`                               x      x      x              x
=================================================================================================== ====== ====== ============== ==============

Sparse Generic Functions
------------------------

========================================= ====== ====== ============== ==============
Function name                             single double single complex double complex
========================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_axpby()`             x      x      x              x
:cpp:func:`rocsparse_gather()`            x      x      x              x
:cpp:func:`rocsparse_scatter()`           x      x      x              x
:cpp:func:`rocsparse_rot()`               x      x      x              x
:cpp:func:`rocsparse_spvv()`              x      x      x              x
:cpp:func:`rocsparse_sparse_to_dense()`   x      x      x              x
:cpp:func:`rocsparse_dense_to_sparse()`   x      x      x              x
:cpp:func:`rocsparse_spmv()`              x      x      x              x
:cpp:func:`rocsparse_spmv_ex()`           x      x      x              x
:cpp:func:`rocsparse_spsv()`              x      x      x              x
:cpp:func:`rocsparse_spmm()`              x      x      x              x
:cpp:func:`rocsparse_spmm_ex()`           x      x      x              x
:cpp:func:`rocsparse_spsm()`              x      x      x              x
:cpp:func:`rocsparse_spgemm()`            x      x      x              x
:cpp:func:`rocsparse_sddmm_buffer_size()` x      x      x              x
:cpp:func:`rocsparse_sddmm_preprocess()`  x      x      x              x
:cpp:func:`rocsparse_sddmm()`             x      x      x              x
========================================= ====== ====== ============== ==============


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

rocsparse_set_mat_storage_mode()
--------------------------------

.. doxygenfunction:: rocsparse_set_mat_storage_mode

rocsparse_get_mat_storage_mode()
--------------------------------

.. doxygenfunction:: rocsparse_get_mat_storage_mode

.. _rocsparse_create_hyb_mat_:

rocsparse_create_hyb_mat()
--------------------------

.. doxygenfunction:: rocsparse_create_hyb_mat

rocsparse_destroy_hyb_mat()
---------------------------

.. doxygenfunction:: rocsparse_destroy_hyb_mat

rocsparse_copy_hyb_mat()
------------------------

.. doxygenfunction:: rocsparse_copy_hyb_mat

rocsparse_create_mat_info()
---------------------------

.. doxygenfunction:: rocsparse_create_mat_info

rocsparse_copy_mat_info()
-------------------------

.. doxygenfunction:: rocsparse_copy_mat_info

.. _rocsparse_destroy_mat_info_:

rocsparse_destroy_mat_info()
----------------------------

.. doxygenfunction:: rocsparse_destroy_mat_info

rocsparse_create_color_info()
-----------------------------

.. doxygenfunction:: rocsparse_create_color_info

rocsparse_destroy_color_info()
------------------------------

.. doxygenfunction:: rocsparse_destroy_color_info

rocsparse_copy_color_info()
---------------------------

.. doxygenfunction:: rocsparse_copy_color_info

rocsparse_create_spvec_descr()
------------------------------

.. doxygenfunction:: rocsparse_create_spvec_descr

rocsparse_destroy_spvec_descr()
-------------------------------

.. doxygenfunction:: rocsparse_destroy_spvec_descr

rocsparse_spvec_get()
---------------------

.. doxygenfunction:: rocsparse_spvec_get

rocsparse_spvec_get_index_base()
--------------------------------

.. doxygenfunction:: rocsparse_spvec_get_index_base

rocsparse_spvec_get_values()
----------------------------

.. doxygenfunction:: rocsparse_spvec_get_values

rocsparse_spvec_set_values()
----------------------------

.. doxygenfunction:: rocsparse_spvec_set_values

rocsparse_create_coo_descr
--------------------------

.. doxygenfunction:: rocsparse_create_coo_descr

rocsparse_create_coo_aos_descr
------------------------------

.. doxygenfunction:: rocsparse_create_coo_aos_descr

rocsparse_create_csr_descr
--------------------------

.. doxygenfunction:: rocsparse_create_csr_descr

rocsparse_create_csc_descr
--------------------------

.. doxygenfunction:: rocsparse_create_csc_descr

rocsparse_create_ell_descr
--------------------------

.. doxygenfunction:: rocsparse_create_ell_descr

rocsparse_create_bell_descr
---------------------------

.. doxygenfunction:: rocsparse_create_bell_descr

rocsparse_destroy_spmat_descr
-----------------------------

.. doxygenfunction:: rocsparse_destroy_spmat_descr

rocsparse_coo_get
-----------------

.. doxygenfunction:: rocsparse_coo_get

rocsparse_coo_aos_get
---------------------

.. doxygenfunction:: rocsparse_coo_aos_get

rocsparse_csr_get
-----------------

.. doxygenfunction:: rocsparse_csr_get

rocsparse_ell_get
-----------------

.. doxygenfunction:: rocsparse_ell_get

rocsparse_bell_get
------------------

.. doxygenfunction:: rocsparse_bell_get

rocsparse_coo_set_pointers
--------------------------

.. doxygenfunction:: rocsparse_coo_set_pointers

rocsparse_coo_aos_set_pointers
------------------------------

.. doxygenfunction:: rocsparse_coo_aos_set_pointers

rocsparse_csr_set_pointers
--------------------------

.. doxygenfunction:: rocsparse_csr_set_pointers

rocsparse_csc_set_pointers
--------------------------

.. doxygenfunction:: rocsparse_csc_set_pointers

rocsparse_ell_set_pointers
--------------------------

.. doxygenfunction:: rocsparse_ell_set_pointers

rocsparse_spmat_get_size
------------------------

.. doxygenfunction:: rocsparse_spmat_get_size

rocsparse_spmat_get_format
--------------------------

.. doxygenfunction:: rocsparse_spmat_get_format

rocsparse_spmat_get_index_base
------------------------------

.. doxygenfunction:: rocsparse_spmat_get_index_base

rocsparse_spmat_get_values
--------------------------

.. doxygenfunction:: rocsparse_spmat_get_values

rocsparse_spmat_set_values
--------------------------

.. doxygenfunction:: rocsparse_spmat_set_values

rocsparse_spmat_get_strided_batch
---------------------------------

.. doxygenfunction:: rocsparse_spmat_get_strided_batch

rocsparse_spmat_set_strided_batch
---------------------------------

.. doxygenfunction:: rocsparse_spmat_set_strided_batch

rocsparse_coo_set_strided_batch
-------------------------------

.. doxygenfunction:: rocsparse_coo_set_strided_batch

rocsparse_csr_set_strided_batch
-------------------------------

.. doxygenfunction:: rocsparse_csr_set_strided_batch

rocsparse_csc_set_strided_batch
-------------------------------

.. doxygenfunction:: rocsparse_csc_set_strided_batch

rocsparse_spmat_get_attribute
-----------------------------

.. doxygenfunction:: rocsparse_spmat_get_attribute

rocsparse_spmat_set_attribute
-----------------------------

.. doxygenfunction:: rocsparse_spmat_set_attribute

rocsparse_create_dnvec_descr
----------------------------

.. doxygenfunction:: rocsparse_create_dnvec_descr

rocsparse_destroy_dnvec_descr
-----------------------------

.. doxygenfunction:: rocsparse_destroy_dnvec_descr

rocsparse_dnvec_get
-------------------

.. doxygenfunction:: rocsparse_dnvec_get

rocsparse_dnvec_get_values
--------------------------

.. doxygenfunction:: rocsparse_dnvec_get_values

rocsparse_dnvec_set_values
--------------------------

.. doxygenfunction:: rocsparse_dnvec_set_values

rocsparse_create_dnmat_descr
----------------------------

.. doxygenfunction:: rocsparse_create_dnmat_descr

rocsparse_destroy_dnmat_descr
-----------------------------

.. doxygenfunction:: rocsparse_destroy_dnmat_descr

rocsparse_dnmat_get
-------------------

.. doxygenfunction:: rocsparse_dnmat_get

rocsparse_dnmat_get_values
--------------------------

.. doxygenfunction:: rocsparse_dnmat_get_values

rocsparse_dnmat_set_values
--------------------------

.. doxygenfunction:: rocsparse_dnmat_set_values

rocsparse_dnmat_get_strided_batch
---------------------------------

.. doxygenfunction:: rocsparse_dnmat_get_strided_batch

rocsparse_dnmat_set_strided_batch
---------------------------------

.. doxygenfunction:: rocsparse_dnmat_set_strided_batch

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

rocsparse_bsrmv_ex_analysis()
-----------------------------

.. doxygenfunction:: rocsparse_sbsrmv_ex_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsrmv_ex_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsrmv_ex_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsrmv_ex_analysis

rocsparse_bsrmv_ex()
--------------------

.. doxygenfunction:: rocsparse_sbsrmv_ex
  :outline:
.. doxygenfunction:: rocsparse_dbsrmv_ex
  :outline:
.. doxygenfunction:: rocsparse_cbsrmv_ex
  :outline:
.. doxygenfunction:: rocsparse_zbsrmv_ex

rocsparse_bsrmv()
-----------------

.. doxygenfunction:: rocsparse_sbsrmv
  :outline:
.. doxygenfunction:: rocsparse_dbsrmv
  :outline:
.. doxygenfunction:: rocsparse_cbsrmv
  :outline:
.. doxygenfunction:: rocsparse_zbsrmv

rocsparse_bsrxmv()
------------------

.. doxygenfunction:: rocsparse_sbsrxmv
  :outline:
.. doxygenfunction:: rocsparse_dbsrxmv
  :outline:
.. doxygenfunction:: rocsparse_cbsrxmv
  :outline:
.. doxygenfunction:: rocsparse_zbsrxmv

rocsparse_bsrsv_zero_pivot()
----------------------------

.. doxygenfunction:: rocsparse_bsrsv_zero_pivot

rocsparse_bsrsv_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sbsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dbsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cbsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zbsrsv_buffer_size

rocsparse_bsrsv_analysis()
--------------------------

.. doxygenfunction:: rocsparse_sbsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsrsv_analysis

rocsparse_bsrsv_solve()
-----------------------

.. doxygenfunction:: rocsparse_sbsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_dbsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_cbsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_zbsrsv_solve

rocsparse_bsrsv_clear()
-----------------------

.. doxygenfunction:: rocsparse_bsrsv_clear

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

rocsparse_gebsrmv()
-------------------

.. doxygenfunction:: rocsparse_sgebsrmv
  :outline:
.. doxygenfunction:: rocsparse_dgebsrmv
  :outline:
.. doxygenfunction:: rocsparse_cgebsrmv
  :outline:
.. doxygenfunction:: rocsparse_zgebsrmv

rocsparse_gemvi_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sgemvi_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgemvi_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgemvi_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgemvi_buffer_size

rocsparse_gemvi()
-----------------

.. doxygenfunction:: rocsparse_sgemvi
  :outline:
.. doxygenfunction:: rocsparse_dgemvi
  :outline:
.. doxygenfunction:: rocsparse_cgemvi
  :outline:
.. doxygenfunction:: rocsparse_zgemvi

.. _rocsparse_level3_functions_:

Sparse Level 3 Functions
========================

This module holds all sparse level 3 routines.

The sparse level 3 routines describe operations between a matrix in sparse format and multiple vectors in dense format that can also be seen as a dense matrix.

rocsparse_bsrmm()
-----------------

.. doxygenfunction:: rocsparse_sbsrmm
  :outline:
.. doxygenfunction:: rocsparse_dbsrmm
  :outline:
.. doxygenfunction:: rocsparse_cbsrmm
  :outline:
.. doxygenfunction:: rocsparse_zbsrmm


rocsparse_gebsrmm()
-------------------

.. doxygenfunction:: rocsparse_sgebsrmm
  :outline:
.. doxygenfunction:: rocsparse_dgebsrmm
  :outline:
.. doxygenfunction:: rocsparse_cgebsrmm
  :outline:
.. doxygenfunction:: rocsparse_zgebsrmm


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

rocsparse_bsrsm_zero_pivot()
----------------------------

.. doxygenfunction:: rocsparse_bsrsm_zero_pivot

rocsparse_bsrsm_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sbsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dbsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cbsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zbsrsm_buffer_size

rocsparse_bsrsm_analysis()
--------------------------

.. doxygenfunction:: rocsparse_sbsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsrsm_analysis

rocsparse_bsrsm_solve()
-----------------------

.. doxygenfunction:: rocsparse_sbsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_dbsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_cbsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_zbsrsm_solve

rocsparse_bsrsm_clear()
-----------------------

.. doxygenfunction:: rocsparse_bsrsm_clear

rocsparse_gemmi()
-----------------

.. doxygenfunction:: rocsparse_sgemmi
  :outline:
.. doxygenfunction:: rocsparse_dgemmi
  :outline:
.. doxygenfunction:: rocsparse_cgemmi
  :outline:
.. doxygenfunction:: rocsparse_zgemmi

.. _rocsparse_extra_functions_:

Sparse Extra Functions
======================

This module holds all sparse extra routines.

The sparse extra routines describe operations that manipulate sparse matrices.

rocsparse_bsrgeam_nnzb()
------------------------

.. doxygenfunction:: rocsparse_bsrgeam_nnzb

rocsparse_bsrgeam()
-------------------

.. doxygenfunction:: rocsparse_sbsrgeam
  :outline:
.. doxygenfunction:: rocsparse_dbsrgeam
  :outline:
.. doxygenfunction:: rocsparse_cbsrgeam
  :outline:
.. doxygenfunction:: rocsparse_zbsrgeam

rocsparse_csrgeam_nnz()
-----------------------

.. doxygenfunction:: rocsparse_csrgeam_nnz

rocsparse_csrgeam()
-------------------

.. doxygenfunction:: rocsparse_scsrgeam
  :outline:
.. doxygenfunction:: rocsparse_dcsrgeam
  :outline:
.. doxygenfunction:: rocsparse_ccsrgeam
  :outline:
.. doxygenfunction:: rocsparse_zcsrgeam

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

rocsparse_csrgemm_symbolic()
----------------------------

.. doxygenfunction:: rocsparse_csrgemm_symbolic

rocsparse_csrgemm()
-------------------

.. doxygenfunction:: rocsparse_scsrgemm
  :outline:
.. doxygenfunction:: rocsparse_dcsrgemm
  :outline:
.. doxygenfunction:: rocsparse_ccsrgemm
  :outline:
.. doxygenfunction:: rocsparse_zcsrgemm

rocsparse_csrgemm_numeric()
---------------------------

.. doxygenfunction:: rocsparse_scsrgemm_numeric
  :outline:
.. doxygenfunction:: rocsparse_dcsrgemm_numeric
  :outline:
.. doxygenfunction:: rocsparse_ccsrgemm_numeric
  :outline:
.. doxygenfunction:: rocsparse_zcsrgemm_numeric

.. _rocsparse_precond_functions_:

Preconditioner Functions
========================

This module holds all sparse preconditioners.

The sparse preconditioners describe manipulations on a matrix in sparse format to obtain a sparse preconditioner matrix.

rocsparse_bsric0_zero_pivot()
-----------------------------

.. doxygenfunction:: rocsparse_bsric0_zero_pivot

rocsparse_bsric0_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_sbsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dbsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cbsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zbsric0_buffer_size

rocsparse_bsric0_analysis()
---------------------------

.. doxygenfunction:: rocsparse_sbsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsric0_analysis

rocsparse_bsric0()
------------------

.. doxygenfunction:: rocsparse_sbsric0
  :outline:
.. doxygenfunction:: rocsparse_dbsric0
  :outline:
.. doxygenfunction:: rocsparse_cbsric0
  :outline:
.. doxygenfunction:: rocsparse_zbsric0

rocsparse_bsric0_clear()
------------------------

.. doxygenfunction:: rocsparse_bsric0_clear

rocsparse_bsrilu0_zero_pivot()
------------------------------

.. doxygenfunction:: rocsparse_bsrilu0_zero_pivot

rocsparse_bsrilu0_numeric_boost()
---------------------------------

.. doxygenfunction:: rocsparse_sbsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_dbsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_cbsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_zbsrilu0_numeric_boost

rocsparse_bsrilu0_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_sbsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dbsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cbsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zbsrilu0_buffer_size

rocsparse_bsrilu0_analysis()
----------------------------

.. doxygenfunction:: rocsparse_sbsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsrilu0_analysis

rocsparse_bsrilu0()
-------------------

.. doxygenfunction:: rocsparse_sbsrilu0
  :outline:
.. doxygenfunction:: rocsparse_dbsrilu0
  :outline:
.. doxygenfunction:: rocsparse_cbsrilu0
  :outline:
.. doxygenfunction:: rocsparse_zbsrilu0

rocsparse_bsrilu0_clear()
-------------------------

.. doxygenfunction:: rocsparse_bsrilu0_clear

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

rocsparse_csritilu0_buffer_size()
---------------------------------

.. doxygenfunction:: rocsparse_csritilu0_buffer_size

rocsparse_csritilu0_preprocess()
--------------------------------

.. doxygenfunction:: rocsparse_csritilu0_preprocess

rocsparse_csritilu0_history()
-----------------------------

.. doxygenfunction:: rocsparse_scsritilu0_history
  :outline:
.. doxygenfunction:: rocsparse_dcsritilu0_history
  :outline:
.. doxygenfunction:: rocsparse_ccsritilu0_history
  :outline:
.. doxygenfunction:: rocsparse_zcsritilu0_history


rocsparse_csritilu0_compute()
-----------------------------

.. doxygenfunction:: rocsparse_scsritilu0_compute
  :outline:
.. doxygenfunction:: rocsparse_dcsritilu0_compute
  :outline:
.. doxygenfunction:: rocsparse_ccsritilu0_compute
  :outline:
.. doxygenfunction:: rocsparse_zcsritilu0_compute


rocsparse_csrilu0_zero_pivot()
------------------------------

.. doxygenfunction:: rocsparse_csrilu0_zero_pivot

rocsparse_csrilu0_numeric_boost()
---------------------------------

.. doxygenfunction:: rocsparse_scsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_ccsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_zcsrilu0_numeric_boost

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

rocsparse_gtsv_buffer_size()
----------------------------

.. doxygenfunction:: rocsparse_sgtsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_buffer_size

rocsparse_gtsv()
----------------

.. doxygenfunction:: rocsparse_sgtsv
  :outline:
.. doxygenfunction:: rocsparse_dgtsv
  :outline:
.. doxygenfunction:: rocsparse_cgtsv
  :outline:
.. doxygenfunction:: rocsparse_zgtsv

rocsparse_gtsv_no_pivot_buffer_size()
-------------------------------------

.. doxygenfunction:: rocsparse_sgtsv_no_pivot_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_no_pivot_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_no_pivot_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_no_pivot_buffer_size

rocsparse_gtsv_no_pivot()
-------------------------

.. doxygenfunction:: rocsparse_sgtsv_no_pivot
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_no_pivot
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_no_pivot
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_no_pivot

rocsparse_gtsv_no_pivot_strided_batch_buffer_size()
---------------------------------------------------

.. doxygenfunction:: rocsparse_sgtsv_no_pivot_strided_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_no_pivot_strided_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_no_pivot_strided_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_no_pivot_strided_batch_buffer_size

rocsparse_gtsv_no_pivot_strided_batch()
---------------------------------------

.. doxygenfunction:: rocsparse_sgtsv_no_pivot_strided_batch
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_no_pivot_strided_batch
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_no_pivot_strided_batch
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_no_pivot_strided_batch

rocsparse_gtsv_interleaved_batch_buffer_size()
----------------------------------------------

.. doxygenfunction:: rocsparse_sgtsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_interleaved_batch_buffer_size

rocsparse_gtsv_interleaved_batch()
----------------------------------

.. doxygenfunction:: rocsparse_sgtsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_interleaved_batch

rocsparse_gpsv_interleaved_batch_buffer_size()
----------------------------------------------

.. doxygenfunction:: rocsparse_sgpsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgpsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgpsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgpsv_interleaved_batch_buffer_size

rocsparse_gpsv_interleaved_batch()
----------------------------------

.. doxygenfunction:: rocsparse_sgpsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_dgpsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_cgpsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_zgpsv_interleaved_batch

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

rocsparse_gebsr2gebsc_buffer_size()
-----------------------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsc_buffer_size

rocsparse_gebsr2gebsc()
-----------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsc
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsc
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsc
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsc

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

rocsparse_bsr2csr()
-------------------

.. doxygenfunction:: rocsparse_sbsr2csr
  :outline:
.. doxygenfunction:: rocsparse_dbsr2csr
  :outline:
.. doxygenfunction:: rocsparse_cbsr2csr
  :outline:
.. doxygenfunction:: rocsparse_zbsr2csr

rocsparse_gebsr2csr()
---------------------

.. doxygenfunction:: rocsparse_sgebsr2csr
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2csr
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2csr
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2csr

rocsparse_gebsr2gebsr_buffer_size()
-----------------------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsr_buffer_size

rocsparse_gebsr2gebsr_nnz()
---------------------------

.. doxygenfunction:: rocsparse_gebsr2gebsr_nnz

rocsparse_gebsr2gebsr()
-----------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsr


rocsparse_csr2bsr_nnz()
-----------------------

.. doxygenfunction:: rocsparse_csr2bsr_nnz

rocsparse_csr2bsr()
-------------------

.. doxygenfunction:: rocsparse_scsr2bsr
  :outline:
.. doxygenfunction:: rocsparse_dcsr2bsr
  :outline:
.. doxygenfunction:: rocsparse_ccsr2bsr
  :outline:
.. doxygenfunction:: rocsparse_zcsr2bsr

rocsparse_csr2gebsr_nnz()
-------------------------

.. doxygenfunction:: rocsparse_csr2gebsr_nnz

rocsparse_csr2gebsr_buffer_size()
---------------------------------

.. doxygenfunction:: rocsparse_scsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsr2gebsr_buffer_size

rocsparse_csr2gebsr()
---------------------

.. doxygenfunction:: rocsparse_scsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_dcsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_ccsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_zcsr2gebsr

rocsparse_csr2csr_compress()
----------------------------

.. doxygenfunction:: rocsparse_scsr2csr_compress
  :outline:
.. doxygenfunction:: rocsparse_dcsr2csr_compress
  :outline:
.. doxygenfunction:: rocsparse_ccsr2csr_compress
  :outline:
.. doxygenfunction:: rocsparse_zcsr2csr_compress

rocsparse_inverse_permutation()
---------------------------------------

.. doxygenfunction:: rocsparse_inverse_permutation

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

rocsparse_nnz_compress()
------------------------

.. doxygenfunction:: rocsparse_snnz_compress
  :outline:
.. doxygenfunction:: rocsparse_dnnz_compress
  :outline:
.. doxygenfunction:: rocsparse_cnnz_compress
  :outline:
.. doxygenfunction:: rocsparse_znnz_compress

rocsparse_nnz()
---------------

.. doxygenfunction:: rocsparse_snnz
  :outline:
.. doxygenfunction:: rocsparse_dnnz
  :outline:
.. doxygenfunction:: rocsparse_cnnz
  :outline:
.. doxygenfunction:: rocsparse_znnz


rocsparse_dense2csr()
---------------------

.. doxygenfunction:: rocsparse_sdense2csr
  :outline:
.. doxygenfunction:: rocsparse_ddense2csr
  :outline:
.. doxygenfunction:: rocsparse_cdense2csr
  :outline:
.. doxygenfunction:: rocsparse_zdense2csr


rocsparse_dense2csc()
---------------------

.. doxygenfunction:: rocsparse_sdense2csc
  :outline:
.. doxygenfunction:: rocsparse_ddense2csc
  :outline:
.. doxygenfunction:: rocsparse_cdense2csc
  :outline:
.. doxygenfunction:: rocsparse_zdense2csc


rocsparse_dense2coo()
---------------------

.. doxygenfunction:: rocsparse_sdense2coo
  :outline:
.. doxygenfunction:: rocsparse_ddense2coo
  :outline:
.. doxygenfunction:: rocsparse_cdense2coo
  :outline:
.. doxygenfunction:: rocsparse_zdense2coo


rocsparse_csr2dense()
---------------------

.. doxygenfunction:: rocsparse_scsr2dense
  :outline:
.. doxygenfunction:: rocsparse_dcsr2dense
  :outline:
.. doxygenfunction:: rocsparse_ccsr2dense
  :outline:
.. doxygenfunction:: rocsparse_zcsr2dense


rocsparse_csc2dense()
---------------------

.. doxygenfunction:: rocsparse_scsc2dense
  :outline:
.. doxygenfunction:: rocsparse_dcsc2dense
  :outline:
.. doxygenfunction:: rocsparse_ccsc2dense
  :outline:
.. doxygenfunction:: rocsparse_zcsc2dense

rocsparse_coo2dense()
---------------------

.. doxygenfunction:: rocsparse_scoo2dense
  :outline:
.. doxygenfunction:: rocsparse_dcoo2dense
  :outline:
.. doxygenfunction:: rocsparse_ccoo2dense
  :outline:
.. doxygenfunction:: rocsparse_zcoo2dense

rocsparse_prune_dense2csr_buffer_size()
---------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_buffer_size

rocsparse_prune_dense2csr_nnz()
-------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_nnz
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_nnz

rocsparse_prune_dense2csr()
---------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr

rocsparse_prune_csr2csr_buffer_size()
-------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_buffer_size

rocsparse_prune_csr2csr_nnz()
-----------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_nnz
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_nnz

rocsparse_prune_csr2csr()
-------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr

rocsparse_prune_dense2csr_by_percentage_buffer_size()
-----------------------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_by_percentage_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_by_percentage_buffer_size

rocsparse_prune_dense2csr_nnz_by_percentage()
---------------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_nnz_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_nnz_by_percentage

rocsparse_prune_dense2csr_by_percentage()
-----------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_by_percentage

rocsparse_prune_csr2csr_by_percentage_buffer_size()
---------------------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_by_percentage_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_by_percentage_buffer_size

rocsparse_prune_csr2csr_nnz_by_percentage()
-------------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_nnz_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_nnz_by_percentage

rocsparse_prune_csr2csr_by_percentage()
---------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_by_percentage

rocsparse_rocsparse_bsrpad_value()
----------------------------------

.. doxygenfunction:: rocsparse_sbsrpad_value
  :outline:
.. doxygenfunction:: rocsparse_dbsrpad_value
  :outline:
.. doxygenfunction:: rocsparse_cbsrpad_value
  :outline:
.. doxygenfunction:: rocsparse_zbsrpad_value

.. _rocsparse_reordering_functions_:

Reordering Functions
========================

This module holds all sparse reordering routines.

The sparse reordering routines describe algorithm for reordering sparse matrices.

rocsparse_csrcolor()
--------------------

.. doxygenfunction:: rocsparse_scsrcolor
  :outline:
.. doxygenfunction:: rocsparse_dcsrcolor
  :outline:
.. doxygenfunction:: rocsparse_ccsrcolor
  :outline:
.. doxygenfunction:: rocsparse_zcsrcolor

Utility Functions
=================

This module holds all sparse utility routines.

The sparse utility routines allow for testing whether matrix data is valid for different matrix formats

rocsparse_check_matrix_csr_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_csr_buffer_size

rocsparse_check_matrix_csr()
----------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_csr
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_csr
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_csr
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_csr

rocsparse_check_matrix_csc_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_csc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_csc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_csc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_csc_buffer_size

rocsparse_check_matrix_csc()
----------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_csc
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_csc
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_csc
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_csc

rocsparse_check_matrix_coo_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_coo_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_coo_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_coo_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_coo_buffer_size

rocsparse_check_matrix_coo()
----------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_coo
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_coo
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_coo
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_coo

rocsparse_check_matrix_gebsr_buffer_size()
------------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_gebsr_buffer_size

rocsparse_check_matrix_gebsr()
------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_gebsr
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_gebsr
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_gebsr
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_gebsr

rocsparse_check_matrix_gebsc_buffer_size()
------------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_gebsc_buffer_size

rocsparse_check_matrix_gebsc()
------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_gebsc
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_gebsc
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_gebsc
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_gebsc

rocsparse_check_matrix_ell_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_ell_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_ell_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_ell_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_ell_buffer_size

rocsparse_check_matrix_ell()
----------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_ell
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_ell
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_ell
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_ell

rocsparse_check_matrix_hyb_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_check_matrix_hyb_buffer_size

rocsparse_check_matrix_hyb()
----------------------------

.. doxygenfunction:: rocsparse_check_matrix_hyb

Sparse Generic Functions
========================

This module holds all sparse generic routines.

The sparse generic routines describe operations that manipulate sparse matrices.

rocsparse_axpby()
-----------------

.. doxygenfunction:: rocsparse_axpby

rocsparse_gather()
------------------

.. doxygenfunction:: rocsparse_gather

rocsparse_scatter()
-------------------

.. doxygenfunction:: rocsparse_scatter

rocsparse_rot()
---------------

.. doxygenfunction:: rocsparse_rot

rocsparse_spvv()
----------------

.. doxygenfunction:: rocsparse_spvv

rocsparse_spmv()
----------------

.. doxygenfunction:: rocsparse_spmv

rocsparse_spmv_ex()
-------------------

.. doxygenfunction:: rocsparse_spmv_ex

rocsparse_spsv()
----------------

.. doxygenfunction:: rocsparse_spsv

rocsparse_spsm()
----------------

.. doxygenfunction:: rocsparse_spsm

rocsparse_spmm()
----------------

.. doxygenfunction:: rocsparse_spmm

rocsparse_spmm_ex()
-------------------

.. doxygenfunction:: rocsparse_spmm_ex

rocsparse_spgemm()
------------------

.. doxygenfunction:: rocsparse_spgemm

rocsparse_sddmm_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sddmm_buffer_size

rocsparse_sddmm_preprocess()
----------------------------

.. doxygenfunction:: rocsparse_sddmm_preprocess

rocsparse_sddmm()
-----------------

.. doxygenfunction:: rocsparse_sddmm

rocsparse_dense_to_sparse()
---------------------------

.. doxygenfunction:: rocsparse_dense_to_sparse

rocsparse_sparse_to_dense()
---------------------------

.. doxygenfunction:: rocsparse_sparse_to_dense
