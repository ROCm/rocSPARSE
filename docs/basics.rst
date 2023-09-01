******
Basics
******

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
ell_val     array of ``m * ell_width`` elements containing the data (floating point).
ell_col_ind array of ``m * ell_width`` elements containing the column indices (integer).
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
ell_val     array of ``m * ell_width`` elements containing the ELL part data (floating point).
ell_col_ind array of ``m * ell_width`` elements containing the ELL part column indices (integer).
coo_val     array of ``nnz`` elements containing the COO part data (floating point).
coo_row_ind array of ``nnz`` elements containing the COO part row indices (integer).
coo_col_ind array of ``nnz`` elements containing the COO part column indices (integer).
=========== =========================================================================================

The HYB format is a combination of the ELL and COO sparse matrix formats. Typically, the regular part of the matrix is stored in
ELL storage format, and the irregular part of the matrix is stored in COO storage format. Three different partitioning schemes can
be applied when converting a CSR matrix to a matrix in HYB storage format. For further details on the partitioning schemes,
see :ref:`rocsparse_hyb_partition_`.

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
|:cpp:func:`rocsparse_bsr_set_pointers`       |
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

============================================================================= ====== ====== ============== ==============
Function name                                                                 single double single complex double complex
============================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xbsrmv_ex_analysis() <rocsparse_sbsrmv_ex_analysis>`     x      x      x              x
:cpp:func:`rocsparse_bsrmv_ex_clear`
:cpp:func:`rocsparse_Xbsrmv_ex() <rocsparse_sbsrmv_ex>`                       x      x      x              x
:cpp:func:`rocsparse_Xbsrmv() <rocsparse_sbsrmv>`                             x      x      x              x
:cpp:func:`rocsparse_Xbsrxmv() <rocsparse_sbsrxmv>`                           x      x      x              x
:cpp:func:`rocsparse_Xbsrsv_buffer_size() <rocsparse_sbsrsv_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xbsrsv_analysis() <rocsparse_sbsrsv_analysis>`           x      x      x              x
:cpp:func:`rocsparse_bsrsv_zero_pivot`
:cpp:func:`rocsparse_bsrsv_clear`
:cpp:func:`rocsparse_Xbsrsv_solve() <rocsparse_sbsrsv_solve>`                 x      x      x              x
:cpp:func:`rocsparse_Xcoomv() <rocsparse_scoomv>`                             x      x      x              x
:cpp:func:`rocsparse_Xcsrmv_analysis() <rocsparse_scsrmv_analysis>`           x      x      x              x
:cpp:func:`rocsparse_csrmv_clear`
:cpp:func:`rocsparse_Xcsrmv() <rocsparse_scsrmv>`                             x      x      x              x
:cpp:func:`rocsparse_Xcsrsv_buffer_size() <rocsparse_scsrsv_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcsrsv_analysis() <rocsparse_scsrsv_analysis>`           x      x      x              x
:cpp:func:`rocsparse_csrsv_zero_pivot`
:cpp:func:`rocsparse_csrsv_clear`
:cpp:func:`rocsparse_Xcsrsv_solve() <rocsparse_scsrsv_solve>`                 x      x      x              x
:cpp:func:`rocsparse_Xcsritsv_buffer_size() <rocsparse_scsritsv_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcsritsv_analysis() <rocsparse_scsritsv_analysis>`       x      x      x              x
:cpp:func:`rocsparse_csritsv_zero_pivot`
:cpp:func:`rocsparse_csritsv_clear`
:cpp:func:`rocsparse_Xcsritsv_solve() <rocsparse_scsritsv_solve>`             x      x      x              x
:cpp:func:`rocsparse_Xellmv() <rocsparse_sellmv>`                             x      x      x              x
:cpp:func:`rocsparse_Xhybmv() <rocsparse_shybmv>`                             x      x      x              x
:cpp:func:`rocsparse_Xgebsrmv() <rocsparse_sgebsrmv>`                         x      x      x              x
:cpp:func:`rocsparse_Xgemvi_buffer_size() <rocsparse_sgemvi_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xgemvi() <rocsparse_sgemvi>`                             x      x      x              x
============================================================================= ====== ====== ============== ==============

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
:cpp:func:`rocsparse_Xbsrgemm_buffer_size() <rocsparse_sbsrgemm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_bsrgemm_nnzb`
:cpp:func:`rocsparse_Xbsrgemm() <rocsparse_sbsrgemm>`                         x      x      x              x
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
:cpp:func:`rocsparse_Xgebsr2gebsc_buffer_size() <rocsparse_sgebsr2gebsc_buffer_size>`                                     x      x      x              x
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
:cpp:func:`rocsparse_inverse_permutation`
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
