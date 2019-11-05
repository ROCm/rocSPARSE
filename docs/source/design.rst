********************
Design Documentation
********************

.. toctree::
   :maxdepth: 3
   :caption: Contents:

Library Source Organization
===========================

The `library/include` directory
-------------------------------
This directory contains all files that are exposed to the user.
The rocSPARSE API, is declared here.

=========================== ====
File                        Description
=========================== ====
`rocsparse.h`               Includes all other API related rocSPARSE header files.
`rocsparse-auxiliary.h`     Declares all rocSPARSE auxiliary functions, such as handle and descriptor management.
`rocsparse-complex-types.h` Defines the rocSPARSE complex data types `rocsparse_float_complex` and `rocsparse_double_complex`.
`rocsparse-functions.h`     Declares all rocSPARSE Sparse Linear Algebra Subroutines of Level1, 2, 3, Extra, Preconditioner and Format Conversion.
`rocsparse-types.h`         Defines all data types used by rocSPARSE.
`rocsparse-version.h.in`    Provides the configured version and settings that is initially set by CMake during compilation.
=========================== ====

The `library/src/` directory
----------------------------
This directory contains all rocSPARSE library source files.
The root of the `library/src/` directory hosts the implementation of the library handle and auxiliary functions.
Furthermore, each subdirectory is responsible for the specific class of sparse linear algebra subroutines.
Finally, the `library/src/include` directory defines :ref:`rocsparse_common`, :ref:`rocsparse_macros`, :ref:`rocsparse_mat_struct` and :ref:`rocsparse_logging`.

The `clients/` directory
------------------------
This directory contains all clients, e.g. samples, unit tests and benchmarks.
Further details are given in :ref:`rocsparse_clients`.

Sparse Linear Algebra Subroutines
---------------------------------
Each sparse linear algebra subroutine is implemented in a set of source files: ``rocsparse_subroutine.cpp``, ``rocsparse_subroutine.hpp`` and ``subroutine_device.h``.

``rocsparse_subroutine.cpp`` implements the C wrapper and the not precision dependent API functionality.
Furthermore, ``rocsparse_subroutine.hpp`` implements the precision dependent API functionality, using the precision as template parameter.
Finally, ``subroutine_device.h`` implements the device code, required for the computation of the subroutine.

.. note:: Each API exposed subroutine is expected to return a :cpp:type:`rocsparse_status`.
.. note:: Additionally, each device function is expected to use the user given stream which is accessible through the libraries handle.

Below is a sample for ``rocsparse_subroutine.cpp``, ``rocsparse_subroutine.hpp`` and ``subroutine_device.h``.

.. code-block:: cpp
   :caption: rocsparse_subroutine.cpp

   #include "rocsparse.h"
   #include "rocsparse_subroutine.hpp"

   /*
    * ===========================
    *    C wrapper
    * ===========================
    */

   extern "C" rocsparse_status rocsparse_ssubroutine(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     const float*     alpha,
                                                     float*           val)
   {
       return rocsparse_subroutine_template(handle, m, alpha, val);
   }

   extern "C" rocsparse_status rocsparse_dsubroutine(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     const double*    alpha,
                                                     double*          val)
   {
       return rocsparse_subroutine_template(handle, m, alpha, val);
   }

   extern "C" rocsparse_status rocsparse_csubroutine(rocsparse_handle               handle,
                                                     rocsparse_int                  m,
                                                     const rocsparse_float_complex* alpha,
                                                     rocsparse_float_complex*       val)
   {
       return rocsparse_subroutine_template(handle, m, alpha, val);
   }

   extern "C" rocsparse_status rocsparse_zsubroutine(rocsparse_handle                handle,
                                                     rocsparse_int                   m,
                                                     const rocsparse_double_complex* alpha,
                                                     rocsparse_double_complex*       val)
   {
       return rocsparse_subroutine_template(handle, m, alpha, val);
   }

.. code-block:: cpp
   :caption: rocsparse_subroutine.hpp

   #pragma once
   #ifndef ROCSPARSE_SUBROUTINE_HPP
   #define ROCSPARSE_SUBROUTINE_HPP

   #include "definitions.h"
   #include "handle.h"
   #include "rocsparse.h"
   #include "subroutine_device.h"
   #include "utility.h"

   #include <hip/hip_runtime.h>

   template <typename T>
   __global__ void subroutine_kernel_host_pointer(rocsparse_int m, T alpha, T* val)
   {
       subroutine_device(m, alpha, val);
   }

   template <typename T>
   __global__ void subroutine_kernel_device_pointer(rocsparse_int m, const T* alpha, T* val)
   {
       subroutine_device(m, *alpha, val);
   }

   template <typename T>
   rocsparse_status rocsparse_subroutine_template(rocsparse_handle handle,
                                                  rocsparse_int    m,
                                                  const T*         alpha,
                                                  T*               val)
   {
       // Check for valid handle
       if(handle == nullptr)
       {
           return rocsparse_status_invalid_handle;
       }

       // Logging
       if(handle->pointer_mode == rocsparse_pointer_mode_host)
       {
           log_trace(handle,
                     replaceX<T>("rocsparse_Xsubroutine"),
                     m,
                     *alpha,
                     (const void*&)val);

           log_bench(handle,
                     "./rocsparse-bench -f subroutine -r",
                     replaceX<T>("X"),
                     "-m",
                     m,
                     "--alpha",
                     *alpha);
       }
       else
       {
           log_trace(handle,
                     replaceX<T>("rocsparse_Xsubroutine"),
                     m,
                     (const void*&)alpha,
                     (const void*&)val);
       }

       // Check size
       if(m < 0)
       {
           return rocsparse_status_invalid_size;
       }

       // Quick return if possible
       if(m == 0)
       {
           return rocsparse_status_success;
       }

       // Check pointer arguments
       if(alpha == nullptr || val == nullptr)
       {
           return rocsparse_status_invalid_pointer;
       }

       // Differentiate between the pointer modes
       if(handle->pointer_mode == rocsparse_pointer_mode_device)
       {
           // Launch kernel
           hipLaunchKernelGGL((subroutine_kernel_device_pointer<T>),
                              dim3(...),
                              dim3(...),
                              0,
                              handle->stream,
                              m,
                              alpha,
                              val);
       }
       else
       {
           // Launch kernel
           hipLaunchKernelGGL((subroutine_kernel_host_pointer<T>),
                              dim3(...),
                              dim3(...),
                              0,
                              handle->stream,
                              m,
                              *alpha,
                              val);
       }

       return rocsparse_status_success;
   }

   #endif // ROCSPARSE_SUBROUTINE_HPP

.. code-block:: cpp
   :caption: subroutine_device.h

   #pragma once
   #ifndef SUBROUTINE_DEVICE_H
   #define SUBROUTINE_DEVICE_H

   #include <hip/hip_runtime.h>

   template <typename T>
   __device__ void subroutine_device(rocsparse_int m, T alpha, T* val)
   {
       ...
   }

   #endif // SUBROUTINE_DEVICE_H

Important Functions and Data Structures
=======================================
This section describes important rocSPARSE functions and data structures.

.. _rocsparse_common:

Commonly Shared Device-Code
---------------------------
The following table lists multiple device functions that are shared among several rocSPARSE functions.

================================= ====
Device function                   Description
================================= ====
``rocsparse_clz()``               Computes the leftmost significant bit position for int and int64 types.
``rocsparse_one()``               Returns a pointer to ``1`` for the specified precision.
``rocsparse_ldg()``               Wrapper to ``__ldg()`` for int, int64, single, double real and complex types.
``rocsparse_nontemporal_load()``  Non-temporal memory load access for int, int64, single, double real and complex types.
``rocsparse_nontemporal_store()`` Non-temporal memory store access for int, int64, single, double real and complex types.
``rocsparse_mul24()``             Multiply 24-bit integer values.
``rocsparse_mad24()``             Multiply 24-bit integers and add a 32-bit value.
``rocsparse_atomic_load()``       Memory model aware atomic load operation for int type.
``rocsparse_atomic_store()``      Memory model aware atomic store operation for int type.
``rocsparse_blockreduce_sum()``   Blockwide reduction sum for int, int64, single, double real and complex types.
``rocsparse_blockreduce_max()``   Blockwide reduction max for int, int64, single, double real and complex types.
``rocsparse_blockreduce_min()``   Blockwide reduction min for int, int64, single, double real and complex types.
``rocsparse_wfreduce_max()``      DPP based wavefront reduction max for int type.
``rocsparse_wfreduce_min()``      DPP based wavefront reduction min for int and int64 types.
``rocsparse_wfreduce_sum()``      DPP based wavefront reduction sum for int, int64, single, double real and complex types.
================================= ====

.. _rocsparse_macros:

Status-Flag Macros
------------------
The following table lists the status-flag macros available in rocSPARSE and their purpose.

=================================== ====
Macro                               Description
=================================== ====
``RETURN_IF_HIP_ERROR(stat)``       Returns, if `stat` is not equal to ``hipSuccess``
``THROW_IF_HIP_ERROR(stat)``        Throws an exception, if `stat` is not equal to ``hipSuccess``
``PRINT_IF_HIP_ERROR(stat)``        Prints an error message, if `stat` is not equal to ``hipSuccess``
``RETURN_IF_ROCSPARSE_ERROR(stat)`` Returns, if `stat` is not equal to :cpp:enumerator:`rocsparse_status_success`
=================================== ====

.. _rocsparse_mat_struct:

The `rocsparse_mat_info` Structure
----------------------------------
The rocSPARSE :cpp:type:`rocsparse_mat_info` is a structure holding all matrix meta information that is gathered during analysis routines.

The following table lists all currently available internal meta data structures:

========================== ====
Meta data structure        Description
========================== ====
``rocsparse_csrmv_info``   Structure to hold analysis meta data for sparse matrix vector multiplication in CSR format.
``rocsparse_csrtr_info``   Structure to hold analysis meta data for operations on sparse triangular matrices, e.g. dependency graph.
``rocsparse_csrgemm_info`` Structure to hold analysis meta data for sparse matrix sparse matrix multiplication in CSR format.
========================== ====

Cross-Routine Data Sharing
``````````````````````````
Already collected meta data, such as the dependency graph of a sparse matrix, can be shared among multiple routines.
For example, if the incomplete LU factorization of a sparse matrix is computed, the gathered analysis data can be shared for subsequent lower triangular solves of the same matrix.
This behavior can be specified by the :ref:`rocsparse_analysis_policy_` parameter.

The following table lists subroutines that can in some cases share meta data:

================================== ====
Subroutine                         Sharing meta data with
================================== ====
:cpp:func:`rocsparse_scsrsv_solve` :cpp:func:`rocsparse_scsrilu0`
:cpp:func:`rocsparse_dcsrsv_solve` :cpp:func:`rocsparse_dcsrilu0`
:cpp:func:`rocsparse_ccsrsv_solve` :cpp:func:`rocsparse_ccsrilu0`
:cpp:func:`rocsparse_zcsrsv_solve` :cpp:func:`rocsparse_zcsrilu0`
:cpp:func:`rocsparse_scsrilu0`     :cpp:func:`rocsparse_scsrsv_solve`
:cpp:func:`rocsparse_dcsrilu0`     :cpp:func:`rocsparse_dcsrsv_solve`
:cpp:func:`rocsparse_ccsrilu0`     :cpp:func:`rocsparse_ccsrsv_solve`
:cpp:func:`rocsparse_zcsrilu0`     :cpp:func:`rocsparse_zcsrsv_solve`
================================== ====

.. note:: It is important to note, that on rocSPARSE extensions, this functionality can be further expanded to improve meta data collection performance significantly.

.. _rocsparse_clients:

Clients
=======
rocSPARSE clients host a variety of different examples as well as a unit test and benchmarking package.
For detailed instructions on how to build rocSPARSE with clients, see :ref:`rocsparse_building`.

Examples
--------
The examples collection offers sample implementations of the rocSPARSE API.
In the following table, available examples with description, are listed.

============== ====
Example        Description
============== ====
example_coomv  Perform sparse matrix vector multiplication in COO format
example_csrmv  Perform sparse matrix vector multiplication in CSR format
example_ellmv  Perform sparse matrix vector multiplication in ELL format
example_handle Show rocSPARSE handle initialization and finalization
example_hybmv  Perform sparse matrix vector multiplication in HYB format
============== ====

Unit Tests
----------
Multiple unit tests are available to test for bad arguments, invalid parameters and sparse routine functionality.
The unit tests are based on `googletest <https://github.com/google/googletest>`_.
The tests cover all routines that are exposed by the API, including all available floating-point precisions.

Benchmarks
----------
rocSPARSE offers a benchmarking tool that can be compiled with the clients package.
The benchmark tool can perform any API exposed routine combined with time measurement.
For `rocsparse-bench` to be compiled, the `libboost-program-options <https://www.boost.org/>`_ package is required.
To set up a benchmark run, multiple options are available.

=================== ====
Command-line option Description
=================== ====
help, h             Prints the help message
sizem, m            Specify the m parameter, e.g. the number of rows of a sparse matrix
sizen, n            Specify the n parameter, e.g. the number of columns of a sparse matrix or the length of a dense vector
sizek, k            Specify the k parameter, e.g. the number of rows of a dense matrix
sizennz, z          Specify the nnz parameter, e.g. the number of non-zero entries of a sparse vector
mtx                 Read from `MatrixMarket (.mtx) format <https://math.nist.gov/MatrixMarket/formats.html>`_. This will override parameters `m`, `n` and `z`
rocalution          Read from `rocALUTION format <https://github.com/ROCmSoftwarePlatform/rocALUTION>`_. This will override parameters `m`, `n`, `z`, `mtx` and `laplacian-dim`
laplacian-dim       Assemble a 2D/3D Laplacian matrix with dimensions `dimx`, `dimy` and `dimz`. `dimz` is optional. This will override parameters `m`, `n`, `z` and `mtx`
alpha               Specify the scalar :math:`\alpha`
beta                Specify the scalar :math:`\beta`
transposeA          Specify whether matrix A is (conjugate) transposed or not, see :ref:`rocsparse_operation_`
transposeB          Specify whether matrix B is (conjugate) transposed or not, see :ref:`rocsparse_operation_`
indexbaseA          Specify the index base of matrix A, see :ref:`rocsparse_index_base_`
indexbaseB          Specify the index base of matrix B, see :ref:`rocsparse_index_base_`
indexbaseC          Specify the index base of matrix C, see :ref:`rocsparse_index_base_`
indexbaseD          Specify the index base of matrix D, see :ref:`rocsparse_index_base_`
action              Specify whether the operation is performed symbolically or numerically, see :ref:`rocsparse_action_`
hybpart             Specify the HYB partitioning type, see :ref:`rocsparse_hyb_partition_`
diag                Specify the diagonal type of a sparse matrix, see :ref:`rocsparse_diag_type_`
uplo                Specify the fill mode of a sparse matrix, see :ref:`rocsparse_fill_mode_`
apolicy             Specify the analysis policy, see :ref:`rocsparse_analysis_policy_`
function, f         Specify the API exposed subroutine to benchmark
precision, r        Floating-point precision: single real, double real, single complex, double complex
verify, v           Specify whether the results should be validated with the host reference implementation
iters, i            Iterations to run inside the timing loop
device, d           Set the device to be used for subsequent benchmark runs
=================== ====
