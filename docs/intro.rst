Introduction
============

rocSPARSE is a library that contains basic linear algebra subroutines for sparse matrices and vectors written in HIP for GPU devices. It is designed to be used from C and C++ code. The functionality of rocSPARSE is organized in the following categories:

* :ref:`rocsparse_auxiliary_functions_` describe available helper functions that are required for subsequent library calls.
* :ref:`rocsparse_level1_functions_` describe operations between a vector in sparse format and a vector in dense format.
* :ref:`rocsparse_level2_functions_` describe operations between a matrix in sparse format and a vector in dense format.
* :ref:`rocsparse_level3_functions_` describe operations between a matrix in sparse format and multiple vectors in dense format.
* :ref:`rocsparse_extra_functions_` describe operations that manipulate sparse matrices.
* :ref:`rocsparse_precond_functions_` describe manipulations on a matrix in sparse format to obtain a preconditioner.
* :ref:`rocsparse_conversion_functions_` describe operations on a matrix in sparse format to obtain a different matrix format.
* :ref:`rocsparse_reordering_functions_` describe operations on a matrix in sparse format to obtain a reordering.
* :ref:`rocsparse_utility_functions_` describe routines useful for checking sparse matrices for valid data

The code is open and hosted here: https://github.com/ROCm/rocSPARSE
