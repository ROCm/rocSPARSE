.. toctree::
   :maxdepth: 4 
   :caption: Contents:

===
API
===

This section provides details of the library API.

Types
-----

There are few data structures that are internal to the library. The pointer types to these
structures are given below. The user would need to use these types to create handles and pass them
between different library functions.

.. doxygentypedef:: rocsparse_handle

.. doxygentypedef:: rocsparse_mat_descr

.. doxygentypedef:: rocsparse_mat_info

.. doxygentypedef:: rocsparse_hyb_mat

Auxiliary Functions
-------------------

The following functions deals with initialization and cleanup of the different types.

.. doxygenfunction:: rocsparse_create_handle

.. doxygenfunction:: rocsparse_destroy_handle

.. doxygenfunction:: rocsparse_set_stream

.. doxygenfunction:: rocsparse_get_stream

.. doxygenfunction:: rocsparse_set_pointer_mode

.. doxygenfunction:: rocsparse_get_pointer_mode

.. doxygenfunction:: rocsparse_create_mat_descr

.. doxygenfunction:: rocsparse_destroy_mat_descr

.. doxygenfunction:: rocsparse_set_mat_index_base

.. doxygenfunction:: rocsparse_get_mat_index_base

.. doxygenfunction:: rocsparse_set_mat_type

.. doxygenfunction:: rocsparse_get_mat_type

.. doxygenfunction:: rocsparse_create_mat_info

.. doxygenfunction:: rocsparse_destroy_mat_info

.. doxygenfunction:: rocsparse_create_hyb_mat

.. doxygenfunction:: rocsparse_destroy_hyb_mat

Sparse Level 1 Functions
------------------------

This section describes all rocSPARSE level 1 sparse linear algebra functions.

.. doxygenfunction:: rocsparse_saxpyi

.. doxygenfunction:: rocsparse_daxpyi

.. doxygenfunction:: rocsparse_sdoti

.. doxygenfunction:: rocsparse_ddoti

.. doxygenfunction:: rocsparse_sgthr

.. doxygenfunction:: rocsparse_dgthr

.. doxygenfunction:: rocsparse_sgthrz

.. doxygenfunction:: rocsparse_dgthrz

.. doxygenfunction:: rocsparse_sroti

.. doxygenfunction:: rocsparse_droti

.. doxygenfunction:: rocsparse_ssctr

.. doxygenfunction:: rocsparse_dsctr

Sparse Level 2 Functions
------------------------

.. doxygenfunction:: rocsparse_scoomv

.. doxygenfunction:: rocsparse_dcoomv

.. doxygenfunction:: rocsparse_csrmv_analysis

.. doxygenfunction:: rocsparse_csrmv_analysis_clear

.. doxygenfunction:: rocsparse_scsrmv

.. doxygenfunction:: rocsparse_dcsrmv

.. doxygenfunction:: rocsparse_sellmv

.. doxygenfunction:: rocsparse_dellmv

.. doxygenfunction:: rocsparse_shybmv

.. doxygenfunction:: rocsparse_dhybmv

Sparse Level 3 Functions
------------------------

.. doxygenfunction:: rocsparse_scsrmm

.. doxygenfunction:: rocsparse_dcsrmm

Sparse Conversion Functions
---------------------------

.. doxygenfunction:: rocsparse_csr2coo

.. doxygenfunction:: rocsparse_coo2csr

.. doxygenfunction:: rocsparse_csr2csc_buffer_size

.. doxygenfunction:: rocsparse_scsr2csc

.. doxygenfunction:: rocsparse_dcsr2csc

.. doxygenfunction:: rocsparse_csr2ell_width

.. doxygenfunction:: rocsparse_scsr2ell

.. doxygenfunction:: rocsparse_dcsr2ell

.. doxygenfunction:: rocsparse_ell2csr_nnz

.. doxygenfunction:: rocsparse_sell2csr

.. doxygenfunction:: rocsparse_dell2csr

.. doxygenfunction:: rocsparse_scsr2hyb

.. doxygenfunction:: rocsparse_dcsr2hyb

.. doxygenfunction:: rocsparse_create_identity_permutation

.. doxygenfunction:: rocsparse_csrsort_buffer_size

.. doxygenfunction:: rocsparse_csrsort

.. doxygenfunction:: rocsparse_coosort_buffer_size

.. doxygenfunction:: rocsparse_coosort_by_row

.. doxygenfunction:: rocsparse_coosort_by_column

Enumerations
------------

This section provides all the enumerations used.

.. doxygenenum:: rocsparse_action

.. doxygenenum:: rocsparse_hyb_partition

.. doxygenenum:: rocsparse_index_base

.. doxygenenum:: rocsparse_layer_mode

.. doxygenenum:: rocsparse_matrix_type

.. doxygenenum:: rocsparse_operation

.. doxygenenum:: rocsparse_pointer_mode

.. doxygenenum:: rocsparse_status
