.. meta::
  :description: rocSPARSE documentation and API reference library
  :keywords: rocSPARSE, ROCm, API, documentation

.. _rocsparse-logging:

********************************************************************
Activity Logging
********************************************************************

Four different environment variables can be set to enable logging in rocSPARSE: ``ROCSPARSE_LAYER``, ``ROCSPARSE_LOG_TRACE_PATH``, ``ROCSPARSE_LOG_BENCH_PATH`` and ``ROCSPARSE_LOG_DEBUG_PATH``.

``ROCSPARSE_LAYER`` is a bit mask that enables logging, and where several logging modes (:ref:`rocsparse_layer_mode_`) can be specified as follows:

================================  =============================================================
``ROCSPARSE_LAYER`` unset         logging is disabled.
``ROCSPARSE_LAYER`` set to ``1``  trace logging is enabled.
``ROCSPARSE_LAYER`` set to ``2``  bench logging is enabled.
``ROCSPARSE_LAYER`` set to ``3``  trace logging and bench logging are enabled.
``ROCSPARSE_LAYER`` set to ``4``  debug logging is enabled.
``ROCSPARSE_LAYER`` set to ``5``  trace logging and debug logging are enabled.
``ROCSPARSE_LAYER`` set to ``6``  bench logging and debug logging are enabled.
``ROCSPARSE_LAYER`` set to ``7``  trace logging and bench logging and debug logging are enabled.
================================  =============================================================

When logging is enabled, each rocSPARSE function call will write the function name and function arguments to the logging stream. The default logging output is streamed to ``stderr``.

.. note::

    Performance will degrade when logging is enabled. By default, the environment variable ``ROCSPARSE_LAYER`` is unset and logging is disabled.

To capture activity logging in a file set the following environment variables as needed: 

  * ``ROCSPARSE_LOG_TRACE_PATH`` specifies a path and file name to capture trace logging streamed to that file
  * ``ROCSPARSE_LOG_BENCH_PATH`` specifies a path and file name to capture bench logging 
  * ``ROCSPARSE_LOG_DEBUG_PATH`` specifies a path and file name to capture debug logging 

.. note::

    If the file cannot be opened, logging output is streamed to ``stderr``.


