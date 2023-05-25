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
