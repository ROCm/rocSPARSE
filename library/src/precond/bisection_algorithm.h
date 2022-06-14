auto bisection_algorithm = [=](auto const row) -> void {
    // Diagonal entry point of the current row
    rocsparse_int row_diag = csr_diag_ind[row];

    // Row entry point
    rocsparse_int row_begin = csr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;

    // Loop over column of current row
    for(rocsparse_int j = row_begin; j < row_diag; ++j)
    {
        // Column index currently being processes
        rocsparse_int local_col = csr_col_ind[j] - idx_base;

        // Corresponding value
        T local_val = csr_val[j];

        // End of the row that corresponds to local_col
        rocsparse_int local_end = csr_row_ptr[local_col + 1] - idx_base;

        // Diagonal entry point of row local_col
        rocsparse_int local_diag = csr_diag_ind[local_col];

        // Structural zero pivot, do not process this row
        if(local_diag == -1)
        {
            local_diag = local_end - 1;
        }

        // Spin loop until dependency has been resolved
        int          local_done    = atomicOr(&done[local_col], 0);
        unsigned int times_through = 0;
        while(!local_done)
        {
            if(SLEEP)
            {
                for(unsigned int i = 0; i < times_through; ++i)
                {
                    __builtin_amdgcn_s_sleep(1);
                }

                if(times_through < 3907)
                {
                    ++times_through;
                }
            }

            local_done = atomicOr(&done[local_col], 0);
        }

        // Make sure updated csr_val is visible
        __threadfence();

        // Load diagonal entry
        T diag_val = csr_val[local_diag];

        // Numeric boost
        if(boost)
        {
            diag_val = (boost_tol >= rocsparse_abs(diag_val)) ? boost_val : diag_val;

            __threadfence();

            if(lid == 0)
            {
                csr_val[local_diag] = diag_val;
            }
        }
        else
        {
            // Row has numerical zero diagonal
            if(diag_val == static_cast<T>(0))
            {
                if(lid == 0)
                {
                    // We are looking for the first zero pivot
                    atomicMin(zero_pivot, local_col + idx_base);
                }

                // Skip this row if it has a zero pivot
                break;
            }
        }

        csr_val[j] = local_val = local_val / diag_val;

        // Loop over the row the current column index depends on
        // Each lane processes one entry
        rocsparse_int l = j + 1;
        for(rocsparse_int k = local_diag + 1 + lid; k < local_end; k += WFSIZE)
        {
            // Perform a binary search to find matching columns
            rocsparse_int r     = row_end - 1;
            rocsparse_int m     = (r + l) >> 1;
            rocsparse_int col_j = csr_col_ind[m];

            rocsparse_int col_k = csr_col_ind[k];

            // Binary search
            while(l < r)
            {
                if(col_j < col_k)
                {
                    l = m + 1;
                }
                else
                {
                    r = m;
                }

                m     = (r + l) >> 1;
                col_j = csr_col_ind[m];
            }

            // Check if a match has been found
            if(col_j == col_k)
            {
                // If a match has been found, do ILU computation
                csr_val[l] = rocsparse_fma(-local_val, csr_val[k], csr_val[l]);
            }
        }
    }

    // Make sure updated csr_val is written to global memory
    __threadfence();

    if(lid == 0)
    {
        // First lane writes "we are done" flag
        atomicOr(&done[row], 1);
    }
};
