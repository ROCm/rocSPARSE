auto hash_algorithm = [=](auto const row) -> void {
    // Diagonal entry point of the current row
    rocsparse_int row_diag = csr_diag_ind[row];

    // Row entry point
    rocsparse_int row_begin = csr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;

    // Fill hash table
    // Loop over columns of current row and fill hash table with row dependencies
    // Each lane processes one entry
    for(rocsparse_int j = row_begin + lid; j < row_end; j += WFSIZE)
    {
        // Insert key into hash table
        rocsparse_int key = csr_col_ind[j];
        // Compute hash
        rocsparse_int hash = (key * 103) & (WFSIZE * HASH - 1);

        // Hash operation
        while(true)
        {
            if(table[hash] == key)
            {
                // key is already inserted, done
                break;
            }
            else if(atomicCAS(&table[hash], -1, key) == -1)
            {
                // inserted key into the table, done
                data[hash] = j;
                break;
            }
            else
            {
                // collision, compute new hash
                hash = (hash + 1) & (WFSIZE * HASH - 1);
            }
        }
    }

    __threadfence_block();

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
        while(!atomicOr(&done[local_col], 0))
            ;

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
        for(rocsparse_int k = local_diag + 1 + lid; k < local_end; k += WFSIZE)
        {
            // Get value from hash table
            rocsparse_int key = csr_col_ind[k];

            // Compute hash
            rocsparse_int hash = (key * 103) & (WFSIZE * HASH - 1);

            // Hash operation
            while(true)
            {
                if(table[hash] == -1)
                {
                    // No entry for the key, done
                    break;
                }
                else if(table[hash] == key)
                {
                    // Entry found, do ILU computation
                    rocsparse_int idx_data = data[hash];
                    csr_val[idx_data] = rocsparse_fma(-local_val, csr_val[k], csr_val[idx_data]);
                    break;
                }
                else
                {
                    // Collision, compute new hash
                    hash = (hash + 1) & (WFSIZE * HASH - 1);
                }
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
