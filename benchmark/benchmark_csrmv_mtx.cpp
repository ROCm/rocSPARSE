/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "benchmark/benchmark.h"
#include "utils.h"

#include <rocsparse.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(stat)                                                     \
{                                                                           \
    hipError_t err = stat;                                                  \
    if (err != hipSuccess)                                                  \
    {                                                                       \
        fprintf(stderr, "HIP error: %d line: %d\n", err, __LINE__);         \
        exit(stat);                                                         \
    }                                                                       \
}

#define ROCSPARSE_CHECK(stat)                                               \
{                                                                           \
    rocsparse_status err = stat;                                           \
    if (err != rocsparse_status_success)                                    \
    {                                                                       \
        fprintf(stderr, "ROCSPARSE error: %d line: %d\n", err, __LINE__);   \
        exit(stat);                                                         \
    }                                                                       \
}

void csrmv(rocsparse_handle handle, rocsparse_operation trans,
           int nrow, int ncol, int nnz, const float *alpha,
           rocsparse_mat_descr descrA, const float *csrValA,
           const int *csrRowPtrA, const int *csrColIndA,
           const float *x, const float *beta, float *y)
{
    ROCSPARSE_CHECK(rocsparse_scsrmv(handle, trans, nrow, ncol, nnz, alpha,
                                     descrA, csrValA, csrRowPtrA, csrColIndA,
                                     x, beta, y));
}

void csrmv(rocsparse_handle handle, rocsparse_operation trans,
           int nrow, int ncol, int nnz, const double *alpha,
           rocsparse_mat_descr descrA, const double *csrValA,
           const int *csrRowPtrA, const int *csrColIndA,
           const double *x, const double *beta, double *y)
{
    ROCSPARSE_CHECK(rocsparse_dcsrmv(handle, trans, nrow, ncol, nnz, alpha,
                                     descrA, csrValA, csrRowPtrA, csrColIndA,
                                     x, beta, y));
}

template <typename ValueType>
void run_benchmark(benchmark::State &state, const hipStream_t stream, int batch,
                   rocsparse_handle handle, rocsparse_operation trans,
                   int nrow, int ncol, int nnz, rocsparse_mat_descr descr,
                   const ValueType *alpha, const ValueType *csrValA,
                   const int *csrRowPtrA, const int *csrColIndA,
                   const ValueType *x, const ValueType *beta, ValueType *y)
{
    // Warm up
    for (int i=0; i<10; ++i)
    {
        csrmv(handle, rocsparse_operation_none,
              nrow, ncol, nnz, alpha, descr, csrValA,
              csrRowPtrA, csrColIndA, x, beta, y);
    }
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _:state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i=0; i<batch; ++i)
        {
            csrmv(handle, rocsparse_operation_none,
                  nrow, ncol, nnz, alpha, descr, csrValA,
                  csrRowPtrA, csrColIndA, x, beta, y);
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double> >(end-start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations()*batch*
        (sizeof(ValueType)*(2*nrow+nnz)+sizeof(int)*(nrow+1+nnz)));
    state.SetItemsProcessed(state.iterations()*batch*2*nnz);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "%s <matrix.mtx> [<trials> <batch_size>]\n", argv[0]);
        return -1;
    }

    int trials = 200;
    int batch_size = 1;

    // Parse command line
    if (argc > 2)
    {
        trials = atoi(argv[2]);
    }
    if (argc > 3)
    {
        batch_size = atoi(argv[3]);
    }

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    benchmark::Initialize(&argc, argv);

    hipStream_t stream = 0;
    hipDeviceProp_t devProp;
    int device_id = 0;

    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    printf("[HIP] Device name: %s\n", devProp.name);

    // Read matrix from file
    int nrow;
    int ncol;
    int nnz;

    int *coo_row = NULL;
    int *coo_col = NULL;
    double *coo_val = NULL;

    if (readMatrixFromMTX(argv[1], nrow, ncol, nnz,
                          &coo_row, &coo_col, &coo_val) != 0)
    {
        fprintf(stderr, "Cannot read MTX file %s\n", argv[1]);
        return -1;
    }
    printf("[MTX] %d x %d matrix with %d nnz\n", nrow, ncol, nnz);

    // Convert to CSR (host) TODO
    int *Aptr = NULL;
    int *Acol = NULL;
    float *Avalf = NULL;
    double *Avald = NULL;

    coo_to_csr(nrow, ncol, nnz, coo_row, coo_col, coo_val,
               &Aptr, &Acol, &Avald);

    Avalf = (float*) malloc(sizeof(float)*nnz);
    for (int i=0; i<nnz; ++i)
    {
        Avalf[i] = (float) Avald[i];
    }

    // Clean up COO structures
    free(coo_row);
    free(coo_col);
    free(coo_val);

    // Sample some random data
    srand(12345ULL);

    float  alphaf = (float) rand() / RAND_MAX;
    double alphad = (double) rand() / RAND_MAX;
    float  betaf  = 0.0f;
    double betad  = 0.0;

    float  *xf = (float*) malloc(sizeof(float)*nrow);
    double *xd = (double*) malloc(sizeof(double)*nrow);
    for (int i=0; i<nrow; ++i)
    {
        xf[i] = (float) rand() / RAND_MAX;
        xd[i] = (double) rand() / RAND_MAX;
    }

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrA));

    // Offload data to device
    int *dAptr = NULL;
    int *dAcol = NULL;
    float  *dAvalf = NULL;
    float  *dxf = NULL;
    float  *dyf = NULL;
    double *dAvald = NULL;
    double *dxd = NULL;
    double *dyd = NULL;

    HIP_CHECK(hipMalloc((void**) &dAptr, sizeof(int)*(nrow+1)));
    HIP_CHECK(hipMalloc((void**) &dAcol, sizeof(int)*nnz));
    HIP_CHECK(hipMalloc((void**) &dAvalf, sizeof(float)*nnz));
    HIP_CHECK(hipMalloc((void**) &dxf, sizeof(float)*nrow));
    HIP_CHECK(hipMalloc((void**) &dyf, sizeof(float)*nrow));
    HIP_CHECK(hipMalloc((void**) &dAvald, sizeof(double)*nnz));
    HIP_CHECK(hipMalloc((void**) &dxd, sizeof(double)*nrow));
    HIP_CHECK(hipMalloc((void**) &dyd, sizeof(double)*nrow));

    HIP_CHECK(hipMemcpy(dAptr, Aptr, sizeof(int)*(nrow+1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, Acol, sizeof(int)*nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAvalf, Avalf, sizeof(float)*nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dxf, xf, sizeof(float)*nrow, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAvald, Avald, sizeof(double)*nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dxd, xd, sizeof(double)*nrow, hipMemcpyHostToDevice));

    // Clear up on host
    free(Aptr);
    free(Acol);
    free(Avalf);
    free(Avald);
    free(xf);
    free(xd);

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        benchmark::RegisterBenchmark("rocsparse_scsrmv", run_benchmark<float>,
                                     stream, batch_size,
                                     handle, rocsparse_operation_none,
                                     nrow, nrow, nnz, descrA, &alphaf, dAvalf,
                                     dAptr, dAcol, dxf, &betaf, dyf),
        benchmark::RegisterBenchmark("rocsparse_dcsrmv", run_benchmark<double>,
                                     stream, batch_size,
                                     handle, rocsparse_operation_none,
                                     nrow, nrow, nnz, descrA, &alphad, dAvald,
                                     dAptr, dAcol, dxd, &betad, dyd)
    };

    for (auto& b:benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
        b->Iterations(trials);
    }

    benchmark::RunSpecifiedBenchmarks();

    // Clear up on device
    HIP_CHECK(hipFree(dAptr));
    HIP_CHECK(hipFree(dAcol));
    HIP_CHECK(hipFree(dAvalf));
    HIP_CHECK(hipFree(dAvald));
    HIP_CHECK(hipFree(dxf));
    HIP_CHECK(hipFree(dxd));
    HIP_CHECK(hipFree(dyf));
    HIP_CHECK(hipFree(dyd));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrA));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
