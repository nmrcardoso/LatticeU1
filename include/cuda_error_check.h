
#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H


#include <cstdlib>


#include <cuda.h> 
#include <cuda_runtime.h>
//#include <cusolverDn.h>
//#include <cusolverSp.h>
#include <cusparse.h>
#include <curand.h>

#include "alloc.h"
#include "enum.h"

 
namespace U1{
  

void cudaDevSync_(const char *func, const char *file, int line);
void cudaSafeCall_(const char *func, const char *file, int line, cudaError_t call);
void cusparseSafeCall_(const char *func, const char *file, int line, cusparseStatus_t call);
//void cusolverSafeCall_(const char *func, const char *file, int line, cusolverStatus_t call);
//void cublasSafeCall_(const char *func, const char *file, int line, cublasStatus_t call);
void curandSafeCall_(const char *func, const char *file, int line, curandStatus_t call);

#define cudaDevSync() cudaDevSync_(__func__, U1::file_name(__FILE__), __LINE__)
#define cudaSafeCall(call) cudaSafeCall_(__func__, U1::file_name(__FILE__), __LINE__, call)
#define cusparseSafeCall(call) cusparseSafeCall_(__func__, U1::file_name(__FILE__), __LINE__, call)
#define cusolverSafeCall(call) cusolverSafeCall_(__func__, U1::file_name(__FILE__), __LINE__, call)
#define cublasSafeCall(call) cublasSafeCall_(__func__, U1::file_name(__FILE__), __LINE__, call)
#define curandSafeCall(call) curandSafeCall_(__func__, U1::file_name(__FILE__), __LINE__, call)


void cudaCheckError_(const char *func, const char *file, int line, const char *errorMessage);
#define cudaCheckError(errorMessage) cudaCheckError_(__func__, U1::file_name(__FILE__), __LINE__, errorMessage)





void Start(int gpuid, Verbosity verb=VERBOSE, TuneMode tune=TUNE_YES);

void Finalize_(const char *func, const char *file, int line, int error);
#define Finalize(_error_) Finalize_(__func__, U1::file_name(__FILE__), __LINE__, _error_)
 
#define printfSbreak(abc,...) printf(abc, ##__VA_ARGS__);
#define printfError(abc,...) do {  \
  printf("Error in %s: %d in %s()\n\t",__FILE__,__LINE__, __func__);  \
  printf(abc, ##__VA_ARGS__); \
  Finalize(1); \
} while (0)
#define errorSbreak(abc,...) do {  \
  printf("Error in %s: %d in %s()\n\t",__FILE__,__LINE__, __func__);  \
  printf(abc, ##__VA_ARGS__); \
  Finalize(1); \
} while (0)



}

#endif

