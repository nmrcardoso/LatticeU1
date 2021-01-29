#ifndef ___ENUM_ALLOC_H___
#define ___ENUM_ALLOC_H___

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>



namespace U1{

typedef enum Verbosity_s {
 SILENT,
 SUMMARIZE,
 VERBOSE,
 DEBUG_VERBOSE
} Verbosity;



typedef enum TuneMode_S {
    TUNE_NO,
    TUNE_YES
} TuneMode;


typedef enum StoreMode_S {
    Host,
    Device,
    Managed
} StoreMode;


#define ALLOC_INVALID_ENUM INT_MIN

typedef enum MemoryType_s {
	ALLOC_MEMORY_DEVICE,
	ALLOC_MEMORY_PINNED,
	ALLOC_MEMORY_MAPPED,
	ALLOC_MEMORY_MANAGED,
	ALLOC_MEMORY_INVALID = ALLOC_INVALID_ENUM
} MemoryType;


// Where the field is stored
typedef enum FieldLocation_s {
	CPU_FIELD_LOCATION = 1,
	CUDA_FIELD_LOCATION = 2,
	INVALID_FIELD_LOCATION = ALLOC_INVALID_ENUM
} FieldLocation;









inline cudaMemcpyKind GetMemcpyKind(const StoreMode in, const StoreMode out){
    cudaMemcpyKind cptype;
    switch( in ) {
    case Host :
        switch( out ) {
        case Host:
            cptype = cudaMemcpyHostToHost;
            break;
        case Device:
            cptype = cudaMemcpyHostToDevice;
            break;
        case Managed:
            cptype = cudaMemcpyHostToDevice;
            break;
        }
        break;
    case Device: 
        switch( out ) {
        case Host:
            cptype = cudaMemcpyDeviceToHost;
            break;
        case Device:
            cptype = cudaMemcpyDeviceToDevice;
            break;
        case Managed:
            cptype = cudaMemcpyDeviceToDevice;
            break;
        }
        break;			
    case Managed: 
        switch( out ) {
        case Host:
            cptype = cudaMemcpyDeviceToHost;
            break;
        case Device:
            cptype = cudaMemcpyDeviceToDevice;
            break;
        case Managed:
            cptype = cudaMemcpyDeviceToDevice;
            break;
        }
        break;			
    }
    return cptype;
}




inline std::string GetMemcpyKindString(cudaMemcpyKind cptype){
	std::string stype = "";
    switch( cptype ) {
		case cudaMemcpyHostToHost:
			stype = "cudaMemcpyHostToHost";
			break;
		case cudaMemcpyHostToDevice:
			stype = "cudaMemcpyHostToDevice";
			break;
		case cudaMemcpyDeviceToHost:
			stype = "cudaMemcpyDeviceToHost";
			break;
		case cudaMemcpyDeviceToDevice:
			stype = "cudaMemcpyDeviceToDevice";
            break;			
    }
	return stype;
}	


}


#endif 


