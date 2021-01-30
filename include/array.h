#ifndef __ARRAY__
#define __ARRAY__

#include <iostream>

#include <cuda.h>
#include "cuda_error_check.h"
#include "enum.h"
#include "parameters.h"
#include "object.h"


namespace U1{

template<class Real>
class Array : public Object{ 	
	public:
	Array(){ 
		ptr = 0; 
		size = 0; 
		ptr_backup = 0; 
		location = Host;
	}
	
	Array(StoreMode location_){ 
		ptr = 0; 
		size = 0; 
		ptr_backup = 0; 
		location = location_;
	}
	
	Array(StoreMode location, size_t size):location(location), size(size){
		ptr = 0;
		ptr_backup = 0;
		Allocate(&ptr, location, size);
	}
	~Array(){ Release(); }
	
	Real*  getPtr(){ return ptr; }
	
	size_t Size(){ return size; }
	
	StoreMode Location(){ return location; }	
	
    M_HOSTDEVICE Real& operator()(const int i){
        return ptr[i];
    }

    M_HOSTDEVICE Real operator()(const int i) const{
        return ptr[i];
    }
    
    M_HOSTDEVICE Real& operator[](const int i){
        return ptr[i];
    }

    M_HOSTDEVICE Real operator[](const int i) const{
        return ptr[i];
    }
    M_HOSTDEVICE Real& at(const int i) {
        return ptr[i];
    }
    M_HOSTDEVICE Real at(const int i) const{
        return ptr[i];
    }
	
	void Copy(Array *in){
		if(ptr){
			if( size != in->Size() ){
				Release(&ptr, location);
				size = in->Size();
				Allocate(&ptr, location, size);
			}
		}	
		else{
			size = in->Size();
			Allocate(&ptr, location, size);
		}
		cpy(in->getPtr(), in->location, getPtr(), location, in->Size());
	}
	void Clear(){
		if(ptr){
			switch(location){
				case Host:
					memset(ptr, 0, size*sizeof(Real));
				break;
				case Device:
					cudaSafeCall(cudaMemset(ptr, 0, size*sizeof(Real)));
				case Managed:
					cudaSafeCall(cudaMemset(ptr, 0, size*sizeof(Real)));
				break;
			}
		}	
	}
	
	size_t Bytes() { return size * sizeof(Real); }
	
	void Backup(){
		Release(&ptr_backup, backup_loc);
	    size_t mfree, mtotal;
	    int gpuid=-1;
	    cudaSafeCall(cudaGetDevice(&gpuid));
	    cudaSafeCall(cudaMemGetInfo(&mfree, &mtotal));
	    std::cout << "Device memory free: " << mfree/(float)(1048576) << " MB of " << mtotal/(float)(1048576) << " MB." << std::endl;
	    std::cout << "Memory size required to backup array: " << Bytes()/(float)(1048576) << " MB." << std::endl;
	    if(mfree > Bytes()){
	        std::cout << "Backup array in Device..." << std::endl;
	        backup_loc = Device;
	    } 
	    else{
	       std::cout << "Backup array in Host..." << std::endl;
	        backup_loc = Host;
	    }	    
		Allocate(&ptr_backup, backup_loc, size);
		cpy(ptr, location, ptr_backup, backup_loc, size);
	}
	
	void Restore(){
		if(ptr_backup){
			cpy(ptr_backup, backup_loc, ptr, location, size);
			Release(&ptr_backup, backup_loc);
		}
		else{
			std::cout << "Error: Cannot restore a dealocated pointer..." << std::endl;
			exit(1);
		}		
	}
    
	friend M_HOST std::ostream& operator<<( std::ostream& out, Array<Real> M ) {
		//out << std::scientific;
		//out << std::setprecision(14);
		for(int i = 0; i < M.size; i++)
			out << i << '\t' << M.ptr[i] << std::endl;;
		return out;
	}
	
	friend M_HOST std::ostream& operator<<( std::ostream& out, Array<Real> *M ) {
		//out << std::scientific;
		//out << std::setprecision(14);
		for(int i = 0; i < M->size; i++)
			out << i << '\t' << M->ptr[i] << std::endl;;
		return out;
	}
	
	
	private:
	Real *ptr;
	Real *ptr_backup;
	StoreMode location; //HOST or DEVICE
	StoreMode backup_loc; //HOST or DEVICE
	size_t size;
	
	void cpy(Real *in, StoreMode lin, Real *out, StoreMode lout, size_t s_in){
		cudaSafeCall(cudaMemcpy(out, in, s_in*sizeof(Real), GetMemcpyKind( lin, lout)));
	}
	
	void Release(){ 
		Release(&ptr, location);
		Release(&ptr_backup, backup_loc);	
	}
	
////////////////////////////////////////////////////////////////////////////////////
	void Allocate(Real **ptr_in, StoreMode loc, size_t insize){
		if(*ptr_in) Release(ptr_in, loc);
		switch(loc){
			case Host:
				*ptr_in = (Real*)safe_malloc(insize*sizeof(Real));
				if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Allocate array " << *ptr_in << " in Host with: " << float(insize*sizeof(Real))/1048576. << " MB" << std::endl;
			break;
			case Device:
				*ptr_in = (Real*)dev_malloc(insize*sizeof(Real));
				if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Allocate array " << *ptr_in << " in Device with: " << float(insize*sizeof(Real))/1048576. << " MB" << std::endl;
			break;
			case Managed:
				*ptr_in = (Real*)managed_malloc(insize*sizeof(Real));
				if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Allocate array " << *ptr_in << " in Managed with: " << float(insize*sizeof(Real))/1048576. << " MB" << std::endl;
			break;
		}	
	}
	void Release(Real **ptr_in, StoreMode loc){
		if(*ptr_in){
			switch(loc){
				case Host:
				    if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Release array " << *ptr_in << " in Host" << std::endl;
					host_free(*ptr_in);
				break;
				case Device:
					if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Release array " << *ptr_in << " in Device" << std::endl;
					dev_free(*ptr_in);
				break;
				case Managed:
					if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Release array " << *ptr_in << " in Managed" << std::endl;
					managed_free(*ptr_in);
				break;
			}
			*ptr_in = 0;
		}
	}
	
};


template<class Real>
Array<Real>* LatticeConvert(Array<Real>* lat, bool eo_to_no);

}

#endif
