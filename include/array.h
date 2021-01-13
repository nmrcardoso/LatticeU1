#ifndef __ARRAY__
#define __ARRAY__

#include <iostream>

#include <cuda.h>
#include "cuda_error_check.h"
#include "enum.h"
#include "parameters.h"


namespace U1{

template<class Real>
class Array{	
	public:
	Array(){ ptr = 0; size = 0; ptr_backup = 0; location = Host;}
	Array(ReadMode location_){ ptr = 0; size = 0; ptr_backup = 0; location = location_;}
	Array(ReadMode location, size_t size):location(location), size(size){
		ptr = 0;
		ptr_backup = 0;
		ptr = Allocate(size);
	}
	~Array(){ Release(); }
	Real*  getPtr(){ return ptr; }
	size_t Size(){ return size; }
	ReadMode Location(){ return location; }
	
	
	
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
				switch(location){
					case Host:
						host_free(ptr);
					break;
					case Device:
						dev_free(ptr);
					break;
				}
				ptr = 0;
				size = in->Size();
				ptr = Allocate(size);
			}
		}	
		else{
			size = in->Size();
			ptr = Allocate(size);
		}
		//std::cout << ptr << "\t" << size << std::endl;
		cpy(in->getPtr(), in->location, getPtr(), location, in->Size());
	}
	void Clear(){
		switch(location){
			case Host:
				memset(ptr, 0, size*sizeof(Real));
			break;
			case Device:
				cudaSafeCall(cudaMemset(ptr, 0, size*sizeof(Real)));
			break;
		}
	
	
	}
	
	void Backup(){
		ptr_backup = Allocate(size);
		cudaSafeCall(cudaMemcpy(ptr_backup, ptr, size*sizeof(Real), GetMemcpyKind( location, location)));
	}
	void Restore(){		
		cudaSafeCall(cudaMemcpy(ptr, ptr_backup, size*sizeof(Real), GetMemcpyKind( location, location)));
		if(ptr_backup){
			switch(location){
				case Host:
					host_free(ptr_backup);
				break;
				case Device:
					dev_free(ptr_backup);
				break;
			}
			ptr_backup = 0;
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
	ReadMode location; //HOST or DEVICE
	size_t size;
	
	Real* Allocate(size_t insize){
		Real* tmp = 0;
		switch(location){
			case Host:
				tmp = (Real*)safe_malloc(insize*sizeof(Real));
				if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Allocate array " << tmp << " in Host with: " << float(size*sizeof(Real))/1048576. << " MB" << std::endl;
			break;
			case Device:
				tmp = (Real*)dev_malloc(insize*sizeof(Real));
				if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Allocate array " << tmp << " in Device with: " << float(size*sizeof(Real))/1048576. << " MB" << std::endl;
			break;
		}	
		return tmp;
	}	
	void cpy(Real *in, ReadMode lin, Real *out, ReadMode lout, size_t s_in){
		cudaSafeCall(cudaMemcpy(out, in, s_in*sizeof(Real), GetMemcpyKind( lin, lout)));
	}
	
	void Release(){ 
		if(ptr){
			switch(location){
				case Host:
				    if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Release array " << ptr << " in Host with: " << float(size*sizeof(Real))/1048576. << " MB" << std::endl;
					host_free(ptr);
				break;
				case Device:
					if(getVerbosity() >= DEBUG_VERBOSE) std::cout << "Release array " << ptr << " in Device with: " << float(size*sizeof(Real))/1048576. << " MB" << std::endl;
					dev_free(ptr);
				break;
			}
			ptr = 0;
		}	
		if(ptr_backup){
			switch(location){
				case Host:
					host_free(ptr_backup);
				break;
				case Device:
					dev_free(ptr_backup);
				break;
			}
			ptr_backup = 0;
		}	
	}
};







}

#endif
