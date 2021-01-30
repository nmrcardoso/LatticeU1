
#ifndef __OBJECT_H__
#define __OBJECT_H__


#include <iostream>
#include <alloc.h>
#include <verbose.h>

namespace U1 {
  
class Object {
  public:
    inline Object() { }
    inline virtual ~Object() { }
    
    inline static void* operator new(std::size_t size) {
      void *ptr = safe_malloc(size);
      if (getVerbosity() >= DEBUG_VERBOSE) 
      	std::cout << "Allocate pointer: " << ptr << std::endl;
      return ptr;
    }
    
    inline static void operator delete(void* ptr) {
      if (getVerbosity() >= DEBUG_VERBOSE) 
      	std::cout << "Release pointer: " << ptr << std::endl;
      host_free(ptr);
    }
  
    inline static void* operator new[](std::size_t size) {
      void *ptr = safe_malloc(size);
      if (getVerbosity() >= DEBUG_VERBOSE) 
      	std::cout << "Allocate pointer: " << ptr << std::endl;
      return ptr;
    }
  
    inline static void operator delete[](void* ptr) {
      if (getVerbosity() >= DEBUG_VERBOSE) 
      	std::cout << "Release pointer: " << ptr << std::endl;
      host_free(ptr);
    }
  };

} // namespace U1

#endif
