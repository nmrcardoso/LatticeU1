
#ifndef __OBJECT_H__
#define __OBJECT_H__


#include <iostream>
#include <alloc.h>

namespace U1 {
  
  class Object {
    
    
  public:
    inline Object() { }
    inline virtual ~Object() { }
    
    inline static void* operator new(std::size_t size) {
      //std::cout << "Allocate pointer: " << std::endl;
      return safe_malloc(size);
    }
    
    inline static void operator delete(void* p) {
      //std::cout << "Release pointer: " << p << std::endl;
      host_free(p);
    }
  
    inline static void* operator new[](std::size_t size) {
      //std::cout << "Allocate pointer: " << std::endl;
      return safe_malloc(size);
    }
  
    inline static void operator delete[](void* p) {
      //std::cout << "Release pointer: " << p << std::endl;
      host_free(p);
    }
  };

} // namespace U1

#endif
