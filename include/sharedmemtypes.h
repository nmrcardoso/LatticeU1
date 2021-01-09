#ifndef SHARED_MEMORY_TYPES_H
#define SHARED_MEMORY_TYPES_H


#include <complex.h>




template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
        {
            extern __shared__ int __ssmem[];
            return (T*)__ssmem;
        }

    __device__ inline operator const T*() const
        {
            extern __shared__ int __ssmem[];
            return (T*)__ssmem;
        }
};
template<>
struct SharedMemory<complexs>
{
    __device__ inline operator       complexs*()
    {
        extern __shared__ complexs __ssmem_s[];
        return (complexs*)__ssmem_s;
    }

    __device__ inline operator const complexs*() const
    {
        extern __shared__ complexs __ssmem_s[];
        return (complexs*)__ssmem_s;
    }
};
template<>
struct SharedMemory<complexd>
{
    __device__ inline operator       complexd*()
    {
        extern __shared__ complexd __ssmem_d[];
        return (complexd*)__ssmem_d;
    }

    __device__ inline operator const complexd*() const
    {
        extern __shared__ complexd __ssmem_d[];
        return (complexd*)__ssmem_d;
    }
};




#endif // #ifndef SHARED_MEMORY_TYPES_H
