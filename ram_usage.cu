#include <iostream>
#include <unistd.h>
#include "cuda.h"

int main()
{
    // show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;

    while (true )
    {
        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

        if ( cudaSuccess != cuda_status ){
            std::cout << "Error: cudaMemGetInfo fails, "
                      << cudaGetErrorString(cuda_status) << std::endl;
            exit(1);
        }

        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;

        std::cout << "GPU memory usage: used = " << used_db/1024.0/1024.0 << ", free = "
                  << free_db/1024.0/1024.0 << " MB, total = " << total_db/1024.0/1024.0 << " MB"
                  << std::endl; sleep(1);
        break;
    }

    return 0;
}
