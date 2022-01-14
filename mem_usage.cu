#include <stdio.h>
#include "cuda.h"

int main()
{
    float free_m, total_m, used_m;
    size_t free_t, total_t;

    cudaMemGetInfo(&free_t, &total_t);

    total_m = total_t / 1048576.0;
    free_m = free_t / 1048576.0 ;
    used_m = total_m - free_m;

    printf("mem total %.2f MB\n", total_m);
    printf("mem free  %.2f MB\n", free_m);
    printf("mem used  %.2f MB\n", used_m);

    return 0;
}
