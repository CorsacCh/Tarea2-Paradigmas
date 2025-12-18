#include <cuda.h>
#include <cstdio>
#include <cstdlib>

void cpuVer();
void gpuVer();
void gpusmVer();
void gputcVer();

int main(int argc, char **argv){
    // 1. Argumentos

    if (argc != 4){
        printf("Error. Ejecutar como ./prog <n> <nt> <alg> \n");
        printf("n: cantidad de CPU threads - alg: 1 (GPU), 2 (GPU), 3 (GPUsm), 4 (GPUtc) \n");
        exit(-1);
    }

    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int alg = atoi(argv[3]);
}