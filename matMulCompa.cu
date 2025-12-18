#include <omp.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace std;

// Versiones de matmul a utilizar
void cpuVer(float *A, float *B, float *C, int n, int nt);
void gpuVer(float *A, float *B, float *C, int n);
//void gpusmVer();
//void gputcVer();

// Funciones extra (tools)
void imprMatriz(float *mat, int n, const char *msg);
void llenaMatriz(float *mat, int n);

// Definicion del kernel
__global__ void kernelMatmul(int n, float *A, float *B, float *C){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < n && ty < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[ty*n + k] * B[k*n + tx];
        C[ty*n + tx] = sum;
    }
}

// Definicion de versiones matmul
void cpuVer(float *A, float *B, float *C, int n, int nt){
    omp_set_num_threads(nt);        // setear cantidad de threads

    double t0 = omp_get_wtime();    // tomar tiempo al iniciar

    #pragma omp parallel for
    for (int i = 0; i < n; i++){    // matmul por threads
        for (int j = 0; j < n; j++){
            float sum = 0.0f;

            for (int k = 0; k < n; k++)
                sum += A[i*n + k] * B[k*n + j];
            
            C[i*n + j] = sum;
        }
    }

    double t1 = omp_get_wtime();    // tomar tiempo al terminar

    printf("Tiempo de computo para CPU: %f segundos \n", t1 - t0);
}

void gpuVer(float *A, float *B, float *C, int n){
    float *dA, *dB, *dC;
    size_t bytes = n*n * sizeof(float);

    cudaMalloc(&dA, bytes);                             // allocar memoria en GPU
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);   // copiar A y B a GPU
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    int BS = 16;                                        // configurar kernel,
    dim3 block(BS, BS);                                 // block y
    dim3 grid((n + BS - 1) / BS, (n + BS - 1) / BS);    // grid

    cudaEvent_t start, stop;                            // crear cuda events para medir tiempo
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);                             // tiempo al iniciar

    kernelMatmul<<<grid, block>>>(n, dA, dB, dC);       // ejecucion

    cudaEventRecord(stop);                              // tiempo al terminar
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);             // calculo de tiempo

    printf("Tiempo de computo para GPU: %f ms \n", ms);

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);   // copiar resultado en C

    cudaFree(dA);                                       // liberar memoria
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Definicion de funciones extra
void imprMatriz(float *A, int n, const char *msg){
    if(n > 40) return;

    printf("%s: \n", msg);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++)
            printf("%f ", A[i*n + j]);
        printf("\n");
    }

    printf("\n");
}

void llenaMatriz(float *A, int n){
    //random device rd;
    mt19937 gen(1313); 
    uniform_real_distribution<> dis(0.0, 1.0);  //randomizador real

    for (int i = 0; i < n*n; i++){
        //A[i] = (float) dis(gen);                //random real
		A[i] = (float) (rand() % n);            //random de prueba
    }
}

int main(int argc, char **argv){
    // 1. Entrada y argumentos

    if (argc != 4){
        printf("Error. Ejecutar como ./prog <n> <nt> <alg> \n");

        printf("n: largo de la matriz. \n");
        printf("nt: cantidad de CPU threads. \n"); 
        printf("alg: 1 (CPU), 2 (GPU), 3 (GPUsm), 4 (GPUtc). \n");

        exit(-1);
    }

    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int alg = atoi(argv[3]);

    // 1,5. Validar entradas
    if (n <= 0){
        printf("Entrada 'n' no válida (debe ser mayor a 0). \n");
        exit(-1);
    }
    
    if(nt < 1){
        printf("Entrada 'nt' no válida (debe ser mayor a 1). \n");
        exit(-1);
    }

    if(alg < 1 || alg > 4){
        printf("Entrada 'alg' no válida (debe estar entre 1 y 4). \n");
        exit(-1);
    }

    printf("args: n = %i, nt = %i, alg = %i \n", n, nt, alg);

    // 2. Allocar memoria para matrices
    float *A = new float[(n*n)];
    float *B = new float[(n*n)];
    float *C = new float[(n*n)];

    // 3. Inicializar y mostrar matrices
    llenaMatriz(A, n);
    llenaMatriz(B, n);

    for (int i = 0; i < n*n; i++)   // inicializar C con 0s
        C[i] = 0.0f;

    imprMatriz(A, n, "A");
    imprMatriz(B, n, "B");

    // 4. Ejecutar version del algoritmo a utilizar
    switch (alg)
    {
    case 1:
        cpuVer(A, B, C, n, nt);
        break;
    
    case 2:
        gpuVer(A, B, C, n);
        break;
    
    case 3:
        //gpusmVer();
        break;

    case 4:
        //gputcVer();
        break;

    default:
        break;
    }

    imprMatriz(C, n, "C = A*B");    // mostrar resultados

    delete[] A;                     // liberar memoria
    delete[] B;
    delete[] C;
}