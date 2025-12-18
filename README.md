# README.md

## Prueba Chica 3 --- INFO188

### Acceso indirecto en GPU usando CUDA

Este programa hace x

------------------------------------------------------------------------

## Modo de uso

    ./prog <n> <nt> <alg>

-   `n`: Tamaño del largo de las matrices.
-   `nt`: Número de threads a utilizar.
-   `alg`: Versión del algoritmo a utilizar:
            1. `cpuVer`: CPU multicore.
            2. `gpuVer`: GPU básica.
            3. `gpusmVer`: GPU con memoria compartida.
            4. `gputcVer`: GPU con tensor cores.<>

Ejemplo:

    ./prog 4 2 2

------------------------------------------------------------------------

