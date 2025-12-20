# TAREA 2 — INFO188  
## MATMUL COMPA

## Integrantes
- **Jorge Cheuquemán**
- **Cristian Vera**

---

## Descripción del proyecto

Este proyecto compara **tres implementaciones** de la multiplicación de matrices cuadradas (N×N):

\[
A \times B = C
\]

Implementaciones evaluadas:

1. **CPU multicore (OpenMP)**  
   Usa múltiples hilos del procesador.
2. **GPU básica (CUDA naive)**  
   Cada thread de la GPU calcula un elemento de la matriz resultado.
3. **GPU con memoria compartida (CUDA tiled)**  
   Optimiza la versión GPU reutilizando *tiles* en **shared memory**.

En cada ejecución, el programa realiza la multiplicación con el algoritmo seleccionado y **mide el tiempo de cómputo** en segundos.

---

## Requisitos

- GPU **NVIDIA** con soporte **CUDA**
- **CUDA Toolkit** (incluye `nvcc`)
- Compilador host con soporte **OpenMP** (por ejemplo `g++`)
- Sistema operativo Linux

---

## Hardware utilizado

Equipo de pruebas: **Lenovo Legion 5**

- **CPU:** AMD Ryzen 5 5600H (6 cores / 12 threads)  
- **RAM:** 16 GB  
- **GPU:** NVIDIA GeForce RTX 3060  

---

## Compilación

El proyecto incluye un **Makefile** que compila el código CUDA habilitando soporte para OpenMP en la CPU.

### Archivos relevantes
- `matMulCompa.cu` : Código fuente principal.
- `Makefile` : Reglas de compilación.
- `prog` : Binario generado.

### Comando de compilación

Desde el directorio del proyecto ejecutar:

```bash
make
```

Esto invoca internamente el siguiente comando:

```bash
nvcc -Xcompiler -fopenmp -O3 matMulCompa.cu -o prog
```

Donde:
- `-O3` activa optimizaciones agresivas del compilador.
- `-fopenmp` habilita paralelismo OpenMP en el código CPU.
- `nvcc` compila y enlaza código CUDA + C/C++.

### Limpiar binarios

Para eliminar el ejecutable generado:

```bash
make clean
```

---

## Modo de uso

```bash
./prog <n> <nt> <alg>
```

### Parámetros

- `n`: Tamaño de las matrices (N×N).
- `nt`: Número de threads a utilizar.
- `alg`: Algoritmo a ejecutar.

| Valor | Algoritmo |
|-----:|-----------|
| 1 | CPU multicore (OpenMP) |
| 2 | GPU básica (CUDA naive) |
| 3 | GPU con shared memory (CUDA tiled) |

### Ejemplo

```bash
./prog 4 2 2
```

---

## Resultados y explicación de gráficos

En esta tarea se generaron gráficos que muestran cómo varía el rendimiento al aumentar el tamaño de la matriz **N×N**.

### Definición de N

`N` corresponde a la dimensión de la matriz cuadrada.

Ejemplo:  
- `N = 256` ⇒ multiplicación de matrices **256 × 256**.

---

## Gráfico: Tiempo vs N

**Tiempo vs N (CPU nt=12 vs GPU vs GPUsm)**

Este gráfico compara el **tiempo promedio de cómputo** de las tres implementaciones para distintos tamaños de matriz.

[GRÁFICO TIEMPO VS TODOS]

### Ejes
- **Eje X:** N (potencias de 2)
- **Eje Y:** Tiempo promedio (s), **escala logarítmica**

### Curvas
- **CPU (nt = 12):** OpenMP.
- **GPU (ALG = 2):** CUDA naive.
- **GPUsm (ALG = 3):** CUDA con shared memory.

### Interpretación

- A medida que **N aumenta**, el tiempo crece en todas las versiones.
- La **CPU** escala mucho peor que las versiones GPU.
- **GPU** y **GPUsm** mantienen tiempos mucho menores (varios órdenes de magnitud).
- **GPUsm** suele ser igual o ligeramente más rápida que la GPU naive gracias al uso de **shared memory**.

### Conclusión principal

La **GPU** ofrece el mejor rendimiento al aumentar `N`, y el uso de **shared memory** aporta una mejora adicional respecto a la versión naive.

---

## Gráficos: % Speedup vs N

### Fórmula utilizada

%speedup = (T_seq / T_par) * 100

Donde:

- `T_seq`: Tiempo de la versión secuencial.  
- `T_par`: Tiempo de la versión paralela.

Es decir, se compara cada tiempo de cómputo con el algoritmo secuencial, de lo que sale un porcentaje de eficiencia del algoritmo secuencial corre el algoritmo comparado.

Ejemplo:
- `9%` ⇒ El algoritmo comparado se ejecuta a un **9%** de la potencia del algoritmo secuencial. O sea, es un 91% más lento.
- `100%` ⇒ El algoritmo comparado tiene el mismo tiempo de cómputo que el algoritmo secuencial.
- `190%` ⇒ El tiempo de cómputo del algoritmo comparado fue un 90% más rápido.

---

### Gráfico 1: CPU — % Speedup vs N (líneas por nt)

[GRÁFICO CPUT]

Compara la versión CPU multicore usando distintos números de hilos (`nt = 2, 4, 6, 8, 12`) contra una versión secuencial.

#### Interpretación

- Para **N pequeños**, el speedup puede ser bajo o irregular debido al overhead de paralelización.
- Para **N grandes**, el cómputo domina y el speedup aumenta al incrementar `nt`.

#### Conclusión principal

En matrices grandes, **OpenMP aprovecha mejor el CPU** al usar más hilos.

---

### Gráfico 2: GPU — % Speedup vs N (GPU y GPUsm)

[GRAFICO GPUGPUSM]

Compara dos implementaciones en GPU:

- **GPU (ALG = 2):** Versión básica.
- **GPUsm (ALG = 3):** Versión optimizada con shared memory.

Ambas se comparan contra la misma versión secuencial.

#### Interpretación

- Para **N pequeños**, el overhead de lanzamiento del kernel afecta el rendimiento.
- Para **N grandes**, el speedup aumenta fuertemente gracias al paralelismo masivo.
- **GPUsm** suele superar a la versión naive por reducir accesos a memoria global.

#### Conclusión principal

En matrices grandes, la **GPU entrega el mayor speedup**, y el uso de **shared memory** es una optimización efectiva.

---

## Observaciones finales

- Para **N pequeños**, el rendimiento está dominado por overhead (threads y kernels).
- Para **N grandes**, **GPUsm** tiende a mejorar consistentemente gracias a la reutilización de datos en shared memory.

## Resultados finales de Speedup

[tabla]
