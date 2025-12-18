OPT = -O3
SOURCE = matMulCompa.cu
BIN = prog
FLAGS = -Xcompiler -fopenmp

all:
	nvcc ${FLAGS} ${OPT} ${SOURCE} -o ${BIN}

clean:
	rm -f ${BIN}