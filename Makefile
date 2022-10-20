CFLAGS = -Wall -g -mp=gpu -gpu=cc80 -mp=noautopar -Msafeptr -march=native -O4 -mavx -mavx2 #-Minfo
DEFS = -DDTU_HPC
LIBS = -lcholmod -lm -lsuitesparseconfig -lcxsparse
INCL = ""

ifdef OWN_CHOLMOD
LIBS += -L$(HOME)/SuiteSparse/SuiteSparse-5.10.1/lib -L/usr/lib64 -lopenblaso-r0.3.3
INCL += -I$(HOME)/SuiteSparse/SuiteSparse-5.10.1/include/
$(info Using custom suite-sparse)
else 
LIBS += -L"/appl/SuiteSparse/5.1.2-sl73/lib/" -L"/appl/OpenBLAS/0.2.20/XeonGold6226R/gcc-6.4.0/lib" -lopenblas
INCL += -I"/appl/SuiteSparse/5.1.2-sl73/include/"
$(info Using old gbar suite-sparse)
endif

$(info LIBS are $(LIBS))
CC = nvc
CXX = g++

OBJ = stencil_methods.o stencil_assembly.o stencil_solvers.o stencil_grid_utility.o stencil_optimization.o local_matrix.o gpu_definitions.o

all: top3d

top3d: top3d.c $(OBJ)
	$(CC) -std=c11 $(CFLAGS) -o $@ $^ $(INCL) $(DEFS) $(LIBS)

%.o: %.c definitions.h
	$(CC) -std=c11 $(CFLAGS) -o $@ -c $< $(INCL) $(DEFS)

clean:
	-rm -f top3dmgcg_matrixfree top3dmgcg_eightColored benchmark test_stencil_methods core top3d *.core *.o 
