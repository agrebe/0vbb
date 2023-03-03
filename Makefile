QDP = /work/01112/tg803910/stampede2/install/qdpxx-scalar
QPHIX = /work/01112/tg803910/stampede2/install/qphix-scalar
QDPCONFIG = $(QDP)/bin/qdp++-config

CC = icpc
CFLAGS = -g -O3 -Wall -qopenmp -shared-intel -qopt-streaming-stores always -march=native -ffast-math \
				 -I$(HOME)/work/stampede2/install/fftw-3.3.8/include \
				 -I./qphix-wrapper
LIBS = -lfftw3_omp -lfftw3 -mkl \
			 -lqphix-wrapper \
			 -lqphix_solver -lqphix_codegen $(shell $(QDPCONFIG) --libs) 
LDFLAGS = -L$(HOME)/work/stampede2/install/fftw-3.3.8/lib \
					-L./qphix-wrapper \
					-L$(QPHIX)/lib $(shell $(QDPCONFIG) --ldflags)

# create main executable
qc: color_tensor.o gamma_container.o main.o read_prop.o \
	run_baryon_2pt.o run_dibaryon_2pt.o \
	run_3pt.o run_sigma_3pt.o run_nnpp_3pt.o \
	run_4pt.o run_sigma_4pt.o run_nnpp_4pt.o
	$(CC) $(CFLAGS) *.o $(LDFLAGS) $(LIBS) 

%.o : %.C
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f *.o a.out
