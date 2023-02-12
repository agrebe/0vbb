CC = icpc
CFLAGS = -g -O3 -Wall -qopenmp -shared-intel -qopt-streaming-stores always -march=native -ffast-math \
				 -I$(HOME)/work/stampede2/install/fftw-3.3.8/include
LIBS = -lfftw3_omp -lfftw3 -mkl
LDFLAGS = -L$(HOME)/work/stampede2/install/fftw-3.3.8/lib

# create main executable
qc: color_tensor.o gamma_container.o main.o read_prop.o \
	run_4pt.o run_sigma_4pt.o
	$(CC) $(CFLAGS) *.o $(LDFLAGS) $(LIBS)

%.o : %.C
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f *.o a.out
