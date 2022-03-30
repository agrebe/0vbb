CC = icpc
CFLAGS = -g -O3 -Wall -qopenmp -shared-intel -qopt-streaming-stores always -march=native -ffast-math

# create main executable
qc: color_tensor.o gamma_container.o main.o read_prop.o \
	run_baryon_2pt.o run_dibaryon_2pt.o run_meson_2pt.o \
	run_nnpp_3pt.o run_sigma_3pt.o \
	run_4pt.o
	$(CC) $(CFLAGS) *.o

%.o : %.C
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f *.o a.out
