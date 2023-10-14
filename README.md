# Neutrinoless Double-Beta Decay

Authors: William Detmold, Zhenghao Fu, Anthony Grebe, David Murphy, and Patrick Oare

Publications: TBA

## Summary

This code computes nuclear matrix elements for neutrinoless double-beta decay from lattice QCD.  In particular, it computes both long-distance interactions (from a light intermediate neutrino propagating between the two weak current insertions) and short-distance interactions (from a heavy intermediate neutrino that has been integrated out at the scale of lattice QCD).  These two contributions give rise to 4-point and 3-point correlation functions, respectively, based on the number of lattice operators used in the calculation.

Throughout, the code uses a wall source and a sparse grid of point sinks.  Propagators are computed from both the source and sink (and sink propagators are reversed using $\gamma_5$ -hermiticity), and these propagators are contracted at the operator insertion point(s) to form the 3-point and 4-point correlation functions.  The sets of contractions for both the single-hadron $\Sigma^- \rightarrow \Sigma^+$ transition and the multi-hadron $nn \rightarrow pp$ transition are included in this codebase.

Quark inversions are performed using the QPhiX inverter library.  All hadron interpolators used in this project project all quarks to positive parity, so only six inversions are needed per propagator.

## Installation

### Prerequisites

Hardware Requirements:
- Processor with AVX-512 support (present on many modern Intel (R) processors used in high-performance computing)
- At least 128 GB RAM (for $32^3 \times 48$ lattices)

Software Dependencies:
- [FFTW3](https://www.fftw.org/download.html)
- Intel Math Kernel Library
- C++ compiler (Intel compiler recommended, as this facilitates linking with MKL)

For computing propagators rather than reading from disk, the following are also required:
- [QDP++](https://github.com/usqcd-software/qdpxx)
- [CMake](https://cmake.org/) (required to build QPhiX)
- [QPhiX](https://github.com/JeffersonLab/qphix)
- [qphix-wrapper](https://github.com/agrebe/qphix-wrapper) (a wrapper around QPhiX that only exposes the inverters needed for this project)

### Instructions

Install the required software dependencies.  When building QDP++, it is recommended to create a build without MPI (`--enable-parallel-arch=scalar`).  This double-beta code does not include MPI support, and disabling MPI support for QDP++ simplifies the list of dependencies (e.g. QMP is not required).

Then edit the Makefile to include the paths to QDP++, QPhiX, qphix-wrapper, and FFTW and run `make`.  The file will generate the executable `a.out`.  The number of threads used is set by the environment variable `OMP_NUM_THREADS`.

### Code Generator
The $nn \rightarrow pp$ transition contains $(4!)^2 / 2$ Wick contractions, which would be tedious and error-prone to write out by hand.  These have been auto-generated and included in the file `run_nnpp.inc`.  It is not necessary to regenerate this file in order to run the code, but it can be created using the script `generator.sh` in the `nnpp-generator` subdirectory.  This calls a Python script `contractions.py` to compute the relevant Wick contractions and then a Bash script `process-traces.sh` to perform text replacement operations using `sed`.

## Performance
This code has been tested on the Stampede2 cluster at the Texas Advanced Computing Center (TACC), on a node with 80 CPU cores (2 sockets of 40 cores each) using Intel's Ice Lake architecture.  Runtime on a single $32^3 \times 48$ gauge configuration is about 8 hours, of which the majority (5 hours) is spent computing propagators, and the remaining 3 hours are divided roughly evenly between 3-point and 4-point contractions.

On systems such as Stampede2 with cores divided across multiple sockets and ample memory, performance can be improved by running multiple copies of the executable in parallel.  With two copies, the runtime increased slightly to about 10 hours, but the amortized cost per configuration dropped to 5 node-hours.

Several code optimizations made to improve performance include the following:
- The use of the Fast Fourier Transform to perform the convolution of quark and neutrino propagators used for the 4-point functions
- Algebraic manipulation to precompute a volume-averaged 4-quark tensor to be used in 3-point and 4-point functions.  This decouples the $O(V)$ volume sum from the $(4!)^2 / 2 \times 6^4 = 3.7\times 10^5$ nuclear spin and color contractions required by the multi-hadron interpolating operators.
- Use of heavily optimized math libraries, including the Fastest Fourier Transform in the West and the Intel Math Kernel Library's BLAS implementation, as well as the heavily optimized QPhiX library for inverting the Dirac operator
- Positive parity projections of all quarks, speeding computation of the 4-quark tensor by a factor of $2^4$ and propagator computation time by a factor of 2
- Writing the $2 \times 2$ complex matrix multiplication kernel in terms of AVX-512 instructions
- Calling QPhiX directly to avoid I/O costs from writing propagators to disk
- Reducing the memory footprint to avoid overhead from network bandwidth and less-than-perfect strong scaling
