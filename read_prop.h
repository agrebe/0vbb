#include <fstream>
#include <iostream>
#include "spin_mat.h"

void read_prop(char * filename, SpinMat * prop, int nt, int nx);
void rescale_prop(SpinMat * prop, int nt, int nx, double factor);
