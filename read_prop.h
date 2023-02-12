#include <fstream>
#include <iostream>
#include <string.h>
#include "spin_mat.h"
#include "gamma_container.h"

void read_prop(char * filename, SpinMat * prop, int nt, int nx);
void rescale_prop(SpinMat * prop, int nt, int nx, double factor);
void reverse_prop(SpinMat * prop, int nt, int nx);
void project_prop(SpinMat * prop, int nt, int nx, int positive);
