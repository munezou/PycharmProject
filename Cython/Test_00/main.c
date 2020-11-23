#include "Python.h"
#include "cybridge.h"
#include <stdio.h>


int main() {
    printf("[main.c] start\n");

    int n_samples = 200;
    double accuracy;

    printf("[main.c] initialize python interpreter\n");
    Py_Initialize();

    printf("[main.c] initialize libcybridge\n");
    PyRun_SimpleString("import sys\nsys.path.insert(0, '')");
    initlibcybridge();

    accuracy = calc_accuracy(n_samples);
    Py_Finalize();

    printf("[main.c] accuracy: %f\n", accuracy);
    return 0;
}