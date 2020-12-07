#include<stdio.h>

// 行列の中身を変えるだけの関数
void cfunc(int N, double *temp){
    int i;
    for(i=0; i<N; i++){
        temp[i] = i * 2.0;
    }
}
