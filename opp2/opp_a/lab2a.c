#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
# define PI		3.14159265358979323846

#define SIZE 2000
#define EPS 1e-4
#define TAO 1e-3

void MultiplyScalarVector(const double *vector, double scalar, double *resVector){
#pragma omp parallel for default(none) shared(vector, scalar, resVector) //schedule(dynamic, 5)
    for (int i = 0; i < SIZE; i++){
        resVector[i] = vector[i] * scalar;
    }
}

void MinusVector(const double *vector1, const double *vector2, double *resVector){
#pragma omp parallel for default(none) shared(vector1, vector2, resVector) //schedule(dynamic, 5)
    for (int i = 0; i < SIZE; i++){
        resVector[i] = vector1[i] - vector2[i];
    }
}

void MultiplyMatrixVector(const double *matrix, const double *vector, double *resVector){
#pragma omp parallel for default(none) shared(matrix, vector, resVector) //schedule(dynamic, 5)
    for (int i = 0; i < SIZE; ++i){
        double sum = 0;
        for (int j = 0; j < SIZE; ++j){
            sum += matrix[i * SIZE + j] * vector[j];
        }
        resVector[i] = sum;
    }
}

double GetNorm(const double *vector){
    double sum = 0;
#pragma omp parallel for default(none) shared(vector) reduction(+:sum) //schedule(dynamic, 5)
    for (int i = 0; i < SIZE; ++i){
        sum += vector[i] * vector[i];
    }
    sum = sqrt(sum);
    return sum;
}


int CheckCriteria(double *vector1, double *vector2, double eps){
    if ((GetNorm(vector1) / GetNorm(vector2)) < eps)
        return 1;
    return 0;
}

void Initialization(double *A, double *u, double *x, double *b)
{
    for (int i = 0; i < SIZE; ++i){
        u[i] = sin(2 * PI * i / SIZE);
        x[i] = 0;
        for (int j = 0; j < SIZE; ++j){
            if (i == j)
                A[i * SIZE + j] = 2.0;
            else
                A[i * SIZE + j] = 1.0;
        }
    }

    MultiplyMatrixVector(A, u, b);
}

int main(){
    int num_threads = 8;
    omp_set_num_threads(num_threads);
    double *x = (double *)malloc(SIZE * sizeof(double));
    double *b = (double *)malloc(SIZE * sizeof(double));
    double *A = (double *)malloc(SIZE * SIZE * sizeof(double));
    double *Ax = (double *)malloc(SIZE * sizeof(double));
    double *u = (double *)malloc(SIZE * sizeof(double));
    double *tempVector = (double *)malloc(SIZE * sizeof(double));



    Initialization(A, u, x, b);

    double startTime, endTime;
    startTime = omp_get_wtime();

    int state = 1;
    while (state){
        MultiplyMatrixVector(A, x, Ax);
        MinusVector(Ax, b, tempVector);
        if (CheckCriteria(tempVector, x, EPS))
            state = 0;
        MultiplyScalarVector(tempVector, TAO, tempVector);
        MinusVector(x, tempVector, x);

    }

    endTime = omp_get_wtime();

    printf("%lf\n", endTime - startTime);


    free(u);
    free(A);
    free(Ax);
    free(tempVector);
    free(x);
    free(b);

    return 0;
}