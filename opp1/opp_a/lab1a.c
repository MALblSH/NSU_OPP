#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define SIZE 200
#define PI 3.14159265358979323846
#define EPS 1e-4
#define TAO 1e-3

void MultiplyScalarAndVector(double *vector, double *resVector) {
    for (int i = 0; i < SIZE; ++i) {
        resVector[i] = vector[i] * TAO;
    }
}

void MinusVectors(double *vector1, double *vector2, double *resVector) {
    for (int i = 0; i < SIZE; ++i) {
        resVector[i] = vector1[i] - vector2[i];
    }
}

void multiplyMatrixAndVector(double *partA, double *x, double *resVector, int *sendcounts, int *displs, int rank) {
    double *partAx = (double*)malloc(sendcounts[rank] * sizeof(double));
    for (int i = 0; i < sendcounts[rank]; ++i) {
        double sum = 0.0;
        for (int j = 0; j < SIZE; ++j) {
            sum += partA[i * SIZE + j] * x[j];
        }
        partAx[i] = sum;
    }
    MPI_Allgatherv(partAx, sendcounts[rank], MPI_DOUBLE, resVector, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    free(partAx);
}

double GetNorm(double *vector) {
    double sum = 0;
    for (int i = 0; i < SIZE; ++i) {
        sum += vector[i] * vector[i];
    }
    return sqrt(sum);
}

int CheckCriteria(double *vector1, double *vector2){
    if ((GetNorm(vector1) / GetNorm(vector2)) < EPS)
        return 1;
    return 0;
}

void SetValues(double *partA, double *u, double *x, double *b, int *sendcounts, int *displs, int rank) {
    for (int i = 0; i < SIZE; ++i) {
        u[i] = sin(2 * PI * i / SIZE);
        x[i] = 0.0;
    }
    
    for (int i = 0; i < sendcounts[rank]; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (displs[rank] + i == j) 
                partA[i * SIZE + j] = 2.0;
            else 
                partA[i * SIZE + j] = 1.0;
        }
    }
    multiplyMatrixAndVector(partA, u, b, sendcounts, displs, rank);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int capacity = SIZE / size;
    int remainder = SIZE % size;
    
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i < remainder) ? (capacity + 1) : capacity;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    double *u = malloc(SIZE * sizeof(double));
    double *x = (double*)malloc(SIZE * sizeof(double));
    double *b = (double*)malloc(SIZE * sizeof(double));
    double *tempVector = (double*)malloc(SIZE * sizeof(double));
    double *Ax = (double*)malloc(SIZE * sizeof(double));
    double *partA = (double*)malloc(sendcounts[rank] * SIZE * sizeof(double));
    double *tempAx = (double*)malloc(sendcounts[rank] * sizeof(double));
    SetValues(partA, u, x, b, sendcounts, displs, rank);

    double t_start = MPI_Wtime();

    int state = 1;
    while (state) {
        multiplyMatrixAndVector(partA, x, Ax, sendcounts, displs, rank);
        MinusVectors(Ax, b, tempVector);
        if (rank == 0){
            if (CheckCriteria(tempVector, b)){
                state = 0;
            }
        }
        MPI_Bcast(&state, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MultiplyScalarAndVector(tempVector, tempVector);
        MinusVectors(x, tempVector, x);
    }
    double t_end = MPI_Wtime();

    if (rank == 0) {
        printf("%lf sec passed.\n", t_end - t_start);
        double maxDifference = 0;
        for (int i = 0; i < SIZE; i++) {
            double difference = fabs(u[i] - x[i]);
            if (difference > maxDifference)
                maxDifference = difference;
        }
        printf("max difference: %lf\n", maxDifference);

        
    }
    free(u);
    free(Ax);
    free(tempVector);
    free(x);
    free(b);
    free(partA);
    
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}