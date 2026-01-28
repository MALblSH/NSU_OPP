#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PI 3.14159265358979323846
#define EPS 1e-4
#define TAO 1e-3
#define SIZE 200

void multiplyMatrixAndVector(double *partA, int localSIZE, double *partX, double *partAx, int *counts, int *displs, int size, int rank) {
    for (int i = 0; i < localSIZE; i++) {
        partAx[i] = 0.0;
    }
    int maxLocalN = ((SIZE % size) == 0) ? (SIZE / size) :  (SIZE /size + 1);
    double *sendBuf = (double*)malloc(maxLocalN * sizeof(double));
    double *recvBuf = (double*)malloc(maxLocalN * sizeof(double));
    int currentSize = counts[rank];
    memcpy(sendBuf, partX, currentSize * sizeof(double));
    int currentOwner = rank;
    for (int step = 0; step < size; step++) {
        int blockSize = counts[currentOwner];
        int blockDisp = displs[currentOwner];
        for (int i = 0; i < localSIZE; i++) {
            double part = 0.0;
            for (int k = 0; k < blockSize; k++) {
                int globalIndex = blockDisp + k;
                part += partA[i*SIZE + globalIndex] * sendBuf[k];
            }
            partAx[i] += part;
        }
        int leftNeighbour = (rank - 1 + size) % size;
        int rightNeighbour = (rank + 1) % size;
        int sendCurSizeAndCurOwner[2] = { currentSize, currentOwner };
        int recvPrevSizeAndPrevOwner[2];
        MPI_Sendrecv(sendCurSizeAndCurOwner, 2, MPI_INT, rightNeighbour, 0, recvPrevSizeAndPrevOwner, 2, MPI_INT, leftNeighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int nextSize = recvPrevSizeAndPrevOwner[0];
        int nextOwner = recvPrevSizeAndPrevOwner[1];
        MPI_Sendrecv(sendBuf, currentSize, MPI_DOUBLE, rightNeighbour, 1, recvBuf, nextSize, MPI_DOUBLE, leftNeighbour, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        currentSize = nextSize;
        currentOwner = nextOwner;
        memcpy(sendBuf, recvBuf, currentSize * sizeof(double));
    }
    free(sendBuf);
    free(recvBuf);
}

void SetValues(double *partA, double *u, double *partX, double *partB, int *sendcounts, int *displs, int rank) {
    for (int i = 0; i < SIZE; ++i) {
        u[i] = sin(2 * PI * i / SIZE);
        
    }
    
    for (int i = 0; i < sendcounts[rank]; ++i) {
        partX[i] = 0.0;
        for (int j = 0; j < SIZE; ++j) {
            if (displs[rank] + i == j) 
                partA[i * SIZE + j] = 2.0;
            else 
                partA[i * SIZE + j] = 1.0;
        }
    }
    for (int i = 0; i < sendcounts[rank]; i++) {
        double sum = 0.0;
        for (int j = 0; j < SIZE; j++) {
            sum += partA[i*SIZE + j] * u[j];
        }
        partB[i] = sum;
    }
}

void MinusVectors(double *vector1, double *vector2, double *resVector, int localSIZE) {
    for (int i = 0; i < localSIZE; ++i) {
        resVector[i] = vector1[i] - vector2[i];
    }
}

void MultiplyScalarAndVector(double *vector, double *resVector, int localSIZE) {
    for (int i = 0; i < localSIZE; ++i) {
        resVector[i] = vector[i] * TAO;
    }
}

double GetNorm(double *vector, int localSIZE){
    double localSum = 0.0;
    for (int i = 0; i < localSIZE; ++i)
    {
        localSum += vector[i] * vector[i];
    }
    double globalSum = 0.0;
    MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(globalSum);
}

int CheckCriteria(double *vector1, double *vector2, int localSIZE){
    if ((GetNorm(vector1, localSIZE) / GetNorm(vector2, localSIZE)) < EPS)
        return 1;
    return 0;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int capacity = SIZE / size;
    int remainder = SIZE % size;
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i < remainder) ? (capacity + 1) : capacity;
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    double* u = (double*)malloc(SIZE * sizeof(double));
    double *partA = (double*)malloc(sendcounts[rank] * SIZE * sizeof(double));
    double *partX = (double*)malloc(sendcounts[rank] * sizeof(double));
    double *partB = (double*)malloc(sendcounts[rank] * sizeof(double));
    double *partAx = (double*)malloc(sendcounts[rank] * sizeof(double));
    double *temp_local = (double*)malloc(sendcounts[rank] * sizeof(double));

    
    

    SetValues(partA, u, partX, partB, sendcounts, displs, rank);
    double t_start = MPI_Wtime();
    while (1) {
        multiplyMatrixAndVector(partA, sendcounts[rank], partX, partAx, sendcounts, displs, size, rank);
        MinusVectors(partAx, partB, temp_local, sendcounts[rank]);
        if (CheckCriteria(temp_local, partB, sendcounts[rank])){
            break;
        }
        MultiplyScalarAndVector(temp_local, temp_local, sendcounts[rank]);
        MinusVectors(partX, temp_local, partX, sendcounts[rank]);
    }
    double t_end = MPI_Wtime();
    double *x = (double*)malloc(SIZE * sizeof(double));
    MPI_Allgatherv(partX, sendcounts[rank], MPI_DOUBLE, x, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%lf sec passed.\n", t_end - t_start);
        double maxDifference = 0;
        for (int i = 0; i < SIZE; i++){
            double difference = fabs(u[i] - x[i]);
            if (difference > maxDifference)
                maxDifference = difference;
        }
        printf("max difference: %lf\n", maxDifference);

    }
    free(partA);
    free(partX);
    free(partB);
    free(partAx);
    free(temp_local);
    free(sendcounts);
    free(displs);
    free(u);
    MPI_Finalize();
    return 0;
}