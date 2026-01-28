#include <iostream>
#include <vector>
#include <mpi.h>
#include <random>
#include <iomanip>
#include <cmath>
#include <cblas.h>

double getMaxDiffWithCorrectAnswer(const std::vector<double>& matrixA, const std::vector<double>& matrixB, const std::vector<double>& matrixC, int rowsA, int colsA, int colsB){
    std::vector<double> matrixD(rowsA * colsB, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, colsB, colsA, 1.0, matrixA.data(), colsA, matrixB.data(), colsB, 0.0, matrixD.data(), colsB);
    double maxDiff = 0.0;
    for (size_t i = 0; i < matrixD.size(); ++i) {
      double diff = std::abs(matrixD[i] - matrixC[i]);
      if (diff > maxDiff) {
        maxDiff = diff;
      }
    }
    return maxDiff;
  }
 

void populateMatrix(double* matrix, int rows, int cols) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(1, 100);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = distrib(gen);
        }
    }
}

void displayMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void computeMatrixProduct(const double* subA, const double* subB, double* subC, int subRows, int commonDim, int subCols) {
    for (int i = 0; i < subRows; ++i) {
    for (int k = 0; k < commonDim; ++k) {
        for (int j = 0; j < subCols; ++j) {
            subC[i * subCols + j] += subA[i * commonDim + k] * subB[k * subCols + j];
        }
    }
}
}

void initializeGrid(MPI_Comm& gridComm, int size, int* gridDims, int* gridCoords, MPI_Comm& rowComm, MPI_Comm& colComm, int& rowCoord, int& colCoord, int processRank) {
    int periods[2] = {0, 0};
    MPI_Dims_create(size, 2, gridDims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, gridDims, periods, 0, &gridComm);
    MPI_Cart_coords(gridComm, processRank, 2, gridCoords);
    rowCoord = gridCoords[0];
    colCoord = gridCoords[1];

    int remainDimsRow[2] = {false, true};
    int remainDimsCol[2] = {true, false};
    MPI_Cart_sub(gridComm, remainDimsRow, &rowComm);
    MPI_Cart_sub(gridComm, remainDimsCol, &colComm);
}

void distributeMatrixA(std::vector<double>& matrixA, std::vector<double>& subA, int rowsA, int colsA, int gridRows, int colCoord, MPI_Comm rowComm, MPI_Comm colComm) {
    int subRows = rowsA / gridRows;
    subA.resize(subRows * colsA);
    if (colCoord == 0) {
        MPI_Scatter(matrixA.data(), subRows * colsA, MPI_DOUBLE, subA.data(), subRows * colsA, MPI_DOUBLE, 0, colComm);
    }
    MPI_Bcast(subA.data(), subRows * colsA, MPI_DOUBLE, 0, rowComm);
}

void distributeMatrixB(std::vector<double>& matrixB, std::vector<double>& subB, int rowsB, int colsB, int gridCols, int rowCoord, MPI_Comm rowComm, MPI_Comm colComm) {
    int subCols = colsB / gridCols;
    subB.resize(rowsB * subCols);

    MPI_Datatype colType, resizedColType;
    MPI_Type_vector(rowsB, subCols, colsB, MPI_DOUBLE, &colType);
    MPI_Type_create_resized(colType, 0, subCols * sizeof(double), &resizedColType);
    MPI_Type_commit(&resizedColType);

    if (rowCoord == 0) {
        MPI_Scatter(matrixB.data(), 1, resizedColType, subB.data(), rowsB * subCols, MPI_DOUBLE, 0, rowComm);
    }
    MPI_Bcast(subB.data(), rowsB * subCols, MPI_DOUBLE, 0, colComm);

    MPI_Type_free(&colType);
    MPI_Type_free(&resizedColType);
}

void gatherMatrixC(std::vector<double>& matrixC, std::vector<double>& subC, int rowsC, int colsC, int gridRows, int gridCols, MPI_Comm gridComm) {
    int subRows = rowsC / gridRows;
    int subCols = colsC / gridCols;

    MPI_Datatype blockType, resizedBlockType;
    MPI_Type_vector(subRows, subCols, colsC, MPI_DOUBLE, &blockType);
    MPI_Type_create_resized(blockType, 0, subCols * sizeof(double), &resizedBlockType);
    MPI_Type_commit(&resizedBlockType);

    std::vector<int> recvCounts(gridRows * gridCols, 1);
    std::vector<int> displacements(gridRows * gridCols);
    for (int i = 0; i < gridRows; ++i) {
        for (int j = 0; j < gridCols; ++j) {
            int globalRank;
            int coords[2] = {i, j};
            MPI_Cart_rank(gridComm, coords, &globalRank);
            displacements[globalRank] = i * gridCols * subRows + j;
        }
    }

    MPI_Gatherv(subC.data(), subRows * subCols, MPI_DOUBLE, matrixC.data(), recvCounts.data(), displacements.data(), resizedBlockType, 0, gridComm);

    MPI_Type_free(&blockType);
    MPI_Type_free(&resizedBlockType);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int processRank, totalProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);

    int gridDims[2] = {0, 0};
    int gridCoords[2];
    MPI_Comm gridComm, rowComm, colComm;
    int rowCoord, colCoord;

    initializeGrid(gridComm, totalProcesses, gridDims, gridCoords, rowComm, colComm, rowCoord, colCoord, processRank);

    int gridRows = gridDims[0], gridCols = gridDims[1];
    int rowsA = 4096, colsA = 2048, colsB = 4096;

    std::vector<double> matrixA, matrixB, matrixC;
    if (processRank == 0) {
        matrixA.resize(rowsA * colsA);
        matrixB.resize(colsA * colsB);
        matrixC.resize(rowsA * colsB);
        populateMatrix(matrixA.data(), rowsA, colsA);
        populateMatrix(matrixB.data(), colsA, colsB);
        /*std::cout << "Matrix A:" << std::endl;
        displayMatrix(matrixA.data(), rowsA, colsA);
        std::cout << "Matrix B:" << std::endl;
        displayMatrix(matrixB.data(), colsA, colsB);*/
    }
    double startTime = MPI_Wtime();
    
    std::vector<double> subA, subB, subC;
    distributeMatrixA(matrixA, subA, rowsA, colsA, gridRows, colCoord, rowComm, colComm);
    distributeMatrixB(matrixB, subB, colsA, colsB, gridCols, rowCoord, rowComm, colComm);

    subC.resize((rowsA / gridRows) * (colsB / gridCols), 0.0);
    computeMatrixProduct(subA.data(), subB.data(), subC.data(), rowsA / gridRows, colsA, colsB / gridCols);

    gatherMatrixC(matrixC, subC, rowsA, colsB, gridRows, gridCols, gridComm);

    double endTime = MPI_Wtime();
    double duration = endTime - startTime;

    if (processRank == 0) {
        double maxDiff = getMaxDiffWithCorrectAnswer(matrixA, matrixB, matrixC, rowsA, colsA, colsB);     
        std::cout << std::fixed << std::setprecision(7);
        std::cout << "Max difference: " << maxDiff << std::endl;
        //std::cout << "Matrix C:" << std::endl;
        //displayMatrix(matrixC.data(), rowsA, colsB);
        std::cout << duration << std::endl;
    }

    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&gridComm);
    MPI_Finalize();
}