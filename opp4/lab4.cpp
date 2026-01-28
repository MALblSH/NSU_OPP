#include <stdio.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <algorithm>
#include <mpi.h>

#define Nx 10
#define Ny 10
#define Nz 128
#define x0 -1
#define y0 -1
#define z0 -1
#define Dx 2
#define Dy 2
#define Dz 2
#define EPS 1e-8
#define A 1e5

double countBoundary(int x, int y, int z) {
    static const double step_x = (double) Dx / (Nx - 1);
    static const double step_y = (double) Dy / (Ny - 1);
    static const double step_z = (double) Dz / (Nz - 1);
    return (x0 + x * step_x) * (x0 + x * step_x) +
           (y0 + y * step_y) * (y0 + y * step_y) +
           (z0 + z * step_z) * (z0 + z * step_z);
}

void setupGrid(double* block, int slice_count, int rank, int total_procs) {
    for (int z = 0; z < slice_count; z++) {
        for (int y = 0; y < Ny; y++) {
            for (int x = 0; x < Nx; x++) {
                int coord = z * Nx * Ny + Nx * y + x;
                if ((rank == 0 && z == 0) || (rank == total_procs - 1 && z == slice_count - 1) ||
                    y == 0 || y == Ny - 1 || x == 0 || x == Nx - 1) {
                    int global_z = Nz / total_procs * rank + z;
                    block[coord] = countBoundary(x, y, global_z);
                } else {
                    block[coord] = 0.0;
                }
            }
        }
    }
}

double countNewValue(double const* block, int x, int y, int z) {
    static const double sqr_step_x = ((double) Dx / (Nx - 1)) * ((double) Dx / (Nx - 1));
    static const double sqr_step_y = ((double) Dy / (Ny - 1)) * ((double) Dy / (Ny - 1));
    static const double sqr_step_z = ((double) Dz / (Nz - 1)) * ((double) Dz / (Nz - 1));
    static const double denom = 1.0 / (2.0 / sqr_step_x + 2.0 / sqr_step_y + 2.0 / sqr_step_z + A);
    double sum_neighbors = (block[(z - 1) * Nx * Ny + y * Nx + x] + 
                            block[(z + 1) * Nx * Ny + y * Nx + x]) / sqr_step_z +
                           (block[z * Nx * Ny + (y - 1) * Nx + x] + 
                            block[z * Nx * Ny + (y + 1) * Nx + x]) / sqr_step_y +
                           (block[z * Nx * Ny + y * Nx + (x - 1)] + 
                            block[z * Nx * Ny + y * Nx + (x + 1)]) / sqr_step_x -
                           (6.0 - A * block[z * Nx * Ny + y * Nx + x]);
    return denom * sum_neighbors;
}

static double processSlice(double const* current_block, int slice_z, double* new_block) {
    double max_change = 0.0;
    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
            int coord = slice_z * Nx * Ny + y * Nx + x;
            if (y == 0 || y == Ny - 1 || x == 0 || x == Nx - 1) {
                new_block[coord] = current_block[coord];
                continue;
            }
            new_block[coord] = countNewValue(current_block, x, y, slice_z);
            double change = std::abs(current_block[coord] - new_block[coord]);
            max_change = change > max_change ? change : max_change;
        }
    }
    return max_change;
}

static double countCore(double const* current_block, int slice_count, int rank,
                        int total_procs, double* new_block) {
    if (rank == 0) {
        std::copy(current_block, current_block + Nx * Ny, new_block);
    }
    if (rank == total_procs - 1) {
        std::copy(current_block + (slice_count - 1) * Nx * Ny,
                  current_block + slice_count * Nx * Ny,
                  new_block + (slice_count - 1) * Nx * Ny);
    }
    int start_slice = 2;
    int end_slice = slice_count - 2;
    double max_change = 0.0;
    for (int slice_z = start_slice; slice_z < end_slice; slice_z++) {
        double slice_change = processSlice(current_block, slice_z, new_block);
        max_change = slice_change > max_change ? slice_change : max_change;
    }
    return max_change;
}

int main() {
    MPI_Init(NULL, NULL);
    int rank, total_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);
    int slice_count;
    if (total_procs == 1) {
        slice_count = Nz;
    } else {
        slice_count = (rank == 0 || rank == total_procs - 1) ? Nz / total_procs + 1 : Nz / total_procs + 2;
    }
    std::vector<double> grid_values(Nx * Ny * slice_count);
    std::vector<double> next_grid(Nx * Ny * slice_count);
    setupGrid(grid_values.data(), slice_count, rank, total_procs);
    int iteration = 0;
    double global_change = EPS;
    
    double startTime = MPI_Wtime();

    while (global_change >= EPS) {
        iteration++;
        MPI_Request comm_requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (rank != 0) {
            MPI_Isend(grid_values.data() + 1 * Nx * Ny, Nx * Ny, MPI_DOUBLE, 
                      rank - 1, 0, MPI_COMM_WORLD, &comm_requests[0]);
            MPI_Irecv(grid_values.data(), Nx * Ny, MPI_DOUBLE, 
                      rank - 1, 0, MPI_COMM_WORLD, &comm_requests[1]);
        }
        if (rank != total_procs - 1) {
            MPI_Isend(grid_values.data() + (slice_count - 2) * Nx * Ny, Nx * Ny, MPI_DOUBLE,
                      rank + 1, 0, MPI_COMM_WORLD, &comm_requests[2]);
            MPI_Irecv(grid_values.data() + (slice_count - 1) * Nx * Ny, Nx * Ny, MPI_DOUBLE,
                      rank + 1, 0, MPI_COMM_WORLD, &comm_requests[3]);
        }
        double local_change = countCore(grid_values.data(), slice_count, rank, total_procs,
                                       next_grid.data());

        MPI_Waitall(4, comm_requests, MPI_STATUSES_IGNORE);

        if (slice_count >= 3) {
            double slice_change = processSlice(grid_values.data(), slice_count - 2, next_grid.data());
            local_change = slice_change > local_change ? slice_change : local_change;
            slice_change = processSlice(grid_values.data(), 1, next_grid.data());
            local_change = slice_change > local_change ? slice_change : local_change;
        }
        grid_values.swap(next_grid);
        if (iteration % 1000 == 0) {
            MPI_Allreduce(&local_change, &global_change, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) {
                std::cout << "Итерация " << iteration << ", global_change = " 
                          << std::scientific << global_change << std::endl;
            }
        }
    }

    double endTime = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Итераций: " << iteration << std::endl;
        std::cout << "Время выполнения: " << std::fixed << std::setprecision(2) 
                  << endTime - startTime << " сек" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
