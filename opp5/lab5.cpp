#include <stdio.h>
#include <pthread.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#define N_TASKS 512
#define REQUEST_TAG 0
#define TASK_TAG 1

typedef struct Task {
    int taskNumber;
    int difficulty;
    char completed;
    char taken;
} Task;

pthread_mutex_t mutex;
Task* task_list;

int total_completed_weight = 0;

int calculate_task(Task* task) {
    pthread_mutex_lock(&mutex);
    if (task->completed == 0) {
        task->completed = 1;
        pthread_mutex_unlock(&mutex);
        usleep(task->difficulty);
        return task->difficulty;
    } else {
        pthread_mutex_unlock(&mutex);
        return 0;
    }
}

int request_task(int request_rank) {
    int task_id = -1;
    int request = 1;
    MPI_Send(&request, 1, MPI_INT, request_rank, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&task_id, 1, MPI_INT, request_rank, TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return task_id;
}

void* worker_thread(void* args) {
    int rank = *((int*)args);
    int size = *((int*)args + 1);

    while (1) {
        Task* task_to_do = NULL;

        for (int i = rank * N_TASKS / size; i < (rank + 1) * N_TASKS / size; ++i) {
            pthread_mutex_lock(&mutex);
            if (!task_list[i].taken) {
                task_to_do = &task_list[i];
                task_list[i].taken = 1;
                pthread_mutex_unlock(&mutex);
                break;
            }
            pthread_mutex_unlock(&mutex);
        }

        if (task_to_do == NULL) {
            for (int dist = 1; dist <= size / 2; ++dist) {
                int neighbor_ranks[2] = {(rank + dist) % size,(rank - dist + size) % size};
                for (int j = 0; j < 2; ++j) {
                    int neighbor = neighbor_ranks[j];
                    if (neighbor == rank) continue;

                    int task_id = request_task(neighbor);
                    if (task_id != -1) {
                        task_to_do = &task_list[task_id];
                        break;
                    }
                }
                if (task_to_do != NULL) break;
            }

            if (task_to_do == NULL) {
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Send(NULL, 0, MPI_INT, rank, REQUEST_TAG, MPI_COMM_WORLD);
                break;
            }
        }

        int weight = calculate_task(task_to_do);
        pthread_mutex_lock(&mutex);
        total_completed_weight += weight;
        pthread_mutex_unlock(&mutex);
    }

    return NULL;
}

void* server_thread(void* args) {
    int rank = *((int*)args);
    int size = *((int*)args + 1);
    int request;
    MPI_Status status;
    
    while (1) {
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_SOURCE == rank) {
            break;
        }

        int task_id = -1;
        for (int i = rank * N_TASKS / size; i <  rank * N_TASKS / size + N_TASKS / size; ++i) {
            pthread_mutex_lock(&mutex);
            if (task_list[i].taken == 0) {
                task_id = i;
                task_list[i].taken = 1;
                pthread_mutex_unlock(&mutex);
                break;
            }
            pthread_mutex_unlock(&mutex);
        }
        MPI_Send(&task_id, 1, MPI_INT, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        printf("Can't init thread\n");
        MPI_Finalize();
        return -1;
    }

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    pthread_attr_t attrs;
    pthread_mutex_init(&mutex, NULL);
    pthread_attr_init(&attrs);
    pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);

    task_list = (Task*)malloc(sizeof(Task) * N_TASKS);
    if (rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < N_TASKS; ++i) {
            task_list[i].taskNumber = i;
            //task_list[i].difficulty = 100 * 1000;
            task_list[i].difficulty = rand() % 100 * 1000;
            //task_list[i].difficulty = (i + 1) * 1000;
            //task_list[i].difficulty = (N_TASKS-i-1) * 1000;
            task_list[i].completed = 0;
            task_list[i].taken = 0;
        }
    }

    MPI_Bcast(task_list, N_TASKS * sizeof(Task), MPI_BYTE, 0, MPI_COMM_WORLD);

    int initial_weight = 0;
    int local_start = rank * N_TASKS / size;
    int local_end = local_start + N_TASKS / size;
    for (int i = local_start; i < local_end; ++i) {
        initial_weight += task_list[i].difficulty;
    }

    pthread_t threads[2];
    int* threadArgs = (int*)malloc(sizeof(int) * 2);
    threadArgs[0] = rank;
    threadArgs[1] = size;
    pthread_create(&threads[0], &attrs, worker_thread, (void*)threadArgs);
    pthread_create(&threads[1], &attrs, server_thread, (void*)threadArgs);

    double start_time = MPI_Wtime();

    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    double end_time = MPI_Wtime();

    int local_completed = 0;
    for (int i = 0; i < N_TASKS; ++i) {
        if (task_list[i].completed == 1) {
            local_completed++;
        }
    }

    int total_completed = 0;
    MPI_Reduce(&local_completed, &total_completed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    
    int* all_initial_weights = NULL;
    int* all_completed_weights = NULL;

    if (rank == 0) {
        all_initial_weights = (int*)malloc(sizeof(int) * size);
        all_completed_weights = (int*)malloc(sizeof(int) * size);
    }

    MPI_Gather(&initial_weight, 1, MPI_INT, all_initial_weights, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&total_completed_weight, 1, MPI_INT, all_completed_weights, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== Статистика по процессам ===\n");
        for (int i = 0; i < size; ++i) {
            printf("Процесс %d: изначальный вес = %d, выполнено = %d, чужая работа = %d\n", i, all_initial_weights[i], all_completed_weights[i], all_completed_weights[i] - all_initial_weights[i]);
        }

        printf("\nВыполнено задач: %d из %d\n", total_completed, N_TASKS);
        printf("Заняло времени: %lf сек\n", end_time - start_time);
        free(all_initial_weights);
        free(all_completed_weights);
    }

    free(threadArgs);
    free(task_list);

    pthread_mutex_destroy(&mutex);
    pthread_attr_destroy(&attrs);

    MPI_Finalize();
    return 0;
}