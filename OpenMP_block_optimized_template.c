#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define N 4096
#define BLOCK_SIZE 512 
#define FactorIntToDouble 1.1

double firstMatrix[N][N] = {0.0};
double secondMatrix[N][N] = {0.0};
double matrixMultiResult[N][N] = {0.0};

// Initialize matrices with random values
void matrixInit() {
    for(int row = 0 ; row < N ; row++) {
        for(int col = 0 ; col < N ; col++) {
            srand(row + col);
            firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
        }
    }
}

// Sequential block matrix multiplication for small blocks
void smallMatrixMult(int upperOfRow, int bottomOfRow, int leftOfCol, int rightOfCol, int transLeft, int transRight) {
    for(int i = upperOfRow; i <= bottomOfRow; i++) {
        for(int j = leftOfCol; j <= rightOfCol; j++) {
            double sum = 0.0;
            for(int k = transLeft; k <= transRight; k++) {
                sum += firstMatrix[i][k] * secondMatrix[k][j];
            }
            matrixMultiResult[i][j] += sum;
        }
    }
}

// Parallelized block matrix multiplication for small blocks (for parallel execution)
void smallMatrixMultParallel(int upperOfRow, int bottomOfRow, int leftOfCol, int rightOfCol, int transLeft, int transRight) {
    #pragma omp parallel for collapse(2)
    for(int i = upperOfRow; i <= bottomOfRow; i++) {
        for(int j = leftOfCol; j <= rightOfCol; j++) {
            double sum = 0.0;
            for(int k = transLeft; k <= transRight; k++) {
                sum += firstMatrix[i][k] * secondMatrix[k][j];
            }
            matrixMultiResult[i][j] += sum;
        }
    }
}

// Block matrix multiplication with task-based division
void matrixMulti(int upperOfRow, int bottomOfRow, int leftOfCol, int rightOfCol, int transLeft, int transRight) {
    if ((bottomOfRow - upperOfRow) < BLOCK_SIZE) {
        smallMatrixMultParallel(upperOfRow, bottomOfRow, leftOfCol, rightOfCol, transLeft, transRight);  // Use parallel function
    } else {
        #pragma omp task
        matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, leftOfCol, (leftOfCol + rightOfCol) / 2, transLeft, (transLeft + transRight) / 2);
        #pragma omp task
        matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, leftOfCol, (leftOfCol + rightOfCol) / 2, (transLeft + transRight) / 2 + 1, transRight);
        
        #pragma omp task
        matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, transLeft, (transLeft + transRight) / 2);
        #pragma omp task
        matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, (transLeft + transRight) / 2 + 1, transRight);
        
        #pragma omp task
        matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, leftOfCol, (leftOfCol + rightOfCol) / 2, transLeft, (transLeft + transRight) / 2);
        #pragma omp task
        matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, leftOfCol, (leftOfCol + rightOfCol) / 2, (transLeft + transRight) / 2 + 1, transRight);
        
        #pragma omp task
        matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, transLeft, (transLeft + transRight) / 2);
        #pragma omp task
        matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, (transLeft + transRight) / 2 + 1, transRight);
        
        #pragma omp taskwait
    }
}

int main() {
    matrixInit();

    // Sequential block-optimized execution time
    double start_time = omp_get_wtime();
    matrixMulti(0, N-1, 0, N-1, 0, N-1);
    double sequential_time = omp_get_wtime() - start_time;
    printf("Sequential Block-Optimized Time: %f seconds\n", sequential_time);

    // Reset the result matrix to zero
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixMultiResult[i][j] = 0.0;
        }
    }

    // Parallel block-optimized execution time
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        matrixMulti(0, N-1, 0, N-1, 0, N-1);
    }
    double parallel_time = omp_get_wtime() - start_time;
    printf("Parallel Block-Optimized Time: %f seconds\n", parallel_time);

    // Calculate speedup
    double speedup = sequential_time / parallel_time;
    printf("Speedup: %f\n", speedup);

    return 0;
}
