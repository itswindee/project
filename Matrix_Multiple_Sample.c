#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define N 4096
#define FactorIntToDouble 1.1; 

double firstMatrix [N] [N] = {0.0};
double secondMatrix [N] [N] = {0.0};
double matrixMultiResult [N] [N] = {0.0};


void matrixMulti()
{
    for(int row = 0 ; row < N ; row++){
        for(int col = 0; col < N ; col++){
            double resultValue = 0;
            for(int transNumber = 0 ; transNumber < N ; transNumber++) {
                resultValue += firstMatrix [row] [transNumber] * secondMatrix [transNumber] [col] ;
            }

            matrixMultiResult [row] [col] = resultValue;
        }
    }
}


void matrixInit()
{
    for(int row = 0 ; row < N ; row++ ) {
        for(int col = 0 ; col < N ;col++){
            srand(row+col);
            firstMatrix [row] [col] = ( rand() % 10 ) * FactorIntToDouble;
            secondMatrix [row] [col] = ( rand() % 10 ) * FactorIntToDouble;
        }
    }
}

void matrixMultiSequential() {
    for(int row = 0 ; row < N ; row++){
        for(int col = 0; col < N ; col++){
            double resultValue = 0;
            for(int transNumber = 0 ; transNumber < N ; transNumber++) {
                resultValue += firstMatrix[row][transNumber] * secondMatrix[transNumber][col];
            }
            matrixMultiResult[row][col] = resultValue;
        }
    }
}




int main() {
    // Initialize matrices
    matrixInit();

    // Sequential execution time
    clock_t t1 = clock();
    matrixMultiSequential(); // No OpenMP
    clock_t t2 = clock();
    double sequentialTime = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Sequential Time: %f seconds\n", sequentialTime);

    // Parallel execution time with OpenMP
    double t3 = omp_get_wtime();
    matrixMulti(); // With OpenMP
    double t4 = omp_get_wtime();
    double parallelTime = t4 - t3;
    printf("Parallel Time: %f seconds\n", parallelTime);

    // Speedup calculation
    double speedup = sequentialTime / parallelTime;
    printf("Speedup: %f\n", speedup);

    return 0;
}

