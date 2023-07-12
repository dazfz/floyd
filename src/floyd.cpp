#include <iostream>
#include <vector>
#include <immintrin.h> // SIMD
#include <omp.h>       // OpenMP

using namespace std;

typedef vector<float> vf;
typedef vector<vector<float>> vff;
typedef vector<double> vd;
typedef vector<vector<double>> vdd;

// Floyd Warshall normal
// Funciona con float o double (cambiar vff por vdd)
vff floyd(const vff &grafo)
{
    int V = grafo.size();
    vff dist(grafo);

    for (int k = 0; k < V; k++)
        for (int i = 0; i < V; i++)
            for (int j = 0; j < V; j++)
                // si dist de: i->k->j < i->j
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
    return dist;
}

// Floyd Warshall vectorizado con 8 elementos
// Funciona solo con float
void floydVec(const vff &grafo)
{
    int V = grafo.size();
    vff dist(grafo);

    for (int k = 0; k < V; k++)
    {
        for (int i = 0; i < V; i++)
        {
            // Crear vector que almacena 8 veces dist[i][k]
            // (para despues comparar con otro vector)
            __m256 ik = _mm256_set1_ps(dist[i][k]);
            for (int j = 0; j < V - 7; j += 8) // ir de 8 en 8
            {
                // Cargar filas (de 8 elementos: j, j+1,..., j+7) al vector SIMD

                // Cargar fila k, dist[k][j] (de 8 en 8)
                __m256 kj = _mm256_loadu_ps(&dist[k][j]);
                // Cargar fila i, dist[i][j] (de 8 en 8)
                __m256 ij = _mm256_loadu_ps(&dist[i][j]);
                // Calcular dist[i][k] + dist[k][j] (fila de 8 elementos)
                __m256 ikj = _mm256_add_ps(ik, kj);
                // Minimo entre dist[i][j] y dist[i][k] + dist[k][j]
                __m256 result = _mm256_min_ps(ij, ikj);
                // Almacenar los resultados de vuelta en dist
                _mm256_storeu_ps(&dist[i][j], result);
            }
            // Elementos restantes: secuencial
            for (int j = V - V % 8; j < V; j++)
                if (dist[i][j] > dist[i][k] + dist[k][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
        }
    }
    // return dist;
}

// Floyd Warshall vectorizado con 8 elementos
// Funciona solo con float
void floydVec16(const vff &grafo)
{
    int V = grafo.size();
    vff dist(grafo);

    for (int k = 0; k < V; k++)
    {
        for (int i = 0; i < V; i++)
        {
            __m512 ik = _mm512_set1_ps(dist[i][k]);
            for (int j = 0; j < V - 15; j += 16)
            {
                __m512 kj = _mm512_loadu_ps(&dist[k][j]);
                __m512 ij = _mm512_loadu_ps(&dist[i][j]);
                __m512 ikj = _mm512_add_ps(ik, kj);
                __m512 result = _mm512_min_ps(ij, ikj);
                _mm512_storeu_ps(&dist[i][j], result);
            }
            for (int j = V - V % 16; j < V; j++)
                if (dist[i][j] > dist[i][k] + dist[k][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
        }
    }
    // return dist;
}

// Floyd Warshall paralelizado con OpenMP basico
// Funciona con float o double (cambiar vff por vdd)
void floydOpenMP(const vff &grafo)
{
    int V = grafo.size();
    vff dist(grafo);

    int i, j, k;
    for (k = 0; k < V; k++)
    {
#pragma omp parallel for private(i, j) schedule(static)
        for (i = 0; i < V; i++)
            for (j = 0; j < V; j++)
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
    }

    // return dist;
}

// Floyd Warshall paralelizado con OpenMP basico y vectorizacion
// Funciona con float o double (cambiar vff por vdd)
void floydVec8Par(const vff &grafo)
{
    int V = grafo.size();
    vff dist(grafo);

    for (int k = 0; k < V; k++)
    {
#pragma omp parallel for
        for (int i = 0; i < V; i++)
        {
            __m256 ik = _mm256_set1_ps(dist[i][k]);
            for (int j = 0; j < V - 7; j += 8)
            {
                __m256 kj = _mm256_loadu_ps(&dist[k][j]);
                __m256 ij = _mm256_loadu_ps(&dist[i][j]);

                __m256 ikj = _mm256_add_ps(ik, kj);
                __m256 result = _mm256_min_ps(ij, ikj);

                _mm256_storeu_ps(&dist[i][j], result);
            }
            for (int j = V - V % 8; j < V; j++)
                if (dist[i][j] > dist[i][k] + dist[k][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
        }
    }

    // return dist;
}

void floydb(vdd &C, const vdd &A, const vdd &B, const int b)
{
    for (int k = 0; k < b; k++)
        for (int i = 0; i < b; i++)
            for (int j = 0; j < b; j++)
                C[i][j] = min(C[i][j], A[i][k] + B[k][j]);
}

// Floyd Warshall paralelizado en bloques con OpenMP
// Funciona solo con double y V tiene que ser divisible por b!!!!
vdd floydBlock(const vdd &grafo, const int b)
{
    int V = grafo.size();
    if (V % b != 0)
    {
        cout << "Error: Tamaño de grafo V no divisible por tamaño de bloque b" << endl;
        return grafo;
    }
    int B = V / b;
    vdd dist(grafo);

    // Crear bloques (submatrices)
    vector<vector<vdd>> blocks(B, vector<vdd>(B, vdd(b, vd(b)))); // Matriz de bloques
    for (int i = 0; i < B; i++)
        for (int j = 0; j < B; j++)
            for (int ii = 0; ii < b; ii++)
                for (int jj = 0; jj < b; jj++)
                    blocks[i][j][ii][jj] = dist[i * b + ii][j * b + jj];

    // Recorrer los bloques de la diagonal
    for (int k = 0; k < B; k++)
    {
        // Calcular FW del bloque actual (que pertenece a la diagonal)
        floydb(blocks[k][k], blocks[k][k], blocks[k][k], b);

// Calcular FW para la fila entera (todas las columnas de la fila)
#pragma omp parallel for
        for (int j = 0; j < B; j++)
            if (j != k)
                floydb(blocks[k][j], blocks[k][k], blocks[k][j], b);

// Calcular FW para la columna entera (todas las filas de la columna)
#pragma omp parallel for
        for (int i = 0; i < B; i++)
        {
            if (i != k)
                floydb(blocks[i][k], blocks[i][k], blocks[k][k], b);
            // Calcular para las restantes
            for (int j = 0; j < B; j++)
                if (j != k)
                    floydb(blocks[i][j], blocks[i][k], blocks[k][j], b);
        }
    }

    // Copiar los resultados a la matriz original
    for (int i = 0; i < B; i++)
        for (int j = 0; j < B; j++)
            for (int ii = 0; ii < b; ii++)
                for (int jj = 0; jj < b; jj++)
                    dist[i * b + ii][j * b + jj] = blocks[i][j][ii][jj];

    return dist;
}