#include <iostream>
#include <vector>
#include <immintrin.h> // SIMD
#include <omp.h>       // OpenMP

using namespace std;

typedef vector<float> vf;
typedef vector<vector<float>> vff;

void floyd(const vff &grafo)
{
    int V = grafo.size();
    vff dist(grafo);

    for (int k = 0; k < V; k++)
        for (int i = 0; i < V; i++)
            for (int j = 0; j < V; j++)
                // si dist de: i->k->j < i->j
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
    // return dist;
}

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