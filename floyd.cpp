#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <immintrin.h>
#include <fstream>

using namespace std;

typedef vector<float> vf;
typedef vector<vector<float>> vff;
const float INF = 1000000;

void print(const vff &dist)
{
    int V = dist.size();
    cout << "dist: " << endl;
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
            dist[i][j] == INF ? cout << setw(3) << "INF " : cout << setw(3) << dist[i][j] << " ";
        cout << endl;
    }
}

vff floyd(const vff &grafo)
{
    int V = grafo.size();
    vff dist(grafo);

    for (int k = 0; k < V; k++)
        for (int i = 0; i < V; i++)
            for (int j = 0; j < V; j++)
                // si dist de: i->k->j < i->j
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
    // print(dist);
    return dist;
}

vff floydVec(const vff &grafo)
{
    int V = grafo.size();
    vff dist(grafo);

    for (int k = 0; k < V; k++)
    {
        for (int i = 0; i < V; i++)
        {
            // Crear vector que almacena 8 veces dist[i][k]
            __m256 ik = _mm256_set1_ps(dist[i][k]);
            // ir de 8 en 8
            for (int j = 0; j < V - 7; j += 8)
            {
                // Casting, y luego carga la fila (de 8 elementos) al vector simd
                // Cargar fila k, dist[k][j] (de 8 en 8)
                __m256 kj = _mm256_loadu_ps(&dist[k][j]);
                // Cargar fila i, dist[i][j] (de 8 en 8)
                __m256 ij = _mm256_loadu_ps(&dist[i][j]);

                // Calcular dist[i][k] + dist[k][j]
                __m256 ikj = _mm256_add_ps(ik, kj);

                // Obtener el mínimo entre dist[i][j] y dist[i][k] + dist[k][j]
                __m256 result = _mm256_min_ps(ij, ikj);

                // Almacenar los resultados de vuelta en la matriz
                _mm256_storeu_ps(&dist[i][j], result);
            }

            // Elementos restantes, secuencial
            for (int j = V - V % 8; j < V; j++)
                if (dist[i][j] > dist[i][k] + dist[k][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
        }
    }
    return dist;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Uso: programa <nombre_archivo>" << endl;
        return 1;
    }
    string s = argv[1];
    ifstream archivo(s);
    int V, E;
    archivo >> V >> E;

    vff grafo(V, vf(V, INF));
    for (int i = 0; i < V; i++)
        grafo[i][i] = 0;

    for (int i = 0; i < E; i++)
    {
        int u, v;
        float w;
        archivo >> u >> v >> w;
        grafo[u - 1][v - 1] = w;
    }
    archivo.close();

    auto start = chrono::high_resolution_clock::now();
    vff dist1 = floyd(grafo);
    auto finish = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Normal: " << duration << " [ms]" << endl;

    start = chrono::high_resolution_clock::now();
    vff dist2 = floydVec(grafo);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Vectorizado: " << duration << " [ms]" << endl;

    if (dist1 == dist2)
        cout << "Las funciones dieron el mismo resultado." << endl;
    else
        cout << "Las funciones dieron resultados diferentes." << endl;

    return 0;
}

/*

4,6

0 1 3
1 0 2
0 3 5
1 3 4
3 2 2
2 1 1

0 3 7 5
2 0 6 4
3 1 0 5
5 3 2 0

4,5

0 2 -2
1 0 4
3 1 -1
1 2 3
2 3 2

0 -1 -2 0
4 0 2 4
5 1 0 2
3 -1 1 0

8, 11
0 1 6
2 1 6
2 4 1
2 5 8
3 1 3
3 7 8
4 2 5
4 7 4
5 6 9
6 4 5
6 7 9

  0   6 INF INF INF INF INF INF
INF   0 INF INF INF INF INF INF
INF   6   0 INF   1   8  17   5
INF   3 INF   0 INF INF INF   8
INF  11   5 INF   0  13  22   4
INF  25  19 INF  14   0   9  18
INF  16  10 INF   5  18   0   9
INF INF INF INF INF INF INF   0

18, 31

0 1 2
0 4 3
1 2 3
1 4 8
1 5 3
2 1 5
2 3 8
3 2 4
3 17 3
5 1 8
5 2 8
5 6 5
5 8 3
6 9 3
7 0 4
7 11 9
8 4 7
8 11 7
9 5 5
10 17 9
11 7 8
11 8 7
11 12 6
12 8 6
12 11 9
12 13 3
12 16 6
13 9 5
16 13 4
16 17 7
17 13 9

  0   2   5  13   3   5  10  23   8  13 INF  15  21  24 INF INF  27  16
 25   0   3  11   8   3   8  21   6  11 INF  13  19  22 INF INF  25  14
 30   5   0   8  13   8  13  26  11  16 INF  18  24  20 INF INF  30  11
 34   9   4   0  17  12  17  30  15  17 INF  22  28  12 INF INF  34   3
INF INF INF INF   0 INF INF INF INF INF INF INF INF INF INF INF INF INF
 22   8   8  16  10   0   5  18   3   8 INF  10  16  19 INF INF  22  19
 30  16  16  24  18   8   0  26  11   3 INF  18  24  27 INF INF  30  27
  4   6   9  17   7   9  14   0  12  17 INF   9  15  18 INF INF  21  20
 19  21  24  32   7  24  29  15   0  21 INF   7  13  16 INF INF  19  26
 27  13  13  21  15   5  10  23   8   0 INF  15  21  24 INF INF  27  24
 50  36  36  44  38  28  33  46  31  23   0  38  44  18 INF INF  50   9
 12  14  17  25  14  17  22   8   7  14 INF   0   6   9 INF INF  12  19
 21  21  21  29  13  13  18  17   6   8 INF   9   0   3 INF INF   6  13
 32  18  18  26  20  10  15  28  13   5 INF  20  26   0 INF INF  32  29
INF INF INF INF INF INF INF INF INF INF INF INF INF INF   0 INF INF INF
INF INF INF INF INF INF INF INF INF INF INF INF INF INF INF   0 INF INF
 36  22  22  30  24  14  19  32  17   9 INF  24  30   4 INF INF   0   7
 41  27  27  35  29  19  24  37  22  14 INF  29  35   9 INF INF  41   0

*/