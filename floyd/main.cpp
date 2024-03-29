#include <iostream>
#include <vector>
#include <fstream>
#include <cstring> // strcmp
#include <chrono>
#include "print.cpp"
#include "floyd.cpp"

using namespace std;

typedef vector<float> vf;
typedef vector<vector<float>> vff;
typedef vector<double> vd;
typedef vector<vector<double>> vdd;

int main(int argc, char *argv[])
{
    if (argc < 2)
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
    vff dist1 =
        floyd(grafo);
    auto finish = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Normal: " << duration << " [ms]" << endl;

    start = chrono::high_resolution_clock::now();
    // vff dist2 =
    floydVec(grafo);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Vectorizado8: " << duration << " [ms]" << endl;

    start = chrono::high_resolution_clock::now();
    // vff dist3 =
    floydVec16(grafo);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Vectorizado16: " << duration << " [ms]" << endl;

    start = chrono::high_resolution_clock::now();
    // vff dist4 =
    floydOpenMP(grafo);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Paralelizado OpenMP: " << duration << " [ms]" << endl;

    start = chrono::high_resolution_clock::now();
    // vff dist5 =
    floydVec8Par(grafo);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Vectorizado + Paralelizado OpenMP: " << duration << " [ms]" << endl;

    vdd grafod(V, vd(V, INF));
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            grafod[i][j] = static_cast<double>(grafo[i][j]);

    start = chrono::high_resolution_clock::now();
    vdd dist6 =
        floydBlock(grafod, 2);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Paralelizado en bloques OpenMP: " << duration << " [ms]" << endl;

    // iguales(dist1, dist2);
    // iguales(dist1, dist3);
    // iguales(dist1, dist4);
    // iguales(dist1, dist5);
    // iguales(dist1, dist6);
    if (argc == 3 && strcmp(argv[2], "-p") == 0)
        reconstruction(grafo);
    return 0;
}
