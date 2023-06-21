#include <iostream>
#include <vector>
#include <fstream>
#include <cstring> // strcmp
#include <chrono>
#include "print.cpp"
#include "floyd.cu"

using namespace std;

typedef vector<float> vf;
typedef vector<vector<float>> vff;

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
    vff dist2 = naiveCUDA(grafo);
  auto  finish = chrono::high_resolution_clock::now();
  auto  duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Cuda Naive: " << duration << " [ms]" << endl;

    start = chrono::high_resolution_clock::now();
    vff lol = xdd(grafo);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Cuda xd: " << duration << " [ms]" << endl;

    start = chrono::high_resolution_clock::now();
    //vff dist4 = CUDA1(grafo);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Cuda: " << duration << " [ms]" << endl;

    start = chrono::high_resolution_clock::now();
    //vff dist5 = CUDA2(grafo);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Cuda: " << duration << " [ms]" << endl;

    // print(dist4);
    //  iguales(dist1, dist2);
    //  iguales(dist1, dist3);
    //  iguales(dist1, dist4);
    iguales(dist2, lol);
    //iguales(dist2, dist4);
    //iguales(dist2, dist5);
    if (argc == 3 && strcmp(argv[2], "-p") == 0)
        reconstruction(grafo);
    return 0;
}