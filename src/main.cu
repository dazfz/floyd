#include <iostream>
#include <vector>
#include <fstream>
#include <cstring> // strcmp
#include <chrono>
#include <cmath>
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
    vff dist1 = floyd(grafo);
    auto finish = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Normal: " << duration << " [ms]" << endl;

    vector<double> tiempos;
    for (int i = 0; i < 10; ++i)
    {
        auto start = chrono::high_resolution_clock::now();
        vff dist2 = CUDA(grafo);
        auto finish = chrono::high_resolution_clock::now();
        double duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
        tiempos.push_back(duration);
        cout << "CUDA (" << i + 1 << "): " << duration << " [ms]" << endl;
    }
    double promedio = 0.0;
    for (double tiempo : tiempos)
        promedio += tiempo;
    promedio /= tiempos.size();

    double sumaDiferenciasCuadrado = 0.0;
    for (double tiempo : tiempos)
    {
        double diferencia = tiempo - promedio;
        sumaDiferenciasCuadrado += diferencia * diferencia;
    }
    double desviacionEstandar = sqrt(sumaDiferenciasCuadrado / tiempos.size());

    cout << "Promedio: " << promedio << " [ms]" << endl;
    cout << "Desviacion Estandar: " << desviacionEstandar << " [ms]" << endl;
    // iguales(dist1, dist2);

    if (argc == 3 && strcmp(argv[2], "-p") == 0)
        reconstruction(grafo);
    return 0;
}