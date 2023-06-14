#include <iostream>
#include <vector>
#include <iomanip> // setear el espacio del print

using namespace std;

typedef vector<float> vf;
typedef vector<vector<float>> vff;
const float INF = 1000000;

void print(const vff &dist)
{
    int V = dist.size();

    cout << endl;
    cout << "dist: " << endl;

    cout << "   ";
    for (int j = 0; j < V; j++)
        cout << setw(3) << j + 1 << " ";
    cout << endl;

    cout << "   ";
    for (int j = 0; j < V; j++)
        cout << "----";
    cout << endl;

    for (int i = 0; i < V; i++)
    {
        cout << i + 1 << "| ";
        for (int j = 0; j < V; j++)
            dist[i][j] == INF ? cout << setw(3) << "INF " : cout << setw(3) << dist[i][j] << " ";
        cout << endl;
    }
}

void iguales(const vff &dist1, const vff &dist2)
{
    int V = dist1.size();
    bool f = true;
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            if (dist1[i][j] != dist2[i][j])
            {
                f = false;
                break;
            }
        }
        if (!f)
            break;
    }
    if (f)
        cout << "iguales" << endl;
    else
        cout << "distintas" << endl;
}

vf camino(int u, int v, const vff &prev)
{
    if (prev[u][v] == -INF)
        return {};

    vf path;
    path.push_back(v);

    while (u != v)
    {
        v = prev[u][v];
        path.insert(path.begin(), v);
    }
    return path;
}

vff reconstruction(const vff &grafo)
{
    int V = grafo.size();
    vff dist(grafo);
    
    vff prev(V, vf(V, -INF));
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            if (dist[i][j] != INF && i != j)
                prev[i][j] = i;
    for (int i = 0; i < V; i++)
        prev[i][i] = 0;

    for (int k = 0; k < V; k++)
        for (int i = 0; i < V; i++)
            for (int j = 0; j < V; j++)
                if (dist[i][j] > dist[i][k] + dist[k][j])
                {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    prev[i][j] = prev[k][j];
                }
    print(grafo);
    print(dist);

    while (1)
    {
        cout << endl;
        cout << "Vertices: ";
        int u, v;
        cin >> u >> v;
        u--, v--;
        vf path = camino(u, v, prev);
        if (path.empty())
        {
            cout << "No existe camino de " << u + 1 << " a " << v + 1 << endl;
            continue;
        }
        cout << "Camino: ";
        for (int i = 0; i < path.size(); i++)
        {
            cout << path[i] + 1;
            if (i != path.size() - 1)
                cout << " -> ";
        }
        cout << endl;
        cout << "Costo: ";
        float cost = 0;
        for (int i = 0; i < path.size() - 1; i++)
        {
            int u = path[i];
            int v = path[i + 1];
            i == 0 ? cout << setw(5) << dist[u][v] : cout << setw(2) << dist[u][v];
            cost += dist[u][v];
            if (i != path.size() - 2)
                cout << " + ";
        }
        cout << " = " << cost << endl;
    }
}