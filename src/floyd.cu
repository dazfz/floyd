#include <iostream>
#include <vector>
#include <fstream>
#include <cstring> // strcmp
#include <chrono>
#include <cuda_runtime.h>
#include "print.cpp"

using namespace std;

typedef vector<float> vf;
typedef vector<vector<float>> vff;

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

__global__ void floydKernel(float *dist, int V, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V && j < V)
    {
        int idx = i * V + j;
        int idx_k = i * V + k;
        int idx_jk = k * V + j;
        if (dist[idx_k] + dist[idx_jk] < dist[idx])
            dist[idx] = dist[idx_k] + dist[idx_jk];
    }
}

vff naiveCUDA(const vff &grafo)
{
    int V = grafo.size();
    size_t size = V * V * sizeof(float);
    float *dist_host = (float *)malloc(size);
    float *dist_device;
    vff result(V, vf(V));

    // Copy graph data from host to device
    for (int i = 0; i < V; i++)
        memcpy(&dist_host[i * V], grafo[i].data(), V * sizeof(float));
    cudaMalloc((void **)&dist_device, size);
    cudaMemcpy(dist_device, dist_host, size, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x, (V + blockSize.y - 1) / blockSize.y);

    for (int k = 0; k < V; k++)
        floydKernel<<<gridSize, blockSize>>>(dist_device, V, k);

    cudaMemcpy(dist_host, dist_device, size, cudaMemcpyDeviceToHost);

    // Copy result from device to host
    for (int i = 0; i < V; i++)
        memcpy(result[i].data(), &dist_host[i * V], V * sizeof(float));

    // Clean up memory
    cudaFree(dist_device);
    free(dist_host);

    return result;
}

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

    start = chrono::high_resolution_clock::now();
    vff dist2 = naiveCUDA(grafo);
    finish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    cout << "Cuda Naive: " << duration << " [ms]" << endl;

    print(dist2);
    iguales(dist1, dist2);

    // iguales(dist1, dist3);
    // iguales(dist1, dist4);
    // iguales(dist1, dist5);
    if (argc == 3 && strcmp(argv[2], "-p") == 0)
        reconstruction(grafo);
    return 0;
}
// __global__ void CUDAtile(vff &grafo, int tsize)
// {
//     int V = grafo.size();
//     auto n = V * V * sizeof(float);
//     int ntiles = size / tsize;

//     // Calculate the indices for the current tile block
//     int iTile = blockIdx.y;
//     int jTile = blockIdx.x;
//     int kTile = threadIdx.x;

//     int iStart = iTile * tileSize;
//     int jStart = jTile * tileSize;
//     int kStart = kTile * tileSize;

//     // Handle the remaining rows and columns
//     int iEnd = (iTile == numTiles - 1) ? size : iStart + tileSize;
//     int jEnd = (jTile == numTiles - 1) ? size : jStart + tileSize;

//     // Process the current tile (CR)
//     for (int i = iStart + threadIdx.y; i < iEnd; i += blockDim.y)
//     {
//         for (int j = jStart; j < jEnd; ++j)
//         {
//             // Calculate the shortest path for the current tile
//             int index = i * size + j;
//             int indexIK = i * size + kStart;
//             int indexKJ = kStart * size + j;
//             if (graph[indexIK] != INF && graph[indexKJ] != INF && graph[indexIK] + graph[indexKJ] < graph[index])
//             {
//                 graph[index] = graph[indexIK] + graph[indexKJ];
//             }
//         }
//     }

//     __syncthreads();

//     // Process tiles to the left (W)
//     for (int wTile = 0; wTile < iTile; ++wTile)
//     {
//         int wStart = wTile * tileSize;
//         int wEnd = wStart + tileSize;
//         for (int i = iStart + threadIdx.y; i < iEnd; i += blockDim.y)
//         {
//             for (int j = jStart; j < jEnd; ++j)
//             {
//                 if (j < wEnd)
//                 {
//                     int index = i * size + j;
//                     int indexIW = i * size + wStart;
//                     int indexWJ = wStart * size + j;
//                     if (graph[indexIW] != INF && graph[indexWJ] != INF && graph[indexIW] + graph[indexWJ] < graph[index])
//                     {
//                         graph[index] = graph[indexIW] + graph[indexWJ];
//                     }
//                 }
//             }
//         }
//     }

//     __syncthreads();

//     // Process tiles to the right (E)
//     for (int eTile = iTile + 1; eTile < numTiles; ++eTile)
//     {
//         int eStart = eTile * tileSize;
//         int eEnd = eStart + tileSize;
//         for (int i = iStart + threadIdx.y; i < iEnd; i += blockDim.y)
//         {
//             for (int j = jStart; j < jEnd; ++j)
//             {
//                 if (j >= eStart)
//                 {
//                     int index = i * size + j;
//                     int indexIE = i * size + eStart;
//                     int indexEJ = eStart * size + j;
//                     if (graph[indexIE] != INF && graph[indexEJ] != INF && graph[indexIE] + graph[indexEJ] < graph[index])
//                     {
//                         graph[index] = graph[indexIE] + graph[indexEJ];
//                     }
//                 }
//             }
//         }
//     }

//     __syncthreads();

//     // Process tiles above (N)
//     for (int nTile = 0; nTile < jTile; ++nTile)
//     {
//         int nStart = nTile * tileSize;
//         int nEnd = nStart + tileSize;
//         for (int i = iStart + threadIdx.y; i < iEnd; i += blockDim.y)
//         {
//             for (int j = jStart; j < jEnd; ++j)
//             {
//                 if (i < nEnd)
//                 {
//                     int index = i * size + j;
//                     int indexIN = i * size + nStart;
//                     int indexNJ = nStart * size + j;
//                     if (graph[indexIN] != INF && graph[indexNJ] != INF && graph[indexIN] + graph[indexNJ] < graph[index])
//                     {
//                         graph[index] = graph[indexIN] + graph[indexNJ];
//                     }
//                 }
//             }
//         }
//     }

//     __syncthreads();

//     // Process tiles below (S)
//     for (int sTile = jTile + 1; sTile < numTiles; ++sTile)
//     {
//         int sStart = sTile * tileSize;
//         int sEnd = sStart + tileSize;
//         for (int i = iStart + threadIdx.y; i < iEnd; i += blockDim.y)
//         {
//             for (int j = jStart; j < jEnd; ++j)
//             {
//                 if (i >= sStart)
//                 {
//                     int index = i * size + j;
//                     int indexIS = i * size + sStart;
//                     int indexSJ = sStart * size + j;
//                     if (graph[indexIS] != INF && graph[indexSJ] != INF && graph[indexIS] + graph[indexSJ] < graph[index])
//                     {
//                         graph[index] = graph[indexIS] + graph[indexSJ];
//                     }
//                 }
//             }
//         }
//     }

//     __syncthreads();

//     // Process the remaining diagonal tiles (NW, NE, SW, SE)
//     for (int dTile = 0; dTile < numTiles; ++dTile)
//     {
//         if (dTile == kTile)
//             continue;
//         int dStart = dTile * tileSize;
//         int dEnd = dStart + tileSize;
//         for (int i = iStart + threadIdx.y; i < iEnd; i += blockDim.y)
//         {
//             for (int j = jStart; j < jEnd; ++j)
//             {
//                 if (i >= dStart && j < kStart)
//                 {
//                     int index = i * size + j;
//                     int indexIK = i * size + kStart;
//                     int indexKD = kStart * size + dStart;
//                     int indexDJ = dStart * size + j;
//                     if (graph[indexIK] != INF && graph[indexKD] != INF && graph[indexDJ] != INF && graph[indexIK] + graph[indexKD] + graph[indexDJ] < graph[index])
//                     {
//                         graph[index] = graph[indexIK] + graph[indexKD] + graph[indexDJ];
//                     }
//                 }
//             }
//         }
//     }
// }

// void CUDA(const vff &grafo, int tsize)
// {
//     int V = grafo.size();
//     size_t size = V * V * sizeof(float);

//     // Convert vff to a raw array
//     std::vector<float> temp(grafo.size() * grafo.size());
//     std::copy(grafo.begin(), grafo.end(), temp.begin());
//     const float *d_dist = temp.data();

//     // Allocate device memory
//     float *d_result;
//     cudaMalloc((void **)&d_result, size);

//     // Copy data from host to device
//     cudaMemcpy(d_result, d_dist, size, cudaMemcpyHostToDevice);

//     // Calculate grid and block dimensions
//     dim3 blockSize(tsize, tsize);
//     dim3 gridSize((V + tsize - 1) / tsize, (V + tsize - 1) / tsize);

//     // Invoke the kernel
//     floydWarshallCUDA<<<gridSize, blockSize>>>(d_result, V, tsize);

//     // Copy results from device to host
//     std::vector<float> result(grafo.size() * grafo.size());
//     cudaMemcpy(result.data(), d_result, size, cudaMemcpyDeviceToHost);

//     // Clean up
//     cudaFree(d_result);
// }