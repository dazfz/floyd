#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "kernels.cu"

using namespace std;

typedef vector<float> vf;
typedef vector<vector<float>> vff;

vff naiveCUDA(const vff &grafo)
{
    int V = grafo.size();
    size_t size = V * V * sizeof(float);

    // Allocate host and device vectors
    thrust::host_vector<float> host(size);
    thrust::device_vector<float> device = host;

    // Copy graph data from host to device
    for (int i = 0; i < V; i++)
        thrust::copy(grafo[i].begin(), grafo[i].end(), device.begin() + i * V);

    // Launch CUDA kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x, (V + blockSize.y - 1) / blockSize.y);

    for (int k = 0; k < V; k++)
        floydKernel<<<gridSize, blockSize>>>(device.data().get(), V, k);

    // Copy result from device to host
    thrust::copy(device.begin(), device.end(), host.begin());

    // Create vff object to store the result
    vff dist(V, vf(V));

    // Copy data from host to dist
    for (int i = 0; i < V; i++)
        thrust::copy(host.begin() + i * V, host.begin() + (i + 1) * V, dist[i].begin());

    return dist;
}

vff xdd(const vff &grafo)
{
    int V = grafo.size();
    size_t size = V * V * sizeof(float);

    // Allocate host and device vectors
    thrust::host_vector<float> host(size);
    thrust::device_vector<float> device = host;

    // Copy graph data from host to device
    for (int i = 0; i < V; i++)
        thrust::copy(grafo[i].begin(), grafo[i].end(), device.begin() + i * V);

    // Launch CUDA kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x, (V + blockSize.y - 1) / blockSize.y);

    for (int k = 0; k < V; k++)
        floydKernel<<<gridSize, blockSize>>>(device.data().get(), V, k);

    // Copy result from device to host
    thrust::copy(device.begin(), device.end(), host.begin());

    // Create vff object to store the result
    vff dist(V, vf(V));

    // Copy data from host to dist
    for (int i = 0; i < V; i++)
        thrust::copy(host.begin() + i * V, host.begin() + (i + 1) * V, dist[i].begin());

    return dist;
}

#define TILE_SIZE 32 // Adjust this value based on the available shared memory

__global__ void floyddKernel(float *dist, int V)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tileRow = by * TILE_SIZE;
    int tileCol = bx * TILE_SIZE;

    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    // Load the current tile (CR) into shared memory
    for (int i = 0; i < TILE_SIZE; i++)
    {
        int row = tileRow + i;
        for (int j = 0; j < TILE_SIZE; j++)
        {
            int col = tileCol + j;

            if (row < V && col < V)
                tile[i][j] = dist[row * V + col];
            else
                tile[i][j] = 1000000;
        }
    }

    __syncthreads();

    // Perform Floyd-Warshall within the current tile (CR)
    for (int k = 0; k < TILE_SIZE; k++)
    {
        for (int i = 0; i < TILE_SIZE; i++)
        {
            for (int j = 0; j < TILE_SIZE; j++)
            {
                if (tile[i][k] + tile[k][j] < tile[i][j])
                    tile[i][j] = tile[i][k] + tile[k][j];
            }
        }

        __syncthreads();

        // Perform calculations for other tile positions

        // Calculate shortest paths for tiles to the left (W)
        for (int i = 0; i < TILE_SIZE; i++)
        {
            for (int j = 0; j < k; j++)
            {
                if (tile[i][k] + tile[k][j] < tile[i][j])
                    tile[i][j] = tile[i][k] + tile[k][j];
            }
        }

        // Calculate shortest paths for tiles to the right (E)
        for (int i = 0; i < TILE_SIZE; i++)
        {
            for (int j = k + 1; j < TILE_SIZE; j++)
            {
                if (tile[i][k] + tile[k][j] < tile[i][j])
                    tile[i][j] = tile[i][k] + tile[k][j];
            }
        }

        // Calculate shortest paths for tiles at the top (N)
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < TILE_SIZE; j++)
            {
                if (tile[i][k] + tile[k][j] < tile[i][j])
                    tile[i][j] = tile[i][k] + tile[k][j];
            }
        }

        // Calculate shortest paths for tiles at the bottom (S)
        for (int i = k + 1; i < TILE_SIZE; i++)
        {
            for (int j = 0; j < TILE_SIZE; j++)
            {
                if (tile[i][k] + tile[k][j] < tile[i][j])
                    tile[i][j] = tile[i][k] + tile[k][j];
            }
        }

        // Calculate shortest paths for remaining diagonal tiles (NW, NE, SW, SE)
        for (int i = 0; i < k; i++)
        {
            for (int j = k + 1; j < TILE_SIZE; j++)
            {
                if (tile[i][k] + tile[k][j] < tile[i][j])
                    tile[i][j] = tile[i][k] + tile[k][j];
            }
        }

        for (int i = k + 1; i < TILE_SIZE; i++)
        {
            for (int j = 0; j < k; j++)
            {
                if (tile[i][k] + tile[k][j] < tile[i][j])
                    tile[i][j] = tile[i][k] + tile[k][j];
            }
        }

        for (int i = k + 1; i < TILE_SIZE; i++)
        {
            for (int j = k + 1; j < TILE_SIZE; j++)
            {
                if (tile[i][k] + tile[k][j] < tile[i][j])
                    tile[i][j] = tile[i][k] + tile[k][j];
            }
        }

        __syncthreads();
    }

    // Write back the updated tile to global memory
    for (int i = 0; i < TILE_SIZE; i++)
    {
        int row = tileRow + i;
        for (int j = 0; j < TILE_SIZE; j++)
        {
            int col = tileCol + j;

            if (row < V && col < V)
                dist[row * V + col] = tile[i][j];
        }
    }
}

vff CUDA1(const vff &grafo)
{
    int V = grafo.size();
    size_t size = V * V * sizeof(float);

    // Allocate host and device vectors
    thrust::host_vector<float> host(size);
    thrust::device_vector<float> device = host;

    // Copy graph data from host to device
    for (int i = 0; i < V; i++)
        thrust::copy(grafo[i].begin(), grafo[i].end(), device.begin() + i * V);

    // Launch CUDA kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x, (V + blockSize.y - 1) / blockSize.y);

    floyddKernel<<<gridSize, blockSize>>>(device.data().get(), V);

    // Copy result from device to host
    thrust::copy(device.begin(), device.end(), host.begin());

    // Create vff object to store the result
    vff dist(V, vf(V));

    // Copy data from host to dist
    for (int i = 0; i < V; i++)
        thrust::copy(host.begin() + i * V, host.begin() + (i + 1) * V, dist[i].begin());

    return dist;
}

__global__ void floydKernel2(float *dist, int V)
{
    int tile_size = blockDim.x; // Assuming blockDim.x = blockDim.y

    // Calculate the tile indices
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // Calculate the starting indices of the current tile
    int start_row = tile_row * tile_size;
    int start_col = tile_col * tile_size;

    // Calculate the ending indices of the current tile
    int end_row = start_row + tile_size;
    int end_col = start_col + tile_size;

    // Perform Floyd-Warshall algorithm for the current tile
    for (int k = 0; k < V; k++)
    {
        // Calculate the shortest path for the current tile (CR)
        for (int i = start_row + threadIdx.y; i < end_row; i += blockDim.y)
        {
            for (int j = start_col + threadIdx.x; j < end_col; j += blockDim.x)
            {
                // Calculate the indices within the graph matrix
                int dist_ik = dist[i * V + k];
                int dist_kj = dist[k * V + j];
                int dist_ij = dist[i * V + j];

                // Update the shortest path if a shorter path is found
                if (dist_ik + dist_kj < dist_ij)
                    dist[i * V + j] = dist_ik + dist_kj;
            }
        }
        __syncthreads();

        // Shortest path for all the tiles (W) left to CR is calculated.
        for (int i = start_row + threadIdx.y; i < end_row; i += blockDim.y)
        {
            for (int j = 0; j < start_col; j++)
            {
                int dist_ik = dist[i * V + k];
                int dist_kj = dist[k * V + j];
                int dist_ij = dist[i * V + j];
                if (dist_ik + dist_kj < dist_ij)
                    dist[i * V + j] = dist_ik + dist_kj;
            }
        }
        __syncthreads();

        // Shortest path for all the tiles (E) on the right of CR is calculated.
        for (int i = start_row + threadIdx.y; i < end_row; i += blockDim.y)
        {
            for (int j = end_col; j < V; j++)
            {
                int dist_ik = dist[i * V + k];
                int dist_kj = dist[k * V + j];
                int dist_ij = dist[i * V + j];
                if (dist_ik + dist_kj < dist_ij)
                    dist[i * V + j] = dist_ik + dist_kj;
            }
        }
        __syncthreads();

        // Shortest path for all the tiles at the top (N) of CR is calculated.
        for (int i = 0; i < start_row; i++)
        {
            for (int j = start_col + threadIdx.x; j < end_col; j += blockDim.x)
            {
                int dist_ik = dist[i * V + k];
                int dist_kj = dist[k * V + j];
                int dist_ij = dist[i * V + j];
                if (dist_ik + dist_kj < dist_ij)
                    dist[i * V + j] = dist_ik + dist_kj;
            }
        }
        __syncthreads();
        // Shortest path for all the tiles at the bottom (S) of CR is calculated.
        for (int i = end_row; i < V; i++)
        {
            for (int j = start_col + threadIdx.x; j < end_col; j += blockDim.x)
            {
                int dist_ik = dist[i * V + k];
                int dist_kj = dist[k * V + j];
                int dist_ij = dist[i * V + j];
                if (dist_ik + dist_kj < dist_ij)
                    dist[i * V + j] = dist_ik + dist_kj;
            }
        }
        __syncthreads();

        // Shortest path for the rest of the tiles at diagonals (NW, NE, SW, & SE) is calculated.
        for (int i = 0; i < V; i++)
        {
            for (int j = 0; j < V; j++)
            {
                if (i < start_row || i >= end_row || j < start_col || j >= end_col)
                {
                    int dist_ik = dist[i * V + k];
                    int dist_kj = dist[k * V + j];
                    int dist_ij = dist[i * V + j];
                    if (dist_ik + dist_kj < dist_ij)
                        dist[i * V + j] = dist_ik + dist_kj;
                }
            }
        }
        __syncthreads();
    }
}

vff CUDA2(const vff &grafo)
{
    int V = grafo.size();
    size_t size = V * V * sizeof(float);

    // Allocate host and device vectors
    thrust::host_vector<float> host(size);
    thrust::device_vector<float> device = host;

    // Copy graph data from host to device
    for (int i = 0; i < V; i++)
        thrust::copy(grafo[i].begin(), grafo[i].end(), device.begin() + i * V);

    // Launch CUDA kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x, (V + blockSize.y - 1) / blockSize.y);

    floydKernel2<<<gridSize, blockSize>>>(device.data().get(), V);

    // Copy result from device to host
    thrust::copy(device.begin(), device.end(), host.begin());

    // Create vff object to store the result
    vff dist(V, vf(V));

    // Copy data from host to dist
    for (int i = 0; i < V; i++)
        thrust::copy(host.begin() + i * V, host.begin() + (i + 1) * V, dist[i].begin());

    return dist;
}

// vff naiveCUDA(const vff &grafo)
// {
//     int V = grafo.size();
//     size_t size = V * V * sizeof(float);
//     float *host = (float *)malloc(size);
//     float *device;

//     // Copy graph data from host to device
//     for (int i = 0; i < V; i++)
//         memcpy(&host[i * V], grafo[i].data(), V * sizeof(float));
//     cudaMalloc((void **)&device, size);
//     cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);

//     // Launch CUDA kernel
//     dim3 blockSize(16, 16);
//     dim3 gridSize((V + blockSize.x - 1) / blockSize.x, (V + blockSize.y - 1) / blockSize.y);

//     for (int k = 0; k < V; k++)
//         floydKernel<<<gridSize, blockSize>>>(device, V, k);

//     cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);

//     // Copy resu from device to host
//     vff dist(V, vf(V));
//     for (int i = 0; i < V; i++)
//         memcpy(dist[i].data(), &host[i * V], V * sizeof(float));

//     cudaFree(device);
//     free(host);
//     return dist;
// }
