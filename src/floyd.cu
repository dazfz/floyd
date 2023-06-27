#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
    // Obtener i (vertical) y j (horizontal) (id del bloque * tamaño del bloque * id del thread)
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Memoria compartida
    __shared__ float ik[32], kj[32];

    // Como la matriz ahora es de una dimension
    // La fila i, se calcula como i * V (V elementos en 1 fila)
    // Luego, sumar indice de la columna (desplazamiento)
    if (i < V && j < V)
    {
        // Si es la primera columna
        if (!threadIdx.x)
            ik[threadIdx.y] = dist[i * V + k];
        // Si es la primera fila
        if (!threadIdx.y)
            kj[threadIdx.x] = dist[k * V + j];
        __syncthreads();

        float ikj = ik[threadIdx.y] + kj[threadIdx.x];
        dist[i * V + j] = fminf(dist[i * V + j], ikj);
    }
}

vff CUDA(const vff &grafo)
{
    int V = grafo.size();
    size_t size = V * V * sizeof(float);

    // Crear un vector en el devices y con el tamaño
    thrust::device_vector<float> device(size);

    // Copiar los datos del grafo al device (recorrer por fila)
    for (int i = 0; i < V; i++)
        thrust::copy(grafo[i].begin(), grafo[i].end(), device.begin() + i * V);

    // Dividir la matriz en bloques de tamaño B
    // Grid es el numero de bloques
    int B = 32;
    dim3 block(B, B);
    dim3 grid((V + B - 1) / B, (V + B - 1) / B);

    // Para cada k, se recorre paralelamente todos los bloques del grid, y se ejecuta el kernel FW
    for (int k = 0; k < V; k++)
        floydKernel<<<grid, block>>>(thrust::raw_pointer_cast(device.data()), V, k);

    // Crear vector para almacenar los resultados y copiarlos desde el device
    vff dist(V, vf(V));
    for (int i = 0; i < V; i++)
        thrust::copy(device.begin() + i * V, device.begin() + (i + 1) * V, dist[i].begin());

    return dist;
}

// vff CUDA(const vff &grafo)
// {
//     int V = grafo.size();
//     size_t size = V * V * sizeof(float);
//     float *host = (float *)malloc(size);
//     float *device;

//     for (int i = 0; i < V; i++)
//         memcpy(&host[i * V], grafo[i].data(), V * sizeof(float));
//     cudaMalloc((void **)&device, size);
//     cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);

//     dim3 blockSize(32, 32);
//     dim3 gridSize((V + blockSize.x - 1) / blockSize.x, (V + blockSize.y - 1) / blockSize.y);

//     for (int k = 0; k < V; k++)
//         floydKernel<<<gridSize, blockSize>>>(device, V, k);

//     cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);

//     vff dist(V, vf(V));
//     for (int i = 0; i < V; i++)
//         memcpy(dist[i].data(), &host[i * V], V * sizeof(float));

//     cudaFree(device);
//     free(host);
//     return dist;
// }
