__global__ void floydKernel(float *dist, int V, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < V && j < V)
    {
        int ij = i * V + j;
        int ik = i * V + k;
        int jk = k * V + j;

        dist[ij] = min(dist[ij], dist[ik] + dist[jk]);
    }
}

__global__ void xd(float *dist, int V, int k)
{
    extern __shared__ float sharedDist[];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = i * V + j;
    int idx_k = i * V + k;
    int idx_jk = k * V + j;

    // Copy data from global memory to shared memory
    sharedDist[idx] = dist[idx];
    sharedDist[idx_k] = dist[idx_k];
    sharedDist[idx_jk] = dist[idx_jk];

    __syncthreads();

    if (i < V && j < V)
        if (sharedDist[idx_k] + sharedDist[idx_jk] < sharedDist[idx])
            sharedDist[idx] = sharedDist[idx_k] + sharedDist[idx_jk];
    __syncthreads();

    // Copy data from shared memory back to global memory
    dist[idx] = sharedDist[idx];
}
