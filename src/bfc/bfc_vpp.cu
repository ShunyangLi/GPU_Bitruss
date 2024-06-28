#include "bfc.cuh"
#include <cuda_runtime.h>

/**
 * butterfly counting with vertex priority
 * @param n number of vertices
 * @param blk_num block number
 * @param cnt the number of butterfly counting
 */
__global__ auto bfc_kernel(uint n, const uint* d_offset, const uint* d_neighbors, uint* d_bitmaps, int* last_uses,
                           const uint* d_degree, uint const blk_num, ull* cnt) -> void {
    uint bid = blockIdx.x;

    __shared__ uint* d_bitmap;
    __shared__ int* last_use;
    __shared__ uint last_idx;
    __shared__ ull local_cnt;

    if (threadIdx.x == 0) {
        d_bitmap = d_bitmaps + bid * n;
        last_use = last_uses + bid * n;
        last_idx = 0;
        local_cnt = 0;
    }

    __syncthreads();

    for (uint u = bid; u < n; u += blk_num) {

        if (u >= n) break;

        __syncthreads();

        for (uint j = threadIdx.x; j < last_idx; j += blockDim.x) {
            if (j >= last_idx) break;
            d_bitmap[last_use[j]] = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            last_idx = 0;
            local_cnt = 0;
        }

        __syncthreads();

        uint const u_nbr_len = d_degree[u];
        uint const* u_nbr = d_neighbors + d_offset[u];

        for (uint i = threadIdx.x; i < u_nbr_len; i += blockDim.x) {
            if (i >= u_nbr_len) break;

            uint v = u_nbr[i];

            if (d_degree[u] < d_degree[v]) continue;
            if (d_degree[u] == d_degree[v] && u <= v) continue;

            uint const* v_nbr = d_neighbors + d_offset[v];
            uint const v_nbr_len = d_degree[v];

            for (uint j = 0; j < v_nbr_len; j++) {
                uint w = v_nbr[j];

                if (d_degree[u] < d_degree[w]) continue;
                if (d_degree[u] == d_degree[w] && u <= w) continue;


                uint old_val = atomicAdd(d_bitmap + w, 1);
                atomicAdd(&local_cnt, old_val);

                if (old_val == 0) {
                    uint idxx = atomicAdd(&last_idx, 1);
                    last_use[idxx] = w;
                }
            }
        }

        __syncthreads();
         if (threadIdx.x == 0) {
             atomicAdd(cnt, local_cnt);
         }

    }
}

/**
 * counting butterflies for each edge with vertex priority
 * @param n number of vertices
 * @param blk_num block number
 */
__global__ auto ebfc_kernel(uint n, const uint* d_offset, const uint* d_neighbors,
                            const uint* d_edge_ids, uint* d_edge_support,
                            uint* d_bitmaps, int* last_uses, const uint* d_degree, uint const blk_num) -> void {

    uint const bid = blockIdx.x;

    __shared__ uint* d_bitmap;
    __shared__ int* last_use;
    __shared__ uint last_idx;

    if (threadIdx.x == 0) {
        d_bitmap = d_bitmaps + bid * n;
        last_use = last_uses + bid * n;
        last_idx = 0;
    }

    __syncthreads();

    for (uint u = bid; u < n; u += blk_num) {

        if (u >= n) break;

        __syncthreads();

        for (uint j = threadIdx.x; j < last_idx; j += blockDim.x) {
            if (j >= last_idx) break;
            d_bitmap[last_use[j]] = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0) last_idx = 0;

        __syncthreads();

        uint const u_nbr_len = d_degree[u];
        uint const* u_nbr = d_neighbors + d_offset[u];

        for (uint i = threadIdx.x; i < u_nbr_len; i += blockDim.x) {
            if (i >= u_nbr_len) break;

            uint v = u_nbr[i];

            if (d_degree[u] < d_degree[v]) continue;
            if (d_degree[u] == d_degree[v] && u <= v) continue;

            uint const* v_nbr = d_neighbors + d_offset[v];
            uint const v_nbr_len = d_degree[v];

            for (uint j = 0; j < v_nbr_len; j++) {
                uint w = v_nbr[j];

                if (d_degree[u] < d_degree[w]) continue;
                if (d_degree[u] == d_degree[w] && u <= w) continue;

                uint old_val = atomicAdd(d_bitmap + w, 1);

                if (old_val == 0) {
                    uint idxx = atomicAdd(&last_idx, 1);
                    last_use[idxx] = w;
                }
            }
        }

        __syncthreads();

        for (uint i = threadIdx.x; i < u_nbr_len; i += blockDim.x) {
            if (i >= u_nbr_len) break;

            uint v = u_nbr[i];

            if (d_degree[u] < d_degree[v]) continue;
            if (d_degree[u] == d_degree[v] && u <= v) continue;

            uint const* v_nbr = d_neighbors + d_offset[v];
            uint const v_nbr_len = d_degree[v];

            for (auto j = 0; j < v_nbr_len; j++) {
                uint w = v_nbr[j];

                if (d_degree[u] < d_degree[w]) continue;
                if (d_degree[u] == d_degree[w] && u <= w) continue;

                if (d_bitmap[w] == 0) continue;

                int dlt = int(d_bitmap[w]) - 1;
                if (dlt) {
                    uint const uv = d_edge_ids[d_offset[u] + i];
                    uint const vw = d_edge_ids[d_offset[v] + j];

                    atomicAdd(d_edge_support + uv, dlt);
                    atomicAdd(d_edge_support + vw, dlt);
                }
            }
        }
    }
}

/**
 * butterfly counting with vertex priority
 * we used
 * @param g graph object
 */
auto bfc_evpp(Graph* g) -> void {

    uint* d_offset;
    uint* d_neighbors;
    uint* d_edge_support;
    uint* d_edge_ids;
    uint* d_bitmaps;
    int* d_last_uses;
    uint* d_degree;
    size_t free_memory;


    // alloca memory
    CER(cudaMalloc(&d_offset, sizeof(uint) * (g->n + 1)));
    CER(cudaMalloc(&d_neighbors, sizeof(uint) * g->m * 2));
    CER(cudaMalloc(&d_edge_support, sizeof(uint) * g->m));
    CER(cudaMalloc(&d_edge_ids, sizeof(uint) * g->m * 2));
    CER(cudaMalloc(&d_degree, sizeof(uint) * g->n));

    cudaMemGetInfo(&free_memory, nullptr);
    uint blk_num = free_memory  * 0.96 / (g->n * 4 * 2);
    blk_num = blk_num > g->n ? BLK_NUMS : blk_num;

    CER(cudaMalloc(&d_bitmaps, sizeof(uint) * g->n * blk_num));
    CER(cudaMalloc(&d_last_uses, sizeof(int) * g->n * blk_num));


    // copy memory from host to device
    cudaMemcpy((void*) d_offset, (void*) g->offsets, sizeof(uint) * (g->n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_neighbors, (void*) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemset((void*) d_edge_support, 0, sizeof(uint) * g->m);
    cudaMemcpy((void*) d_edge_ids, (void*) g->edge_ids, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemset((void*) d_bitmaps, 0, sizeof(uint) * g->n * blk_num);
    cudaMemset((void*) d_last_uses, -1, sizeof(int) * g->n * blk_num);
    cudaMemcpy((void*) d_degree, (void*) g->degrees, sizeof(uint) * g->n, cudaMemcpyHostToDevice);

    ebfc_kernel<<<blk_num, BLK_DIM>>>(g->n, d_offset, d_neighbors, d_edge_ids, d_edge_support, d_bitmaps, d_last_uses, d_degree, blk_num);

    cudaDeviceSynchronize();

    // get the max edge support
    cudaMemcpy((void*) g->edge_support, (void*) d_edge_support, sizeof(uint) * g->m, cudaMemcpyDeviceToHost);
    g->support_max = *std::max_element(g->edge_support, g->edge_support + g->m);
    log_info("butterfly counting with vertex priority on gpu with %d blocks, and the max edge support is %'d", blk_num, g->support_max);

// count butterfly
#ifdef COUNT_BUTTERFLY
    ull* d_cnt;
    CER(cudaMalloc(&d_cnt, sizeof(ull)));
    CER(cudaMemset(d_cnt, 0, sizeof(ull)));

    bfc_kernel<<<blk_num, BLK_DIM>>>(g->n, d_offset, d_neighbors, d_bitmaps, d_last_uses, d_degree, blk_num, d_cnt);
    cudaDeviceSynchronize();

    ull cnt;
    cudaMemcpy(&cnt, d_cnt, sizeof(ull), cudaMemcpyDeviceToHost);
    log_info("total butterfly counting with vertex priority on gpu: %'llu", cnt);

    cudaFree(d_cnt);
#endif

    // get synchronize error
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        exit(EXIT_FAILURE);
    }

    // free cuda memory
    cudaFree(d_offset);
    cudaFree(d_neighbors);
    cudaFree(d_edge_support);
    cudaFree(d_edge_ids);
    cudaFree(d_bitmaps);
    cudaFree(d_last_uses);
    cudaFree(d_degree);
}


/**
 * butterfly counting (cnt_only) with vertex priority
 * we used
 * @param g graph object
 */
auto bfc_vpp(Graph* g) -> void {

    uint* d_offset;
    uint* d_neighbors;
    uint* d_edge_support;
    uint* d_edge_ids;
    uint* d_bitmaps;
    int* d_last_uses;
    uint* d_degree;
    size_t free_memory;
    ull* d_cnt;


    // alloca memory
    CER(cudaMalloc(&d_offset, sizeof(uint) * (g->n + 1)));
    CER(cudaMalloc(&d_neighbors, sizeof(uint) * g->m * 2));
    CER(cudaMalloc(&d_edge_support, sizeof(uint) * g->m));
    CER(cudaMalloc(&d_edge_ids, sizeof(uint) * g->m * 2));
    CER(cudaMalloc(&d_degree, sizeof(uint) * g->n));
    CER(cudaMalloc(&d_cnt, sizeof(ull)));

    cudaMemGetInfo(&free_memory, nullptr);
    uint blk_num = free_memory  * 0.96 / (g->n * 4 * 2);
    blk_num = blk_num > g->n ? BLK_NUMS : blk_num;

    CER(cudaMalloc(&d_bitmaps, sizeof(uint) * g->n * blk_num));
    CER(cudaMalloc(&d_last_uses, sizeof(int) * g->n * blk_num));


    // copy memory from host to device
    cudaMemcpy((void*) d_offset, (void*) g->offsets, sizeof(uint) * (g->n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_neighbors, (void*) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemset((void*) d_edge_support, 0, sizeof(uint) * g->m);
    cudaMemcpy((void*) d_edge_ids, (void*) g->edge_ids, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemset((void*) d_bitmaps, 0, sizeof(uint) * g->n * blk_num);
    cudaMemset((void*) d_last_uses, -1, sizeof(int) * g->n * blk_num);
    cudaMemcpy((void*) d_degree, (void*) g->degrees, sizeof(uint) * g->n, cudaMemcpyHostToDevice);
    CER(cudaMemset(d_cnt, 0, sizeof(ull)));

    bfc_kernel<<<blk_num, BLK_DIM>>>(g->n, d_offset, d_neighbors, d_bitmaps, d_last_uses, d_degree, blk_num, d_cnt);
    cudaDeviceSynchronize();

    // get the max edge support
    ull cnt;
    cudaMemcpy(&cnt, d_cnt, sizeof(ull), cudaMemcpyDeviceToHost);
    log_info("butterfly counting with vertex priority on gpu with %d blocks, and btf number: %'llu", blk_num, cnt);


    // get synchronize error
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        exit(EXIT_FAILURE);
    }

    // free cuda memory
    cudaFree(d_offset);
    cudaFree(d_neighbors);
    cudaFree(d_edge_support);
    cudaFree(d_edge_ids);
    cudaFree(d_bitmaps);
    cudaFree(d_last_uses);
    cudaFree(d_degree);
    cudaFree(d_cnt);
}
