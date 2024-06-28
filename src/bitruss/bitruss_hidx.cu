#include "bitruss.cuh"


using namespace std;

/**
 * binary search find if x in the neighbor of u,
 * if x in the neighbor of u, then return the neighbor position of x in the neighbor of u
 * if x not in the neighbor of u, then return -1
 * @param nbr neighbor array
 * @param left left index
 * @param right right index
 * @param value the value to find
 * @return idx or -1
 */
__inline__ __device__ auto binary_search(const uint *nbr, uint left, uint right, uint value) -> int {
    int edge_id = -1;
    while (left <= right) {
        uint mid = left + ((right - left) >> 1);
        if (nbr[mid] == value) {
            edge_id = int(mid);
            break;
        } else if (nbr[mid] < value) {
            left = mid + 1;
        } else {
            if (mid == 0) break;
            right = mid - 1;
        }
    }

    return edge_id;
}


/**
 * gpu bitruss decomposition based on h index
 * @param num_edge the number of edges
 * @param d_edges the edges array ([u, v], ...)
 * @param d_offset the neighbor offset array for each node
 * @param d_neighbors the neighbors array (v1, v2, ...
 * @param c_edge_support the current edge support for butterfly counting
 * @param p_edge_support the privous round edge support for butterfly counting
 * @param flag _stop or not
 */
__global__ void bitruss_decomposition_h_index(uint num_edge, const uint *d_edges, uint *d_offset,
                                              uint *d_neighbors, uint *c_edge_support,
                                              const uint *p_edge_support, const uint *d_edge_ids, uint *flag,
                                              uint *d_bitmaps, uint offset, uint blk_num, uint m) {

    uint bid = blockIdx.x;
    __shared__ uint* d_bitmap;
    __shared__ int idx;

    if (threadIdx.x == 0) {
        d_bitmap = d_bitmaps + bid * offset;
        idx = 0;
    }
    __syncthreads();

    for (uint uv = bid; uv < m; uv += blk_num) {
        if (uv >= m) break;

        if (threadIdx.x == 0) {
            idx = 0;
        }

        __syncthreads();

        uint const u = d_edges[uv * 2];
        uint const v = d_edges[uv * 2 + 1];

        uint const* u_nbr = d_neighbors + d_offset[u];
        uint const u_nbr_len = d_offset[u + 1] - d_offset[u];
        uint const *v_nbr = d_neighbors + d_offset[v];
        uint const v_nbr_len = d_offset[v + 1] - d_offset[v];

        for (uint i = threadIdx.x; i < v_nbr_len; i += blockDim.x) {
            uint const w = v_nbr[i];
            if (w == u) continue;

            uint const wv = d_edge_ids[d_offset[v] + i];
            uint const* w_nbr = d_neighbors + d_offset[w];
            uint const w_nbr_len = d_offset[w + 1] - d_offset[w];

            uint j = 0;
            uint k = 0;
            while (j < w_nbr_len && k < u_nbr_len) {
                if (w_nbr[j] < u_nbr[k]) {
                    j++;
                } else if (w_nbr[j] > u_nbr[k]) {
                    k++;
                } else if (w_nbr[j] == u_nbr[k] && w_nbr[j] == v) {
                    // get the edge id
                    j++;
                    k++;
                } else {
                    // get edge id
                    uint const wx = d_edge_ids[d_offset[w] + j];
                    uint const ux = d_edge_ids[d_offset[u] + k];
                    j++;
                    k++;

                    if (p_edge_support[ux] == 0 || p_edge_support[wx] == 0 || p_edge_support[wv] == 0) continue;
                    uint const min_val = min(min(p_edge_support[ux], p_edge_support[wx]), p_edge_support[wv]);
                    uint idxx = atomicAdd(&idx, 1);
                    d_bitmap[idxx] = min_val;
                }
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            int stack[32];
            int top = -1;
            stack[++top] = 0;
            stack[++top] = idx - 1;

            while (top >= 0) {
                int end = stack[top--];
                int start = stack[top--];

                if (start >= end) continue;

                uint pivot = d_bitmap[(start + end) / 2];
                int i = start - 1;
                int j = end + 1;

                while (true) {
                    while (d_bitmap[++i] > pivot);
                    while (d_bitmap[--j] < pivot);
                    if (i >= j) break;
                    uint temp = d_bitmap[i];
                    d_bitmap[i] = d_bitmap[j];
                    d_bitmap[j] = temp;
                }

                stack[++top] = start;
                stack[++top] = i - 1;
                stack[++top] = j + 1;
                stack[++top] = end;
            }

            uint h_index = 0;
            for (uint i = 0; i < idx; i++) {
                if (d_bitmap[i] >= i + 1) {
                    h_index = i + 1;
                } else {
                    break;
                }
            }

            // set current edge support atomically
            c_edge_support[uv] = h_index;
            if (h_index != p_edge_support[uv]) {
                atomicExch(flag, 0);
            }

        }
        __syncthreads();
    }
}


/**
 * bitruss decomposition based on h-index
 * for edge support, we only need to kown the current round and privous round value
 * two array to store the current edge support and privous edge support
 * @param g graph object
 */
auto g_bitruss_hindex(Graph *g) -> void {

    bfc_evpp(g);
    log_info("running bitruss decomposition based on h-index on GPU");

    uint *flag;
    uint *c_edge_support, *p_edge_support;
    uint *d_edges;
    uint *d_offset;
    uint *d_neighbors;
    uint *d_edge_ids;
    // to store the min value of the edge support
    uint *d_bitmaps;

    auto blocks = int (TOTAL_MEMORY * 0.9 / 4 / g->support_max ) - 1;
    if (blocks > g->m) blocks = g->m;

    // alloca memory
    cudaMalloc(&d_edges, sizeof(uint) * g->m * 2);
    cudaMalloc(&d_offset, sizeof(uint) * (g->n + 1));
    cudaMalloc(&d_neighbors, sizeof(uint) * g->m * 2);
    // previouse round edge support should be g->edge_support value
    cudaMalloc((void **) &p_edge_support, sizeof(uint) * g->m);
    cudaMalloc((void **) &c_edge_support, sizeof(uint) * g->m);
    cudaMalloc((void **) &d_edge_ids, sizeof(uint) * g->m * 2);

    // copy memory from host to device
    cudaMemcpy(p_edge_support, g->edge_support, sizeof(uint) * g->m, cudaMemcpyHostToDevice);
    cudaMemset(c_edge_support, 0, sizeof(uint) * g->m);
    cudaMemcpy((void *) d_edges, (void *) g->edges, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_offset, (void *) g->offsets, sizeof(uint) * (g->n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_neighbors, (void *) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_edge_ids, (void *) g->edge_ids, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);

    CER(cudaMalloc(&d_bitmaps, sizeof(uint) * g->support_max * blocks));

    // the set the framework, flag is to check whether _stop
    uint coverge = 0, round = 0;
    cudaMalloc(&flag, sizeof(uint));

    while (!coverge) {
        round += 1;
        cudaMemset(flag, 1, sizeof(uint));

        bitruss_decomposition_h_index<<<blocks, BLK_DIM>>>(g->m, d_edges, d_offset, d_neighbors, c_edge_support,
                                                       p_edge_support, d_edge_ids, flag, d_bitmaps, g->support_max, blocks, g->m);
        cudaDeviceSynchronize();

        cudaMemcpy(&coverge, flag, sizeof(uint), cudaMemcpyDeviceToHost);
        // swap the edge support
        cudaMemcpy(p_edge_support, c_edge_support, sizeof(uint) * g->m, cudaMemcpyDeviceToDevice);
        cudaMemset(c_edge_support, 0, sizeof(uint) * g->m);
    }


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_trace("CUDA error: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // get the bitrussness for each edge
    cudaMemcpy(g->bitruss, p_edge_support, sizeof(uint) * g->m, cudaMemcpyDeviceToHost);

    cudaFree(d_edges);
    cudaFree(d_offset);
    cudaFree(d_neighbors);
    cudaFree(c_edge_support);
    cudaFree(p_edge_support);
    cudaFree(d_edge_ids);
    cudaFree(flag);
    cudaFree(d_bitmaps);

    auto kmax =
            *std::max_element(g->bitruss, g->bitruss + g->m);
    log_info("k-bitruss max value %d, with %d rounds", kmax, round);
}
