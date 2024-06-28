#include "bfc.cuh"
#include "cuda_runtime.h"
#include <thrust/execution_policy.h>


using namespace std;

// define some global variables
__constant__ uint* c_offset;
__constant__ uint* c_row;
__constant__ int c_threadsPerEdge;
__constant__ long* c_sums;
__constant__ uint* c_bitmap;
__constant__ uint* c_nonZeroRow;
__constant__ uint c_edgeSize;
__constant__ uint c_nodeSize;
__constant__ uint c_nodeoffset;


__global__ void bfc(uint totalNodeNum, uint nodenum_select, uint nonZeroSize) {
    long sum = 0;
    // get the block id
    uint curRowNum = blockIdx.x;
    uint intSizePerBitmap = nodenum_select;
    uint* myBitmap = c_bitmap + blockIdx.x * intSizePerBitmap;

    while (true) {

        uint u_id;
        if (curRowNum < nonZeroSize) u_id = c_nonZeroRow[curRowNum];
        else
            u_id = totalNodeNum;

        // check if the vertex id is out of node range
        // u_id is the current node id
        if (u_id >= totalNodeNum) break;

        // get the current node's neighbor and neighbor length
        uint* curNodeNbr = c_row + c_offset[u_id];
        uint curNodeNbrLength = c_offset[u_id + 1] - c_offset[u_id];

        // init the bitmap when the thread is the first thread in the block
        if (threadIdx.x == 0) memset(myBitmap, 0, sizeof(int) * intSizePerBitmap);

        __threadfence();
        if (nonZeroSize > 32) {
            // for each neighbor of current node, get the two hoop neighbor
            for (uint i = 0; i < (curNodeNbrLength + blockDim.x - 1) / blockDim.x; i++) {
                uint v_id = i * blockDim.x + threadIdx.x;
                if (v_id < curNodeNbrLength) {
                    uint curNbr = curNodeNbr[v_id];
                    uint* twoHoopNbr = c_row + c_offset[curNbr];
                    uint twoHoopNbrLength = c_offset[curNbr + 1] - c_offset[curNbr];
                    for (uint j = 0; j < twoHoopNbrLength; j++) {
                        uint w_id = twoHoopNbr[j];
                        if (w_id > u_id) {
                            // printf("u_id is %d, v_id is %d, w_id is %d\n", u_id, v_id, w_id);
                            atomicAdd(myBitmap + w_id, 1);
                        }
                    }
                    __threadfence();
                }
            }
        } else {
            for (uint i = 0; i < (curNodeNbrLength + blockDim.x - 1) / blockDim.x; i++) {
                uint v_id = i * blockDim.x + threadIdx.x;
                if (v_id < curNodeNbrLength) {
                    uint curNbr = curNodeNbr[v_id];
                    uint* twoHoopNbr = c_row + c_offset[curNbr];
                    uint twoHoopNbrLength = c_offset[curNbr + 1] - c_offset[curNbr];
                    for (int j = 0; j < twoHoopNbrLength; j++) {
                        uint w_id = twoHoopNbr[j];
                        if (w_id > u_id) {
                            // printf("u_id is %d, v_id is %d, w_id is %d\n", u_id, v_id, w_id);
                            atomicAdd(myBitmap + w_id - c_nodeoffset, 1);
                        }
                    }
                }
            }
        }

        __syncthreads();
        curRowNum += gridDim.x;
    }

    // compute the number of butterflies for each block
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (blockIdx.x < nonZeroSize) {
        for (uint i = 0; i < (nonZeroSize + blockDim.x - 1) / blockDim.x; i++) {
            uint j = i * blockDim.x + threadIdx.x;
            if (j < nonZeroSize) {
                sum += myBitmap[j] * (myBitmap[j] - 1) / 2;
            }
        }
        __threadfence();
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
        if (threadIdx.x % 32 == 0) {
            c_sums[idx >> 5] = sum;
        }
    }
    __syncthreads();
}


/**
 * compute the number of butterflies in G
 * @param g graph object
 */
auto butterfly_counting(Graph* g) -> void {
    uint nonZeroSize = 0;
    int curThreadsPerEdge = 1;
    uint nodeNum = g->u_num + g->l_num;
    uint* nonZeroRow = new uint[nodeNum];
    // denote _start from which side (upper or lower)
    uint nodeoffset = g->u_num > g->l_num ? g->u_num : 0;

    // nonZeroSize denotes the upper or lower vertices that have neighbors
    for (auto u = nodeoffset; u < nodeNum; u++) {
        if (g->offsets[u] != g->offsets[u + 1]) {
            nonZeroRow[nonZeroSize++] = u;
        }
    }


    auto blockSize = 32;
    auto blockNum = 30 * 2048 / blockSize;

    auto intSizePerBitmap = g->u_num + g->l_num;
    auto maxWarpPerGrid = blockNum * blockSize / 32;

    //    if (blockNum * intSizePerBitmap * sizeof(int) / 1024 > 8 * 1024 * 1024) {
    //        log_error("RUN OUT OF GLOBAL MEMORY!!");
    //        exit(0);
    //    }

    uint *t_edgeOffset, *t_edgeRow;
    long* t_sum;

    // init t_sum
    cudaMalloc(&t_sum, sizeof(long) * maxWarpPerGrid);
    cudaMemset(t_sum, 0, sizeof(long) * maxWarpPerGrid);

    long* h_sum = new long[maxWarpPerGrid];
    // init edge offset and edge row
    cudaMalloc(&t_edgeOffset, sizeof(uint) * (nodeNum + 1));
    cudaMalloc(&t_edgeRow, sizeof(uint) * g->m * 2);
    cudaMemcpy((void*) t_edgeOffset, (void*) g->offsets, sizeof(uint) * (nodeNum + 1), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) t_edgeRow, (void*) g->neighbors, sizeof(uint) * (g->m * 2), cudaMemcpyHostToDevice);

    // init nonZeroRow with t_nonzerrow
    uint* t_nonZeroRow;
    cudaMalloc(&t_nonZeroRow, sizeof(uint) * nonZeroSize);
    cudaMemcpy((void*) t_nonZeroRow, (void*) nonZeroRow, sizeof(uint) * nonZeroSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_nonZeroRow, &t_nonZeroRow, sizeof(uint*));

    // just for init some useful variables in butterfly counting code
    uint* d_bitmaps;
    cudaMalloc(&d_bitmaps, sizeof(uint) * intSizePerBitmap * blockNum);
    cudaMemcpyToSymbol(c_bitmap, &d_bitmaps, sizeof(uint*));
    cudaMemcpyToSymbol(c_edgeSize, &g->u_num, sizeof(uint));
    cudaMemcpyToSymbol(c_nodeSize, &nodeNum, sizeof(uint));
    cudaMemcpyToSymbol(c_nodeoffset, &nodeoffset, sizeof(uint));
    cudaMemcpyToSymbol(c_offset, &t_edgeOffset, sizeof(uint*));
    cudaMemcpyToSymbol(c_row, &t_edgeRow, sizeof(uint*));
    cudaMemcpyToSymbol(c_sums, &t_sum, sizeof(long*));
    cudaMemcpyToSymbol(c_threadsPerEdge, &curThreadsPerEdge, sizeof(int));

    // butterfly counting kernel code here
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    cudaEventRecord(start_time);

    bfc<<<blockNum, blockSize>>>(nodeNum, nonZeroSize, nonZeroSize);

    cudaDeviceSynchronize();
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    float cudatime;
    cudaEventElapsedTime(&cudatime, start_time, stop_time);

    log_info("cuda exact counting time: %f s", cudatime / 1000);
    cudaMemcpy((void*) h_sum, (void*) t_sum, sizeof(long) * maxWarpPerGrid, cudaMemcpyDeviceToHost);
    long bfCount = thrust::reduce(h_sum, h_sum + maxWarpPerGrid);
    log_info("the number of butterflies is %ld", bfCount);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    // free cuda memory
    cudaEventDestroy(start_time);
    cudaEventDestroy(stop_time);
    cudaFree(t_edgeOffset);
    cudaFree(t_edgeRow);
    cudaFree(t_sum);
    cudaFree(d_bitmaps);

    delete[] h_sum;
}


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
__inline__ __device__ auto binary_search(const uint* nbr, uint left, uint right, uint value) -> int {
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
 * compute the number of butterflies for each edge navie version
 * @param num_edge the number of edges
 * @param d_edges the edges pair
 * @param d_offset the offset of each node
 * @param d_neighbors the neighbors of each node
 * @param d_edge_support the number of butterflies for each edge
 */
__global__ void bfc_per_edge(uint num_edge, const uint* d_edges, uint* d_offset, uint* d_neighbors, uint* d_edge_support) {
    // get the edge id
    uint e_id = blockIdx.x * blockDim.x + threadIdx.x;

    // make sure the edge id is in the range
    if (e_id >= num_edge) return;

    uint u = d_edges[e_id * 2];
    uint v = d_edges[e_id * 2 + 1];

    uint* u_nbr = d_neighbors + d_offset[u];
    uint u_nbr_len = d_offset[u + 1] - d_offset[u];

    for (uint i = 0; i < u_nbr_len; i++) {
        uint w = u_nbr[i];
        if (w == v) continue;
        uint* w_nbr = d_neighbors + d_offset[w];
        uint w_nbr_len = d_offset[w + 1] - d_offset[w];
        for (uint j = 0; j < w_nbr_len; j++) {
            uint x = w_nbr[j];
            if (x == u) continue;

            // check if x in the neighbor of v
            // if x in the neighbor of v, then there is a butterfly
            // therefore, use binary search to check if x in the neighbor of v
            uint left = 0, right = d_offset[v + 1] - d_offset[v] - 1;
            uint* v_nbr = d_neighbors + d_offset[v];

            auto edge_id = binary_search(v_nbr, left, right, x);
            if (edge_id != -1) {
                atomicAdd(&d_edge_support[e_id], 1);
            }
        }
    }
}


/**
 * compute the number of butterflies for each edge load balance version
 * @param num_edge the number of edges
 * @param d_edges the edges pair
 * @param d_offset the offset of each node
 * @param d_neighbors the neighbors of each node
 * @param d_edge_support the number of butterflies for each edge
 */
__global__ void bfc_es_lb(uint num_edge, const uint* d_edges, uint* d_offset, uint* d_neighbors, uint* d_edge_support) {

    auto bID = blockIdx.x;
    auto stride = gridDim.x;

    // get the edge id for each block
    for (auto e_id = bID; e_id < num_edge; e_id += stride) {
        //        uint e_id = blockIdx.x;
        // make sure the edge id is in the range
        if (e_id >= num_edge) return;

        uint u = d_edges[e_id * 2];
        uint v = d_edges[e_id * 2 + 1];

        uint* u_nbr = d_neighbors + d_offset[u];
        uint u_nbr_len = d_offset[u + 1] - d_offset[u];

        for (uint i = 0; i < (u_nbr_len + blockDim.x - 1) / blockDim.x; i++) {
            uint u_id = i * blockDim.x + threadIdx.x;
            if (u_id >= u_nbr_len) break;

            uint w = u_nbr[u_id];
            if (w == v) continue;

            uint* w_nbr = d_neighbors + d_offset[w];
            uint w_nbr_len = d_offset[w + 1] - d_offset[w];
            for (uint j = 0; j < w_nbr_len; j++) {
                uint x = w_nbr[j];
                if (x == u) continue;

                // check if x in the neighbor of v
                // if x in the neighbor of v, then there is a butterfly
                // therefore, use binary search to check if x in the neighbor of v
                uint left = 0, right = d_offset[v + 1] - d_offset[v] - 1;
                uint* v_nbr = d_neighbors + d_offset[v];

                auto edge_id = binary_search(v_nbr, left, right, x);
                if (edge_id != -1) {
                    atomicAdd(&d_edge_support[e_id], 1);
                }
            }
        }
    }
}

/**
 * TODO compute the number of butterflies for each edge load balance version saving computing resource
 * @param num_edge the number of edges
 * @param d_edges the edges pair
 * @param d_offset the offset of each node
 * @param d_neighbors the neighbors of each node
 * @param d_edge_support the number of butterflies for each edge
 */
__global__ void bfc_es_lb_adv(uint num_edge, const uint* d_edges, uint* d_offset,
                              uint* d_neighbors, uint* d_edge_support, const uint* d_edge_ids) {

    auto bID = blockIdx.x;
    auto stride = gridDim.x;

    // get the edge id for each block
    for (auto e_id = bID; e_id < num_edge; e_id += stride) {
        //        uint e_id = blockIdx.x;
        // make sure the edge id is in the range
        if (e_id >= num_edge) return;

        uint u = d_edges[e_id * 2];
        uint v = d_edges[e_id * 2 + 1];

        uint* u_nbr = d_neighbors + d_offset[u];
        uint u_nbr_len = d_offset[u + 1] - d_offset[u];

        for (uint i = 0; i < (u_nbr_len + blockDim.x - 1) / blockDim.x; i++) {
            uint u_id = i * blockDim.x + threadIdx.x;
            if (u_id >= u_nbr_len) break;

            uint w = u_nbr[u_id];
            if (w == v) continue;

            uint uw = d_edge_ids[d_offset[u] + u_id];
            if (uw < e_id) continue;

            uint* w_nbr = d_neighbors + d_offset[w];
            uint w_nbr_len = d_offset[w + 1] - d_offset[w];
            for (uint j = 0; j < w_nbr_len; j++) {
                uint x = w_nbr[j];
                if (x == u) continue;

                uint wx = d_edge_ids[d_offset[w] + j];
                if (x < u) continue;
                if (wx < uw) continue;

                // check if x in the neighbor of v
                // if x in the neighbor of v, then there is a butterfly
                // therefore, use binary search to check if x in the neighbor of v
                uint left = 0, right = d_offset[v + 1] - d_offset[v] - 1;
                uint* v_nbr = d_neighbors + d_offset[v];

                auto edge_id = binary_search(v_nbr, left, right, x);
                if (edge_id == -1) continue;

                uint xv = d_edge_ids[d_offset[v] + edge_id];
                if (xv > wx) continue;

                // print all edge id
                //                printf("e_id is %d, uw is %d, wx is %d, xv is %d\n", e_id, uw, wx, xv);
                atomicAdd(&d_edge_support[e_id], 1);
                atomicAdd(&d_edge_support[wx], 1);
                atomicAdd(&d_edge_support[xv], 1);
                atomicAdd(&d_edge_support[uw], 1);
            }
        }
    }
}

/**
 * compute the number of butterflies for each edge
 * @param g graph object
 */
auto butterfly_counting_per_edge(Graph* g) -> void {
    log_info("running butterfly counting per edge");

    auto node_num = g->u_num + g->l_num;
    uint* d_edges;
    uint* d_offset;
    uint* d_neighbors;
    uint* d_edge_support;
    uint* d_edge_ids;

    // alloca memory
    cudaMalloc(&d_edges, sizeof(uint) * g->m * 2);
    cudaMalloc(&d_offset, sizeof(uint) * (node_num + 1));
    cudaMalloc(&d_neighbors, sizeof(uint) * g->m * 2);
    cudaMalloc(&d_edge_support, sizeof(uint) * g->m);
    cudaMalloc(&d_edge_ids, sizeof(uint) * g->m * 2);

    // copy memory from host to device
    cudaMemcpy((void*) d_edges, (void*) g->edges, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_offset, (void*) g->offsets, sizeof(uint) * (node_num + 1), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_neighbors, (void*) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_edge_support, (void*) g->edge_support, sizeof(uint) * g->m, cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_edge_ids, (void*) g->edge_ids, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    cudaEventRecord(start_time);

    //    auto blockSize = 32;
    //    auto blockNum = (g->m + blockSize - 1) / blockSize;
    //    bfc_per_edge<<< blockNum, blockSize >>> (g->m - 1, d_edges, d_offset, d_neighbors, d_edge_support);
    //
    //    cudaDeviceSynchronize();
    //    cudaEventRecord(stop_time);
    //    cudaEventSynchronize(stop_time);
    //    float cudatime;
    //    cudaEventElapsedTime( & cudatime, start_time, stop_time);
    //
    //    log_info("butterfly edge support baseline counting time: %f s", cudatime / 1000);

    cudaEventRecord(start_time);
    float cudatime = 0;
    //    auto blockSize = 64;
    //    auto blockNum = g->m;

    auto blockSize = dim3(128);
    //    auto gridSize = dim3((g->m + blockSize.x - 1) / blockSize.x);

    bfc_es_lb<<<g->m, blockSize>>>(g->m, d_edges, d_offset, d_neighbors, d_edge_support);
    //    bfc_es_lb_adv<<<g->m, blockSize>>>(g->m, d_edges, d_offset, d_neighbors, d_edge_support, d_edge_ids);

    cudaDeviceSynchronize();
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&cudatime, start_time, stop_time);

    log_info("butterfly edge support load balance counting time: %f s", cudatime / 1000);

    memset(g->edge_support, 0, sizeof(uint) * g->m);
    // copy d_edge_support from deive to host
    cudaMemcpy((void*) g->edge_support, (void*) d_edge_support, sizeof(uint) * g->m, cudaMemcpyDeviceToHost);

    // free useless memeory
    cudaFree(d_edges);
    cudaFree(d_offset);
    cudaFree(d_neighbors);
    cudaFree(d_edge_support);
    cudaFree(d_edge_ids);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_trace("CUDA error: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // get the max edge support
    g->support_max = *std::max_element(g->edge_support, g->edge_support + g->m);

    log_info("the max edge support is %d", g->support_max);

#ifdef DISPLAY_RESULT
    auto map = std::unordered_map<uint, uint>();
    for (auto e = 0; e < g->m; e++) {
        if (g->edge_support[e] == 0) continue;
        map[g->edge_support[e]]++;
        std::cout << "e " << e << " edge support " << g->edge_support[e] << std::endl;
    }

    // print map
    for (auto& it : map) {
        std::cout << "edge support " << it.first << " count " << it.second << std::endl;
    }
#endif
}