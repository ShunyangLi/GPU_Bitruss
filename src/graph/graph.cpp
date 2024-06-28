#include "graph.h"

// check graph for consist edge id
auto Graph::check_graph() const -> void {
    log_warn("start checking graph, this may take a while");

    {
#pragma omp parallel num_threads(THREADS) default(none)
        {
#pragma omp for schedule(dynamic)
            for (auto u = 0; u < u_num; u++) {
                auto u_nbr_len = offsets[u + 1] - offsets[u];
                for (auto i = 0; i < u_nbr_len; i++) {
                    auto e_id = edge_ids[offsets[u] + i];

                    auto v = neighbors[offsets[u] + i];
                    auto v_nbr = neighbors + offsets[v];
                    auto v_nbr_len = offsets[v + 1] - offsets[v];

                    for (auto j = 0; j < v_nbr_len; j++) {
                        if (v_nbr[j] == u) {
                            auto n_eid = edge_ids[offsets[v] + j];
                            if (e_id != n_eid) {
                                log_error("edge id not equal");
                                exit(EXIT_FAILURE);
                            }

                            auto tu = edges[e_id * 2];
                            auto tv = edges[e_id * 2 + 1];

                            if (tu != u || tv != v) {
                                log_error("edge id corresponding (u, v) incorrect");
                                exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
            }
        }
    }

    log_success("graph check pass !");
}

/**
 * load http://konect.cc/ graph data to binary file
 * @param graph_file graph file from konect
 * @param bin_file store the binary file
 * @param bin_file store the binary file
 */

auto Graph::process_graph(const std::string& path) -> void {
    log_info("processing graph file: %s", path.c_str());

    auto file = std::ifstream(path);
    std::string line;

    if (!file.is_open()) {
        log_error("Cannot open file %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }

    auto upper_ids = std::vector<uint>(MAX_IDS, MAX_IDS);
    auto lower_ids = std::vector<uint>(MAX_IDS, MAX_IDS);

    auto upper_edges = std::vector<std::vector<uint>>();
    auto lower_edges = std::vector<std::vector<uint>>();


    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == '%')  continue;

        std::istringstream iss(line);
        int u, v;
        iss >> u >> v;

        if (upper_ids[u] == MAX_IDS) {
            upper_ids[u] = u_num ++;
        }

        if (lower_ids[v] == MAX_IDS) {
            lower_ids[v] = l_num ++;
        }

        upper_edges.resize(u_num);
        lower_edges.resize(l_num);

        upper_edges[upper_ids[u]].push_back(lower_ids[v]);
        lower_edges[lower_ids[v]].push_back(upper_ids[u]);

        m += 1;
    }

    file.close();

    // re-assign the vertices id for lower vertices
    auto offset = u_num;
    for (auto & nv : upper_edges) {
        for (auto & v : nv) {
            v += offset;
        }
    }

    // then merge the upper and lower vertices
    upper_edges.resize(u_num + l_num);

    for (auto i = 0; i < lower_edges.size(); i++) {
        upper_edges[i + offset] = lower_edges[i];
    }


    // sort the neighbor of each vertex with parallel
    {
#pragma omp parallel num_threads(THREADS)
        {
#pragma omp for schedule(dynamic)
            for (auto & upper_edge : upper_edges) {
                std::sort(upper_edge.begin(), upper_edge.end());
            }
        }
    }


    n = u_num + l_num;
    u_max = u_num;

    degrees = new uint[n];
    offsets = new uint[n + 1];
    neighbors = new uint[m * 2];
    edges = new uint[m * 2];
    edge_ids = new uint[m * 2];
    edge_support = new int[m];
    bitruss = new uint[m];

    for (auto u = 0;  u < n; u++) {
        degrees[u] = upper_edges[u].size();
    }

    // assign offset
    offsets[0] = 0;
    for (auto i = 0; i < n; i++) {
        offsets[i + 1] = offsets[i] + degrees[i];
    }

    // assign neighbors
    auto all_neighbors = std::vector<uint>();
    for (auto u = 0; u < n; u++) {
        for (auto v : upper_edges[u]) {
            all_neighbors.push_back(v);
        }
    }

    assert(all_neighbors.size() == m * 2);
    std::copy(all_neighbors.begin(), all_neighbors.end(), neighbors);

    // init the edges pair
    auto e_id = 0;
    for (uint u = 0; u < u_num; u++) {

        auto u_nbr = neighbors + offsets[u];
        auto u_nbr_len = offsets[u + 1] - offsets[u];

        for (auto i = 0; i < u_nbr_len; i++) {
            auto v = u_nbr[i];

            edge_ids[offsets[u] + i] = e_id / 2;
            // record edges pair
            edges[e_id] = u;
            edges[e_id + 1] = v;
            e_id += 2;
        }
    }

    {
#pragma omp parallel num_threads(THREADS) default(none)
        {
#pragma omp for schedule(dynamic)
            for (auto v = u_num; v < u_num + l_num; v++) {

                auto v_nbr = neighbors + offsets[v];
                auto v_nbr_len = offsets[v + 1] - offsets[v];

                for (auto i = 0; i < v_nbr_len; i++) {
                    auto u = v_nbr[i];
                    auto u_nbr = neighbors + offsets[u];
                    auto left = 0;
                    auto right = int(offsets[u + 1] - offsets[u] - 1);

                    while (left <= right) {
                        auto mid = left + ((right - left) >> 1);
                        if (u_nbr[mid] == v) {
                            auto t_id = edge_ids[offsets[u] + mid];
#pragma omp atomic write
                            edge_ids[offsets[v] + i] = t_id;
                            break;
                        } else if (u_nbr[mid] < v) {
                            left = mid + 1;
                        } else {
                            if (mid == 0) break;
                            right = mid - 1;
                        }
                    }
                }
            }
        }
    }

    // assign edge ids

    bitruss = new uint[m];
    std::fill(bitruss, bitruss + m, 0);
    std::fill(edge_support, edge_support + m, 0);


    log_info("graph with upper: %'d, lower: %'d, vertices: %'d, edges: %'d", u_num, l_num, n, m);

#ifdef CHECK_GRAPH
    check_graph();
#endif
}

Graph::Graph(const std::string& filename, bool is_to_bin) {
    u_num = 0;
    l_num = 0;
    m = 0;
    u_max = 0;
    support_max = 0;
    n = 0;
    if (is_to_bin) {
        process_graph(filename);
    } else {
        if (filename.find(".bin") == std::string::npos) {
            log_error("graph file should be binary file");
            exit(EXIT_FAILURE);
        }
        load_graph_bin(filename);
    }
}

/**
 * convert graph to binary file
 * @param filename
 */
auto Graph::graph_to_bin(const std::string& filename) -> void {
    log_info("convert graph to binary file: %s", filename.c_str());

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    out.write(reinterpret_cast<const char*>(&u_num), sizeof(u_num));
    out.write(reinterpret_cast<const char*>(&l_num), sizeof(l_num));
    out.write(reinterpret_cast<const char*>(&u_max), sizeof(u_max));
    out.write(reinterpret_cast<const char*>(&m), sizeof(m));
    out.write(reinterpret_cast<const char*>(&support_max), sizeof(support_max));

    // write array member
    auto size = static_cast<std::streamsize>(sizeof(uint));
    out.write(reinterpret_cast<const char*>(neighbors), size * 2 * m);
    out.write(reinterpret_cast<const char*>(offsets), size * (u_num + l_num + 1));
    out.write(reinterpret_cast<const char*>(degrees), size * (u_num + l_num));
    out.write(reinterpret_cast<const char*>(edge_support), size * m);
    out.write(reinterpret_cast<const char*>(edges), size * 2 * m);
    out.write(reinterpret_cast<const char*>(edge_ids), size * 2 * m);
    out.write(reinterpret_cast<const char*>(bitruss), size * m);

    out.close();
}

/**
 * load graph from binary file
 * @param filename
 */
auto Graph::load_graph_bin(const std::string& filename) -> void {
    log_info("loading graph from binary file: %s", filename.c_str());

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // read single integer member
    in.read(reinterpret_cast<char*>(&u_num), sizeof(u_num));
    in.read(reinterpret_cast<char*>(&l_num), sizeof(l_num));
    in.read(reinterpret_cast<char*>(&u_max), sizeof(u_max));
    in.read(reinterpret_cast<char*>(&m), sizeof(m));
    in.read(reinterpret_cast<char*>(&support_max), sizeof(support_max));

    // allocate memory
    neighbors = new uint[2 * m];
    offsets = new uint[u_num + l_num + 1];
    degrees = new uint[u_num + l_num];
    edge_support = new int[m];
    edges = new uint[2 * m];
    edge_ids = new uint[2 * m];
    bitruss = new uint[m];

    // read array member
    auto size = static_cast<std::streamsize>(sizeof(uint));

    in.read(reinterpret_cast<char*>(neighbors), size * 2 * m);
    in.read(reinterpret_cast<char*>(offsets), size * (u_num + l_num + 1));
    in.read(reinterpret_cast<char*>(degrees), size * (u_num + l_num));
    in.read(reinterpret_cast<char*>(edge_support), size * m);
    in.read(reinterpret_cast<char*>(edges), size * 2 * m);
    in.read(reinterpret_cast<char*>(edge_ids), size * 2 * m);
    in.read(reinterpret_cast<char*>(bitruss), size * m);

    in.close();

    n = u_num + l_num;

    log_info("graph with upper: %'d, lower: %'d, vertices: %'d, edges: %'d", u_num, l_num, n, m);
}

Graph::~Graph() {
    delete[] neighbors;
    delete[] offsets;
    delete[] degrees;
}
