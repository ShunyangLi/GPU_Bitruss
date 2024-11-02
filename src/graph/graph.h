/**
 * @file graph.h
 * @brief Graph data structure and related operations.
 *
 * This header file defines the Graph class, which encapsulates the
 * representation of a bipartite graph, including methods for loading
 * graphs from files (both binary and text formats) and converting
 * graphs to binary format. The Graph class maintains various attributes
 * related to the graph, such as the number of vertices and edges,
 * degrees of vertices, and edge support values.
 */

#pragma once
#ifndef BITRUSS_GRAPH_H
#define BITRUSS_GRAPH_H

#include "util/utility.h"


/**
 * @class Graph
 * @brief Represents a graph data structure for bitruss decomposition.
 *
 * @details This class provides methods for loading, processing, and managing graph data.
 * It supports loading from binary files and includes functionality for handling graph edges,
 * vertex degrees, and truss values.
 *
 * @note This class is intended for use in GPU-accelerated bitruss decomposition applications.
 */
class Graph {
public:
    explicit Graph(const std::string& filename, bool is_bin);
    auto graph_to_bin(const std::string& filename) -> void;
    auto load_graph_bin(const std::string& filename) -> void;
    ~Graph();

private:
    auto process_graph(const std::string& path) -> void;
    auto check_graph() const -> void;

public:
    // upper/lower vertices number, upper vertices max id, edge number
    uint u_num{}, l_num{}, u_max{}, m{}, n{};
    uint support_max{};
    uint* neighbors{};   // neighbors array, length = 2 * m
    uint* offsets{};     // offsets array, length = u_num + l_num + 1
    uint* degrees{};     // degrees array, length = u_num + l_num
    int* edge_support{};// edge support array, length = m
    uint* edges{};       // edges array, length = 2 * m
    uint* edge_ids{};    // edge ids array, length = 2 * m
    uint* bitruss{};     // bitruss number array, length = m
};

#endif//BITRUSS_GRAPH_H
