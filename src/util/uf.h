#pragma once
#ifndef BITRUSS_UF_H
#define BITRUSS_UF_H

class UnionFind {
public:
    explicit UnionFind(uint n) {
        this->count = n;
        parent = new uint[n];

        for (int i = 0; i < n; i++) parent[i] = i;
    }

    auto union_(uint p, uint q) -> void {
        uint rootP = find(p);
        uint rootQ = find(q);

        if (rootP == rootQ) return;
        parent[rootQ] = rootP;
        count--;
    }

    auto connected(uint p, uint q) -> bool {
        uint rootP = find(p);
        uint rootQ = find(q);
        return rootP == rootQ;
    }

    auto find(uint x) -> uint {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    // return the number of connected components
    [[nodiscard]] uint count_() const {
        return count;
    }

private:
    uint count;
    uint *parent;
};

#endif//BITRUSS_UF_H