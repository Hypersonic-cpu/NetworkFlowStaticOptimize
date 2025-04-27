#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ranges>
#include <tuple>
#include <string>
#include <sstream>
#include <chrono>

#include "shortest_paths.hh"

namespace stdv = std::views;

auto
ReadEdges(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        exit(1);
    }
    // [u, v, id]
    std::vector< std::tuple<int, int, int> > ret {};

    std::string line; 
    // Skip the first line
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);

        // Extract each number, separated by commas
        std::string temp;
        // [0] = edge number
        std::getline(ss, temp, ',');
        int edgeNo{ std::stoi(temp) };

        // Extract second number (b)
        std::getline(ss, temp, ',');
        int u{ std::stoi(temp) };

        // Extract third number (c)
        std::getline(ss, temp, ',');
        int v{ std::stoi(temp) };

         ret.emplace_back(std::make_tuple(u, v, edgeNo));

        // Now a, b, c, d contain the four integers from the line
        std::cout << "Read values: " << edgeNo << ", " << u << ", " << v << ", " << std::endl;
    }
    return ret;
}

int main(int argc, char* argv[]) {
    assert(argc >= 4);
    int const N { std::atoi(argv[1]) };
    int const S { std::atoi(argv[3]) };
    // int const T { std::atoi(argv[4]) };
    DirectedGraph g(N);
    using namespace std;
    
    auto const edges{ ReadEdges(std::string(argv[2])) };

    for (auto [u, v, no] : edges) {
        cout << u << " " << v << " " << no << endl;
        g.at(u).emplace_back(DirectedEdge{ v, no });
    }

    auto duration = chrono::microseconds(0);

    for (auto Ter = 0; Ter < N; ++Ter) {
        auto start = std::chrono::high_resolution_clock::now();
        auto lst = GraphAlgo::ShortestPaths(g, S, Ter);
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        duration += dur;    

        cout << "[" << Ter << "\t] ";
        for (auto elem: lst) {
            auto [_, v, _] = edges[elem];
            std::cout << elem << " p(" << v << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << "# TOTAL TIME USAGE: " << duration.count() << " us" << std::endl;
    return 0;
}
