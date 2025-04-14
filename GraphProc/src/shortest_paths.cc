#include "shortest_paths.hh"

#include <cassert>
#include <limits>
#include <list>
#include <queue>
#include <vector>

std::list<int> 
GraphAlgo::ShortestPaths(DirectedGraph const& NextOf, 
              int const src, int const dst) {
    int const N = NextOf.size();
    // Dijkstra
    // No `visited` control since the relaxed nodes must have 
    // dist[node] <- dist[cur] + 1.
    // std::vector<bool> visited( N, false );
    std::vector<int> dist( N, std::numeric_limits<int>::max() );
    DirectedGraph prev( N );
    std::queue<int> qu {};
    
    dist.at(src) = 0; 
    qu.emplace(src);
    
    while(!qu.empty()) {
        int cur = qu.front();
        qu.pop();
        // `cur` has reached optimal dist.
        
        for (auto [to, no]: NextOf[cur]) {
            if (dist.at(cur) + 1 < dist.at(to)) {
                // New shortest path.
                prev.at(to).assign({ DirectedEdge{ cur, no } });
                qu.emplace(to);
                dist.at(to) = dist.at(cur) + 1;
            } else if (dist[cur] + 1 == dist[to]) {
                // Add new path.
                prev.at(to).emplace_back(DirectedEdge{ cur, no });
            }
        }
    }

    return GraphAlgo::BfsEdges(prev, dst);
}

std::list<int> 
GraphAlgo::BfsEdges(DirectedGraph const& NextOf, int const src) {
    std::list<int> edges {};
    std::vector<bool> inQueue( NextOf.size(), false);
    std::queue<int> qu {};
    qu.emplace(src);
    while (!qu.empty()) {
        auto cur = qu.front();
        qu.pop();
        for (auto [to, no]: NextOf.at(cur)) {
            edges.emplace_back(no);
            if (!inQueue.at(to)) {
                inQueue.at(to)= true;
                qu.emplace(to);
            }
        }
    }
    return edges;
}

