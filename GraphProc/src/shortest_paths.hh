#ifndef __SHORTEST_PATH__

#include <vector>
#include <list>

struct DirectedEdge {
    int Dst;
    int No;
};

using DirectedGraph = std::vector< std::list<DirectedEdge> >; 

class GraphAlgo {
public:
    static std::list<int> 
    ShortestPaths(DirectedGraph const& NextOf, 
                  int const src, int const dst);
    
    static std::list<int> 
    BfsEdges(DirectedGraph const& NextOf, int const src);
};

#endif //__SHORTEST_PATH__

