#ifndef HEADER_GRAPHS
#define HEADER_GRAPHS

#include <iostream>
#include <vector>

struct Edge {
    int src;
    int dest;
};

class Graph {
    int numVertices;
    int numEdges;

public:
    std::vector<std::vector<int>> adjList;
    // Construtor para criar o grafo a partir de um vetor de arestas e número de vértices
    Graph(const std::vector<Edge> &edges, int n) ;

    // Construtor para criar o grafo a partir de um arquivo DIMACS
    Graph(const std::string &filename) ;

    // Método para verificar a existência de uma aresta
    auto findEdgeIfExists(int nodeA, int nodeB) ;

    // Método para imprimir o grafo
    void printGraph() const ;

    int getNumVertices() const;
    int getNumEdges();

    const std::vector<int>& getNeighbors(int idVertice) const;
};

#endif