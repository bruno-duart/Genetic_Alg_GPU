#include "graphs.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

// Construtor para criar o grafo a partir de um vetor de arestas e número de vértices
Graph::Graph(const std::vector<Edge> &edges, int n)
{
    adjList.resize(n);
    for (auto &edge : edges)
    {
        adjList[edge.src].push_back(edge.dest);
        adjList[edge.dest].push_back(edge.src); // Para grafos não direcionados
    }
}

// Construtor para criar o grafo a partir de um arquivo DIMACS
Graph::Graph(const std::string &filename)
{
    int n = 0, m = 0;
    std::vector<Edge> edges;

    // Abrir o arquivo para leitura
    std::ifstream infile(filename);
    if (!infile)
    {
        std::cerr << "Erro ao abrir o arquivo!" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(infile, line))
    {
        if (line[0] == 'p')
        {
            std::istringstream iss(line);
            std::string dummy;
            iss >> dummy >> dummy >> n >> m;
            adjList.resize(n);
            numVertices = n;
        }
        else if (line[0] == 'e')
        {
            std::istringstream iss(line);
            char dummy;
            int src, dest;
            iss >> dummy >> src >> dest;
            src--; // Ajuste para índices baseados em 0
            dest--;
            edges.push_back({src, dest});
        }
    }

    for (auto &edge : edges)
    {
        adjList[edge.src].push_back(edge.dest);
        adjList[edge.dest].push_back(edge.src); // Para grafos não direcionados
    }
    numEdges = edges.size();
}

// Método para verificar a existência de uma aresta
auto Graph::findEdgeIfExists(int nodeA, int nodeB)
{
    for (auto it = adjList[nodeA].begin(); it != adjList[nodeA].end(); ++it)
    {
        if (*it == nodeB)
        {
            return it;
        }
    }
    return adjList[nodeA].end(); // Indica que a aresta não existe
}

// int Graph::edgeExists(int nodeA, int nodeB){

// }

// Método para imprimir o grafo
void Graph::printGraph() const
{
    std::cout << "|V| : " << numVertices << '\n';
    std::cout << "|E| : " << numEdges << '\n';
    for (int i = 0; i < adjList.size(); ++i)
    {
        std::cout << i << "--> "; // Ajuste para exibir índices baseados em 1
        for (int v : adjList[i])
        {
            std::cout << v << " "; // Ajuste para exibir índices baseados em 1
        }
        std::cout << std::endl;
    }
}

int Graph::getNumVertices() const {
    return numVertices;
}

int Graph::getNumEdges() {
    return numEdges;
}

const std::vector<int>& Graph::getNeighbors(int idVertice) const{
    return adjList[idVertice];
}