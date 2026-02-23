/**
 * @file graphs.cpp
 * @brief Implementação dos métodos da classe Graph e parseamento de arquivos DIMACS.
 */
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
    // Alimenta a lista de adjacência de forma simétrica (Grafo Não-Direcionado)
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
    // Percorre o arquivo linha por linha
    while (std::getline(infile, line))
    {
        // Linha 'p': Configuração do Problema (ex: p edge 450 8168)
        if (line[0] == 'p')
        {
            std::istringstream iss(line);
            std::string dummy;
            // Descarta 'p' e 'edge', guarda n (|V|) e m (|E|)
            iss >> dummy >> dummy >> n >> m;
            adjList.resize(n);
            numVertices = n;
        }
        // Linha 'e': Definição de Aresta (ex: e 1 5)
        else if (line[0] == 'e')
        {
            std::istringstream iss(line);
            char dummy;
            int src, dest;
            iss >> dummy >> src >> dest;

            // Ajuste crucial: Instâncias DIMACS começam a contar do vértice 1.
            // O C++ requer arrays indexados a partir do 0.
            src--; // Ajuste para índices baseados em 0
            dest--;
            edges.push_back({src, dest});
        }
    }

    // Processa o vetor temporário de arestas para construir a Lista de Adjacência simétrica
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
    // Realiza uma busca linear simples na lista de vizinhos de nodeA
    for (auto it = adjList[nodeA].begin(); it != adjList[nodeA].end(); ++it)
    {
        if (*it == nodeB)
        {
            return it;
        }
    }
    // Retorna o iterador 'end' para indicar que o alvo não foi encontrado
    return adjList[nodeA].end(); // Indica que a aresta não existe
}

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