/**
 * @file graphs.h
 * @brief Estruturas de dados e classes para representação de grafos em CPU.
 * * Define a estrutura básica de um Grafo não-direcionado utilizando Listas de Adjacência.
 * Fornece métodos para carregamento de instâncias padronizadas (DIMACS) e consultas básicas.
 */
#ifndef HEADER_GRAPHS
#define HEADER_GRAPHS

#include <iostream>
#include <vector>

/**
 * @struct Edge
 * @brief Representa uma aresta simples conectando dois vértices.
 */
struct Edge {
    int src;  ///< ID do vértice de origem (source)
    int dest; ///< ID do vértice de destino (destination)
};

/**
 * @class Graph
 * @brief Representação Orientada a Objetos de um Grafo Não-Direcionado.
 * * Armazena o grafo na memória principal (Host) utilizando um vetor de vetores 
 * (Lista de Adjacência). Ideal para manipulação dinâmica e inicialização, mas 
 * não otimizado para GPU (que requer arrays contíguos como o CSR).
 */
class Graph {
    int numVertices; ///< Número total de vértices (|V|).
    int numEdges;    ///< Número total de arestas (|E|).

public:
    /** * @brief Lista de Adjacência do grafo.
     * O índice do vetor externo representa o ID do vértice, e o vetor interno 
     * contém os IDs de todos os seus vizinhos.
     */
    std::vector<std::vector<int>> adjList;
    
    /**
     * @brief Construtor que inicializa o grafo a partir de uma lista de arestas.
     * @param edges Vetor constante de arestas (Edge).
     * @param n Número total de vértices.
     */
    Graph(const std::vector<Edge> &edges, int n) ;

    /**
     * @brief Construtor que carrega o grafo a partir de um arquivo no formato DIMACS.
     * Lê o arquivo de texto iterativamente. Trata a linha 'p' para configuração de 
     * tamanho e linhas 'e' para as arestas. Faz a conversão automática de índices 
     * 1-based (DIMACS) para 0-based (C++).
     * @param filename Caminho para o arquivo da instância (ex: "instances/le450_15a.col").
     */
    Graph(const std::string &filename) ;

    /**
     * @brief Verifica se uma aresta direta existe entre dois vértices.
     * @param nodeA Vértice de origem.
     * @param nodeB Vértice de destino.
     * @return auto Iterador para a posição do vizinho na lista de adjacência, 
     * ou adjList[nodeA].end() se a aresta não existir.
     */
    auto findEdgeIfExists(int nodeA, int nodeB) ;

    /**
     * @brief Imprime a estrutura do grafo no console (apenas para debug).
     */
    void printGraph() const ;

    int getNumVertices() const;
    int getNumEdges();

    /**
     * @brief Obtém a lista de adjacência (vizinhos) de um vértice específico.
     * Retorna por referência constante para evitar cópias de memória.
     * @param idVertice ID do vértice consultado.
     * @return const std::vector<int>& Referência ao vetor de vizinhos.
     */
    const std::vector<int>& getNeighbors(int idVertice) const;
};

#endif