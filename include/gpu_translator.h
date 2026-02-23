/**
 * @file gpu_translator.h
 * @brief Definições e estruturas para conversão de grafos para a GPU.
 * * Este arquivo contém a estrutura CSRGraph e as assinaturas das funções
 * responsáveis por traduzir a representação em Lista de Adjacência do C++ 
 * para o formato CSR (Compressed Sparse Row), que é altamente otimizado 
 * para leituras paralelas na memória da GPU.
 */
#ifndef GPU_TRANSLATOR_H
#define GPU_TRANSLATOR_H

#include "graphs.h"
#include <vector>

/**
 * @struct CSRGraph
 * @brief Representa um grafo no formato Compressed Sparse Row (CSR).
 * * O formato CSR lineariza a lista de adjacência em dois arrays principais:
 * `row_offsets` e `col_indices`. Isso permite acesso rápido aos vizinhos
 * de um vértice e minimiza o uso de memória, ideal para arquiteturas CUDA.
 */
struct CSRGraph {
    int num_vertices; ///< Número total de vértices no grafo (|V|).
    int num_edges;///< Número total de arestas direcionadas (2x arestas para grafos não-direcionados).
    
    // --- Ponteiros para memória do DEVICE (GPU) ---
    /** * @brief Ponteiro na GPU para o array de offsets.
     * O tamanho é (|V| + 1). Indica onde começa e termina a lista de vizinhos do vértice 'i'.
     * Vizinhos do vértice 'i' estão no intervalo [d_row_offsets[i], d_row_offsets[i+1] - 1].
     */
    int *d_row_offsets; 

    /** * @brief Ponteiro na GPU para o array de índices de colunas (vizinhos).
     * O tamanho é igual a `num_edges`. Contém os IDs contíguos dos vértices adjacentes.
     */
    int *d_col_indices; 

    // Dados no HOST (CPU) 
    std::vector<int> h_row_offsets; ///< Cópia do array de offsets na memória RAM.
    std::vector<int> h_col_indices; ///< Cópia do array de índices na memória RAM.
};

/**
 * @brief Converte um grafo da representação OO (Lista de Adjacência) para o formato CSR.
 * * Esta função percorre o grafo original e constrói os arrays `h_row_offsets` e 
 * `h_col_indices` na memória do Host (CPU). Os ponteiros de Device ainda não são alocados.
 * * @param graph Objeto Graph constante contendo a estrutura original.
 * @return CSRGraph Estrutura preenchida com os dados CSR na CPU.
 */
CSRGraph convertToCSR(const Graph& graph);

/**
 * @brief Aloca memória e envia os arrays CSR da CPU (Host) para a GPU (Device).
 * * Utiliza `cudaMalloc` para alocar memória VRAM correspondente aos arrays CSR
 * e `cudaMemcpy` (HostToDevice) para transferir os dados das std::vectors
 * `h_row_offsets` e `h_col_indices` para `d_row_offsets` e `d_col_indices`.
 * * @param csr Referência para a estrutura CSRGraph já preenchida na CPU.
 */
void uploadGraphToGPU(CSRGraph& csr);

/**
 * @brief Libera a memória VRAM alocada para o grafo na GPU.
 * * Executa `cudaFree` nos ponteiros de Device. Deve ser chamada ao final do 
 * programa para evitar vazamento de memória de vídeo (VRAM leak).
 * * @param csr Referência para a estrutura CSRGraph cujos ponteiros de Device serão liberados.
 */
void freeGraphOnGPU(CSRGraph& csr);

#endif // GPU_TRANSLATOR_H