/**
 * @file gpu_translator.cu
 * @brief Implementação das funções de tradução e transferência de grafos para a GPU.
 */

#include "gpu_translator.h"
#include <cuda_runtime.h>
#include <iostream>

CSRGraph convertToCSR(const Graph& graph) {
    CSRGraph csr;
    csr.num_vertices = graph.getNumVertices();

    // O primeiro offset sempre começa em 0
    csr.h_row_offsets.push_back(0);

    int current_offset{0};
    // Itera sobre todos os vértices do grafo
    for (int i{0}; i < graph.getNumVertices(); ++i){
        // Recupera a lista de adjacência do vértice 'i'
        const std::vector<int>& neighbors = graph.getNeighbors(i);
        
        // Concatena os vizinhos no array plano (flattened array)
        for (int neighbor: neighbors){
            csr.h_col_indices.push_back(neighbor);
        }
        
        // Atualiza o offset somando a quantidade de vizinhos deste vértice.
        // Isso define onde a lista de vizinhos do PRÓXIMO vértice começará.
        current_offset += neighbors.size();
        csr.h_row_offsets.push_back(current_offset);
    }

    // O tamanho do array de índices corresponde ao total de conexões mapeadas.
    // Em um grafo não-direcionado, cada aresta (u, v) é representada duas vezes (u->v e v->u).
    csr.num_edges = csr.h_col_indices.size(); // Deve ser 2x o número de arestas (grafo não direcionado)

    return csr;
}

void uploadGraphToGPU(CSRGraph& csr) {
    // 1. Alocação de Memória no Device (VRAM)
    // d_row_offsets requer (V + 1) elementos
    cudaMalloc((void**)&csr.d_row_offsets, csr.h_row_offsets.size() * sizeof(int));
    // d_col_indices requer 2*E elementos
    cudaMalloc((void**)&csr.d_col_indices, csr.h_col_indices.size() * sizeof(int));

    // 2. Transferência de Dados (Host -> Device)
    // Bloqueia a CPU até que a cópia para a GPU seja finalizada de forma síncrona
    cudaMemcpy(csr.d_row_offsets, csr.h_row_offsets.data(), csr.h_row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csr.d_col_indices, csr.h_col_indices.data(), csr.h_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
}

void freeGraphOnGPU(CSRGraph& csr) {
    // Libera a memória previamente alocada na VRAM para evitar memory leaks
    cudaFree(csr.d_row_offsets);
    cudaFree(csr.d_col_indices);
}
