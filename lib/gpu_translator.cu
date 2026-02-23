#include "gpu_translator.h"
#include <cuda_runtime.h>
#include <iostream>

CSRGraph convertToCSR(const Graph& graph) {
    CSRGraph csr;
    csr.num_vertices = graph.getNumVertices();

    // 1. Construir Row Offsets e Column Indices
    csr.h_row_offsets.push_back(0);

    int current_offset{0};

    for (int i{0}; i < graph.getNumVertices(); ++i){
        const std::vector<int>& neighbors = graph.getNeighbors(i);
        
        // Copia os vizinhos para o vetor plano
        for (int neighbor: neighbors){
            csr.h_col_indices.push_back(neighbor);
        }
        
        // Atualiza o offset
        current_offset += neighbors.size();
        csr.h_row_offsets.push_back(current_offset);
    }

    csr.num_edges = csr.h_col_indices.size(); // Deve ser 2x o número de arestas (grafo não direcionado)

    return csr;
}

void uploadGraphToGPU(CSRGraph& csr) {
    // Aloca memória na GPU
    cudaMalloc((void**)&csr.d_row_offsets, csr.h_row_offsets.size() * sizeof(int));
    cudaMalloc((void**)&csr.d_col_indices, csr.h_col_indices.size() * sizeof(int));

    // Copia os dados da CPU para a GPU
    cudaMemcpy(csr.d_row_offsets, csr.h_row_offsets.data(), csr.h_row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csr.d_col_indices, csr.h_col_indices.data(), csr.h_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
}

void freeGraphOnGPU(CSRGraph& csr) {
    cudaFree(csr.d_row_offsets);
    cudaFree(csr.d_col_indices);
}
