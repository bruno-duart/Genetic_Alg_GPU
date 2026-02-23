#ifndef GPU_TRANSLATOR_H
#define GPU_TRANSLATOR_H

#include "graphs.h"
#include <vector>

// Estrutura que segura os dados prontos para a GPU
struct CSRGraph {
    int num_vertices;
    int num_edges;
    
    // Ponteiros para memória do DEVICE (GPU)
    int *d_row_offsets; 
    int *d_col_indices; 

    // Dados no HOST (CPU) 
    std::vector<int> h_row_offsets;
    std::vector<int> h_col_indices;
};

CSRGraph convertToCSR(const Graph& graph);
void uploadGraphToGPU(CSRGraph& csr);
void freeGraphOnGPU(CSRGraph& csr);

#endif