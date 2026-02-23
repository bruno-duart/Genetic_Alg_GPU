#ifndef HEADER_GENETIC_GPU
#define HEADER_GENETIC_CPU


#include "gpu_translator.h"
#include <vector>

class GpuPopulation {
private:
    int pop_size;
    int num_vertices;
    
    int* d_genes; 
    int* d_fitness; 
    void* d_rng_states; 
    int* d_genes_next; 

public:
    GpuPopulation(int p_size, int n_verts); 
    
    ~GpuPopulation();

    // Inicialização de população aleatória
    void initializeRandomPopulation(const int num_colors);

    // Métodos de ajuda
    void setGenesFromVector(const std::vector<int>& host_genes); 
    void getGenesToVector(std::vector<int>& host_genes) const;
    void getFitnessToVector(std::vector<int>& host_fitness) const;
    // Cálculo de Fitness na GPU
    void evaluateFitness(const CSRGraph &graph);

    void swapPopulation();
    void preserveElite(int num_vertices);
    void evolveGeneration(const CSRGraph& graph, float mutation_rate, int num_colors);
    void evolveGenerationHeritage(const CSRGraph& graph, float mutation_rate, int num_colors);
    
    int* getGenesPtr() const { return d_genes; }
    int* getFitnessPtr() const { return d_fitness; }

};

#endif