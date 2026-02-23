/**
 * @file genetic_gpu.cuh
 * @brief Declaração da classe GpuPopulation para execução de Algoritmo Genético em CUDA.
 */
#ifndef HEADER_GENETIC_GPU
#define HEADER_GENETIC_CPU

#include "gpu_translator.h"
#include <vector>

/**
 * @class GpuPopulation
 * @brief Gerencia a população, memória de vídeo e kernels do Algoritmo Genético na GPU.
 * * Utiliza o padrão de "Double Buffering" (Ping-Pong) com `d_genes` e `d_genes_next` 
 * para alternar entre as gerações atuais e futuras, evitando condição de corrida (race condition) 
 * e cópias de memória onerosas entre Host e Device.
 */
class GpuPopulation {
private:
    int pop_size;      ///< Tamanho da população (número de indivíduos).
    int num_vertices;  ///< Número de vértices do grafo.
    
    // --- Ponteiros de Device (Memória da GPU) ---
    int* d_genes;        ///< Vetor linearizado da população atual (tamanho: pop_size * num_vertices).
    int* d_fitness;      ///< Vetor contendo o fitness (conflitos) de cada indivíduo (tamanho: pop_size).
    void* d_rng_states;  ///< Estados do gerador de números aleatórios do cuRAND (opaco como void* no header).
    int* d_genes_next;   ///< Buffer auxiliar para armazenar a próxima geração de filhos.

public:
    /**
     * @brief Construtor da classe. Aloca memória na VRAM e inicializa os estados do cuRAND.
     * @param p_size Tamanho da população desejada.
     * @param n_verts Número de vértices da instância de grafo atual.
     */
    GpuPopulation(int p_size, int n_verts); 

    /**
     * @brief Destrutor. Libera toda a memória de vídeo alocada.
     */
    ~GpuPopulation();

    /**
     * @brief Inicializa a população (d_genes) com cores aleatórias diretamente na GPU.
     * @param num_colors Número de cores (k) permitidas.
     */
    void initializeRandomPopulation(const int num_colors);

    // --- Métodos Auxiliares de Transferência de Memória ---

    /**
     * @brief Copia um vetor de genes do Host (CPU) para o Device (GPU).
     * @param host_genes Referência constante para o std::vector contendo a população inicial.
     */
    void setGenesFromVector(const std::vector<int>& host_genes); 
    /**
     * @brief Copia a população atual do Device (GPU) de volta para o Host (CPU).
     * @param host_genes Referência para o std::vector que receberá os dados.
     */
    void getGenesToVector(std::vector<int>& host_genes) const;
    /**
     * @brief Copia o vetor de fitness do Device (GPU) para o Host (CPU).
     * @param host_fitness Referência para o std::vector que receberá os valores de fitness.
     */
    void getFitnessToVector(std::vector<int>& host_fitness) const;
    
    // --- Métodos Principais do Algoritmo Genético ---

    /**
     * @brief Avalia o fitness (contagem de conflitos) de todos os indivíduos em paralelo.
     * @param graph O grafo estruturado no formato CSR (alocado na GPU).
     */
    void evaluateFitness(const CSRGraph &graph);

    /**
     * @brief Alterna os ponteiros da população atual e da próxima geração (Double Buffering).
     */
    void swapPopulation();

    /**
     * @brief Aplica Elitismo usando a biblioteca Thrust para redução paralela.
     * @param num_vertices Necessário para calcular o offset de memória do indivíduo.
     */
    void preserveElite(int num_vertices);
    /**
     * @brief Gera uma nova população usando Single Point Crossover e Mutação padrão.
     * @param graph Grafo CSR (não utilizado neste crossover específico, mas mantido na assinatura).
     * @param mutation_rate Chance de um gene sofrer mutação (0.0 a 1.0).
     * @param num_colors Número de cores (k) para a mutação aleatória.
     */
    void evolveGeneration(const CSRGraph& graph, float mutation_rate, int num_colors);
    /**
     * @brief Gera uma nova população usando o Crossover de Herança de Sucesso (Heurístico).
     * Analisa o grafo para priorizar a herança de cores que não geram conflito.
     * @param graph Grafo CSR (utilizado para verificar conflitos dos pais).
     * @param mutation_rate Chance de mutação.
     * @param num_colors Número de cores.
     */
    void evolveGenerationHeritage(const CSRGraph& graph, float mutation_rate, int num_colors);
    
    // --- Getters ---
    int* getGenesPtr() const { return d_genes; }
    int* getFitnessPtr() const { return d_fitness; }
};

#endif