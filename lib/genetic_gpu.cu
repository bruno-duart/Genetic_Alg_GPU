#include "genetic_gpu.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

__global__ void init_rng(curandState *states, int n, unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

GpuPopulation::GpuPopulation(int p_size, int n_verts)
    : pop_size{p_size}, num_vertices{n_verts}
{
    // Alocação de memória
    cudaMalloc((void **)&d_fitness, pop_size * (sizeof(int)));
    cudaMalloc((void **)&d_genes, pop_size * num_vertices * sizeof(int));
    cudaMalloc((void **)&d_genes_next, pop_size * num_vertices * sizeof(int));
    cudaMalloc((void **)&d_rng_states, pop_size * sizeof(curandState));

    // Geometria lançamento
    int threadsPerBlock = 256;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    // Inicialiando RNG
    init_rng<<<blocksPerGrid, threadsPerBlock>>>((curandState *)d_rng_states, pop_size, time(NULL));

    // Sincronizando para garantir que GPU finalizou
    cudaDeviceSynchronize();
}

// Destrutor: Libera tudo (cudaFree)
GpuPopulation::~GpuPopulation()
{
    cudaFree((void **)d_fitness);
    cudaFree((void **)d_genes);
    cudaFree((void **)d_genes_next);
    cudaFree((void **)d_rng_states);
}

__global__ void initialize_random_population_kernel(
    int *pop,
    curandState *rng_states,
    int pop_size,
    int num_vertices,
    int num_colors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size)
        return;

    curandState localState = rng_states[idx];
    int *my_genes = &pop[idx * num_vertices];

    for (int i{0}; i < num_vertices; ++i)
    {
        my_genes[i] = (int)((curand_uniform(&localState)) * (num_colors - 0.001f));
    }

    rng_states[idx] = localState;
}

void GpuPopulation::initializeRandomPopulation(const int num_colors)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    initialize_random_population_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_genes,
        (curandState *)d_rng_states,
        pop_size,
        num_vertices,
        num_colors);
    cudaDeviceSynchronize();
}

void GpuPopulation::setGenesFromVector(const std::vector<int> &host_genes)
{
    if (host_genes.size() != (pop_size * num_vertices))
    {
        std::cerr << "Erro: Tamanho do vetor incorreto!" << std::endl;
        return;
    }

    cudaMemcpy(d_genes, host_genes.data(), host_genes.size() * sizeof(int), cudaMemcpyHostToDevice);
}
void GpuPopulation::getGenesToVector(std::vector<int> &host_genes) const
{
    host_genes.resize(pop_size * num_vertices); // Garante que cabe
    cudaMemcpy(host_genes.data(), d_genes,
               pop_size * num_vertices * sizeof(int),
               cudaMemcpyDeviceToHost);
}
void GpuPopulation::getFitnessToVector(std::vector<int> &host_fitness) const
{
    host_fitness.resize(pop_size);
    cudaMemcpy(host_fitness.data(), d_fitness,
               pop_size * sizeof(int),
               cudaMemcpyDeviceToHost);
}

__global__ void evaluate_fitness_kernel(
    int *genes,
    int *fitness,
    const int *row_offsets,
    const int *col_indices,
    int num_vertices,
    int num_indv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_indv)
        return;

    int *my_genes = &genes[idx * num_vertices];

    int conflicts{0};

    for (int u{0}; u < num_vertices; ++u)
    {
        int color_u = my_genes[u];

        int start_index = row_offsets[u];
        int end_index = row_offsets[u + 1];

        for (int k{start_index}; k < end_index; ++k)
        {
            int v = col_indices[k];

            if (my_genes[v] == color_u)
            {
                conflicts++;
            }
        }
    }

    fitness[idx] = conflicts / 2;
}

void GpuPopulation::evaluateFitness(const CSRGraph &graph)
{
    int threadsPerBlock{256};
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    evaluate_fitness_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_genes,
        d_fitness,
        graph.d_row_offsets,
        graph.d_col_indices,
        num_vertices,
        pop_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Erro no Kernel Fitness: " << cudaGetErrorString(err) << std::endl;
    }

    // Espera a GPU terminar antes de deixar a CPU continuar
    cudaDeviceSynchronize();
}

void GpuPopulation::swapPopulation()
{
    std::swap(d_genes, d_genes_next);
}

__device__ int tournament_selection(curandState *localState, int *fitness, int pop_size, int k = 3)
{
    int best_idx = -1;
    int best_val = 99999999; // Inf

    for (int i{0}; i < k; ++i)
    {
        // Sorteia um índice aleatório da população
        int rand_idx = curand_uniform(localState) * (pop_size - 1);

        // Verifica o melhor
        int val = fitness[rand_idx];
        if (val < best_val)
        {
            best_val = val;
            best_idx = rand_idx;
        }
    }
    return best_idx;
}

__global__ void evolve_kernel(
    int *current_pop,        // Leitura
    int *next_pop,           // Escrita
    int *fitness,            // Para o torneio
    curandState *rng_states, // RNG states
    int pop_size,
    int num_vertices,
    int num_colors,
    float mutation_rate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size)
        return;

    // 1. Carrega o estado RNG da thread
    curandState localState = rng_states[idx];

    // 2. Seleção: Escolhe Pai e Mãe via Torneio
    int parent1_idx = tournament_selection(&localState, fitness, pop_size);
    int parent2_idx = tournament_selection(&localState, fitness, pop_size);

    // Ponteiros para os genes dos pais
    int *p1_genes = &current_pop[parent1_idx * num_vertices];
    int *p2_genes = &current_pop[parent2_idx * num_vertices];

    // Ponteiro para descendente
    int *genes = &next_pop[idx * num_vertices];

    // 3. Crossover e Mutação (Gene a Gene)
    for (int i{0}; i < num_vertices; ++i)
    {
        int gene_val;

        if (curand_uniform(&localState) > 0.5f)
        {
            gene_val = p1_genes[i];
        }
        else
        {
            gene_val = p2_genes[i];
        }

        // Mutanção: Pequena chance de trocar a cor
        if (curand_uniform(&localState) < mutation_rate)
        {
            // Sorteia nova cor
            gene_val = (int)(curand_uniform(&localState) * (num_colors - 0.001f));
        }

        // Escreve no novo indivíduo
        genes[i] = gene_val;
    }

    // 4. Salva o estado do RNG atualizado para a próxima geração
    rng_states[idx] = localState;
}

__global__ void evolve_kernel_single_point_crossover(
    int *current_pop,        // Leitura
    int *next_pop,           // Escrita
    int *fitness,            // Para o torneio
    curandState *rng_states, // RNG states
    int pop_size,
    int num_vertices,
    int num_colors,
    float mutation_rate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size)
        return;

    // 1. Carrega o estado RNG da thread
    curandState localState = rng_states[idx];

    // 2. Seleção: Escolhe Pai e Mãe via Torneio
    int parent1_idx = tournament_selection(&localState, fitness, pop_size);
    int parent2_idx = tournament_selection(&localState, fitness, pop_size);

    // Ponteiros para os genes dos pais
    int *p1_genes = &current_pop[parent1_idx * num_vertices];
    int *p2_genes = &current_pop[parent2_idx * num_vertices];

    // Ponteiro para descendente
    int *genes = &next_pop[idx * num_vertices];

    // 3. Single Point Crossover e Mutação (Gene a Gene)
    int single_point_x = (int)curand_uniform(&localState) * num_vertices;

    for (int i{0}; i < num_vertices; ++i)
    {
        int gene_val;

        if (i < single_point_x)
        {
            gene_val = p1_genes[i];
        }
        else
        {
            gene_val = p2_genes[i];
        }

        // Mutação: Pequena chance de trocar a cor
        if (curand_uniform(&localState) < mutation_rate)
        {
            // Sorteia nova cor
            gene_val = (int)(curand_uniform(&localState) * (num_colors - 0.001f));
        }

        // Escreve no novo indivíduo
        genes[i] = gene_val;
    }

    // 4. Salva o estado do RNG atualizado para a próxima geração
    rng_states[idx] = localState;
}

__global__ void evolve_kernel_heritage_crossover(
    int *current_pop, // Leitura
    int *next_pop,    // Escrita
    int *fitness,     // Para o torneio
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    curandState *rng_states, // RNG states
    int pop_size,
    int num_vertices,
    int num_colors,
    float mutation_rate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size)
        return;

    // 1. Carrega o estado RNG da thread
    curandState localState = rng_states[idx];

    // 2. Seleção: Escolhe Pai e Mãe via Torneio
    int parent1_idx = tournament_selection(&localState, fitness, pop_size);
    int parent2_idx = tournament_selection(&localState, fitness, pop_size);

    // Ponteiros para os genes dos pais
    int *p1_genes = &current_pop[parent1_idx * num_vertices];
    int *p2_genes = &current_pop[parent2_idx * num_vertices];

    // Ponteiro para descendente
    int *genes = &next_pop[idx * num_vertices];

    // 3. Single Point Crossover e Mutação (Gene a Gene)
    int single_point_x = (int)curand_uniform(&localState) * num_vertices;

    for (int i{0}; i < num_vertices; ++i)
    {
        int color_p1 = p1_genes[i];
        int color_p2 = p2_genes[i];
        int gene_val;

        if (color_p1 == color_p2)
        {
            gene_val = color_p1;
        }
        else
        {
            // As cores divergem
            bool p1_conflicts = false;
            bool p2_conflicts = false;

            int start = row_offsets[i];
            int end = row_offsets[i + 1];

            // Verifica Pai 1
            for (int k = start; k < end; ++k)
            {
                if (p1_genes[col_indices[k]] == color_p1)
                {
                    p1_conflicts = true;
                    break; // Já achou um erro, para de procurar
                }
            }

            // Verifica Pai 2 (Se P1 já falhou, P2 tem chance de salvar)
            // Otimização: Se P1 não falhou, nem precisaria ver P2 estritamente,
            // mas vamos ver para desempatar
            for (int k = start; k < end; ++k)
            {
                if (p2_genes[col_indices[k]] == color_p2)
                {
                    p2_conflicts = true;
                    break;
                }
            }

            // Tomada de Decisão
            if (!p1_conflicts && p2_conflicts)
            {
                gene_val = color_p1; // Pai 1 ganha
            }
            else if (p1_conflicts && !p2_conflicts)
            {
                gene_val = color_p2; // Pai 2 ganha
            }
            else
            {
                // Ambos ruins ou ambos bons: Sorteio (ou mantém Single Point aqui se preferir)
                if (curand_uniform(&localState) > 0.5f)
                    gene_val = color_p1;
                else
                    gene_val = color_p2;
            }
        }

        // Mutação: Pequena chance de trocar a cor
        if (curand_uniform(&localState) < mutation_rate)
        {
            // Sorteia nova cor
            gene_val = (int)(curand_uniform(&localState) * (num_colors - 0.001f));
        }

        // Escreve no novo indivíduo
        genes[i] = gene_val;
    }

    // 4. Salva o estado do RNG atualizado para a próxima geração
    rng_states[idx] = localState;
}

// Método Wrapper na Classe
void GpuPopulation::evolveGeneration(const CSRGraph &graph, float mutation_rate, int num_colors)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    evolve_kernel_single_point_crossover<<<blocksPerGrid, threadsPerBlock>>>(
        d_genes,
        d_genes_next,
        d_fitness,
        (curandState *)d_rng_states,
        pop_size,
        num_vertices,
        num_colors,
        mutation_rate);
    cudaDeviceSynchronize();
}

void GpuPopulation::evolveGenerationHeritage(const CSRGraph &graph, float mutation_rate, int num_colors)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    evolve_kernel_heritage_crossover<<<blocksPerGrid, threadsPerBlock>>>(
        d_genes,
        d_genes_next,
        d_fitness,
        graph.d_row_offsets,
        graph.d_col_indices,
        (curandState *)d_rng_states,
        pop_size,
        num_vertices,
        num_colors,
        mutation_rate);
    cudaDeviceSynchronize();
}

void GpuPopulation::preserveElite(int num_vertices)
{
    // 1. "Envelopa" o ponteiro bruto da GPU para o Thrust entender
    thrust::device_ptr<int> dt_fitness(d_fitness);

    // 2. Encontra o iterador (ponteiro) para o menor valor (melhor fitness)
    thrust::device_ptr<int> min_ptr = thrust::min_element(dt_fitness, dt_fitness + pop_size);

    // 3. Calcula o índice (distância do início)
    int best_idx = min_ptr - dt_fitness;

    // 4. Copia os genes do ELITE (current) para a posição 0 do NEXT
    // Sobrescreve o primeiro filho, garantindo a imortalidade do melhor encontrado
    cudaMemcpy(
        &d_genes_next[0],
        &d_genes[best_idx * num_vertices],
        num_vertices * sizeof(int),
        cudaMemcpyDeviceToDevice);
}