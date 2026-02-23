/**
 * @file main.cpp
 * @brief Ponto de entrada do Algoritmo Genético em GPU para o Problema de Coloração de Grafos.
 * * Este arquivo orquestra a leitura da instância, alocação de memória na GPU (CSR),
 * inicialização da população e o loop evolutivo principal (Avaliação, Seleção, 
 * Crossover, Mutação e Elitismo) juntamente com as condições de parada.
 */

#include "graphs.h"
#include "gpu_translator.h"
#include "genetic_gpu.cuh"
#include "solution.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>

/** @brief Tamanho fixo da população. Recomenda-se múltiplos de 256 (tamanho do bloco CUDA). */
const int POP_SIZE = 2048;
/** @brief Número máximo de gerações permitidas antes da parada do algoritmo. */
const int MAX_GENERATIONS = 5000;
/** @brief Probabilidade de mutação aplicada a cada gene (vértice) durante a reprodução. */
const float MUTATION_RATE = 0.05f;

int main(int argc, char *argv[])
{

    // 1. Lendo argumentos
    if (argc < 4)
    {
        std::cerr << "Uso ./main <arquivo_col> <num_cores> <seed>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int num_colors = std::stoi(argv[2]);
    unsigned long seed_val = std::stoul(argv[3]);
    
    std::cout << "Carregando Grafo: " << filename << std::endl;
    Graph graph(filename);
    std::mt19937 rng(seed_val);

    std::cout << "Convertendo para CSR e enviando para GPU..." << std::endl;
    CSRGraph csr = convertToCSR(graph);
    uploadGraphToGPU(csr);

    // Instancia o Gerenciador GPU
    GpuPopulation gpuPop(POP_SIZE, csr.num_vertices);

    std::cout << "Gerando população inicial (" << POP_SIZE << " indivíduos)..." << std::endl;
    gpuPop.initializeRandomPopulation(num_colors);

    // Vetor auxiliar para ler o fitness de volta na CPU (para estatísticas)
    std::vector<int> host_fitness_buffer;

    // ---- LOOP EVOLUTIVO ----
    std::cout << "Iniciando evolução por " << MAX_GENERATIONS << " gerações..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // CONFIGURAÇÃO DO TIMEOUT INTERNO (10s a menos que o o Python que mata em 180s)
    const double TIME_LIMIT_SEC = 170.0;
    int best_fit_global = 999999;
    int gen_found_best = -1;
    int gen{0};

    for (; gen < MAX_GENERATIONS; ++gen)
    {
        // Avaliar população atual na GPU
        gpuPop.evaluateFitness(csr);

        if (gen % 100 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> current_elapsed = current_time - start_time;

            if (current_elapsed.count() > TIME_LIMIT_SEC) 
            {
                // Salva o melhor fitness atual antes de sair
                gpuPop.getFitnessToVector(host_fitness_buffer);
                int current_best = *std::min_element(host_fitness_buffer.begin(), host_fitness_buffer.end());
                if (current_best < best_fit_global) best_fit_global = current_best;

                // Imprime resultado parcial usando 'gen' atual
                std::cout << "CSV_RESULT;" << best_fit_global << ";" << current_elapsed.count() << ";" << gen << std::endl;
                
                freeGraphOnGPU(csr);
                return 0; // Sai com sucesso
            }
        }

        // Relatório a cada 100 gerações (processo lento)
        if (gen % 500 == 0 || gen == MAX_GENERATIONS - 1)
        {
            gpuPop.getFitnessToVector(host_fitness_buffer);

            // Encontra o melhor fitness
            int current_best = *std::min_element(host_fitness_buffer.begin(), host_fitness_buffer.end());

            if (current_best < best_fit_global) 
                best_fit_global = current_best;

            // Critério de Parada Antecipada
            if (best_fit_global == 0)
            {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_to_solution = now - start_time;

                std::cout << "CSV_RESULT;" << best_fit_global << ";" << time_to_solution.count() << ";" << gen << std::endl;

                freeGraphOnGPU(csr);
                return 0;
            }
        }

        // Evolução (Seleção + Crossover + Mutação)
        gpuPop.evolveGenerationHeritage(csr, MUTATION_RATE, num_colors);
        // Preserva o melhor da geração atual
        gpuPop.preserveElite(csr.num_vertices);
        // Troca buffers
        gpuPop.swapPopulation();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // --- RESULTADOS FINAIS ---
    // Garante que o fitness final está atualizado
    gpuPop.evaluateFitness(csr);
    gpuPop.getFitnessToVector(host_fitness_buffer);
    int final_best = *std::min_element(host_fitness_buffer.begin(), host_fitness_buffer.end());

    if (final_best < best_fit_global) {
        best_fit_global = final_best;
    }

    // 4. SAÍDA em FORMATO: CSV_RESULT;fitness;tempo_segundos;generacoes
    std::cout << "CSV_RESULT;" << best_fit_global << ";" << elapsed.count() << ";" << gen << std::endl;
    // --- 6. LIMPEZA ---
    freeGraphOnGPU(csr);
}

// nvcc -Wno-deprecated-gpu-targets main.cpp lib/graphs.cpp lib/solution.cpp lib/utils.cpp lib/gpu_translator.cu lib/genetic_gpu.cu -Iinclude/ -o main
// ./main le450_15a.col 15 123456