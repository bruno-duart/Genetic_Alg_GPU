/**
 * @file solution.h
 * @brief Estruturas e funções para manipulação e avaliação de soluções (Indivíduos) na CPU.
 * * Este arquivo define como uma solução para o Problema de Coloração de Grafos (GCP)
 * é representada, avaliada e modificada no Host.
 */

#ifndef HEADER_SOLUTION
#define HEADER_SOLUTION

#include "graphs.h"
#include <iostream>
#include <vector>
#include <random>

/** * @typedef Individual
 * @brief Representa um indivíduo (solução). O índice é o vértice, o valor é a cor.
 */
using Individual = std::vector<int>;
/** * @typedef Fitness
 * @brief Representa a qualidade da solução (número de arestas em conflito).
 */
using Fitness = int;

/**
 * @struct NeighborMove
 * @brief Representa um movimento na vizinhança de busca local (mudança de cor de um vértice).
 */
struct NeighborMove {
    int vertex_id; ///< ID do vértice que terá a cor alterada.
    int new_color; ///< Nova cor a ser atribuída ao vértice.
    int delta_E;   ///< Variação na função objetivo (fitness) caso o movimento seja aplicado.
};

/**
 * @brief Calcula o fitness total (conflitos) de um indivíduo do zero.
 * @param graph Estrutura do grafo.
 * @param indv O indivíduo a ser avaliado.
 * @param fitness Referência onde o valor do fitness calculado será armazenado.
 */
void evaluate_fitness(const Graph &graph, const Individual& indv, Fitness& fitness);

/**
 * @brief Calcula o fitness total e popula um vetor com o número de conflitos por vértice.
 * @param graph Estrutura do grafo.
 * @param indv O indivíduo a ser avaliado.
 * @param fitness Referência onde o valor do fitness calculado será armazenado.
 * @param conflicts_per_vertex Vetor preenchido com a contagem de conflitos para cada vértice.
 */
void evaluate_fitness_incremental(const Graph &graph, const Individual &indv, Fitness &fitness, std::vector<int> &conflicts_per_vertex);

/**
 * @brief Preenche um indivíduo existente com cores aleatórias.
 * @param num_color Número total de cores disponíveis (k).
 * @param graph Estrutura do grafo.
 * @param indv Referência do indivíduo que será modificado.
 * @param rng Gerador de números aleatórios.
 */
void random_individual(int num_color, const Graph &graph, Individual& indv, std::mt19937 &rng);

/**
 * @brief Copia os dados (genes e fitness) de um indivíduo para outro.
 * @param from Indivíduo de origem.
 * @param fit_from Fitness do indivíduo de origem.
 * @param to Indivíduo de destino.
 * @param fit_to Variável que receberá o fitness do indivíduo de destino.
 */
void copy_individual(const Individual &from, const Fitness& fit_from, Individual &to, Fitness& fit_to);

/**
 * @brief Instancia e retorna um novo indivíduo com cores aleatórias.
 * @param num_color Número total de cores disponíveis.
 * @param graph Estrutura do grafo.
 * @param rng Gerador de números aleatórios.
 * @return Individual O novo indivíduo gerado.
 */
Individual initialize_individual(int num_color, const Graph &graph, std::mt19937 &rng);

/**
 * @brief Imprime o array de genes (cores) do indivíduo no console.
 * @param indv Indivíduo a ser impresso.
 */
void individual_toString(const Individual& indv); 

/**
 * @brief Imprime os genes e o valor de fitness do indivíduo no console.
 * @param indv Indivíduo a ser impresso.
 * @param fit Fitness do indivíduo.
 */
void print_individual(const Individual& indv, const Fitness& fit);

/**
 * @brief Calcula a variação de fitness (delta) se um vértice mudar de cor.
 * Utilizado para avaliação rápida em algoritmos de busca local.
 * @param graph Estrutura do grafo.
 * @param indv Indivíduo base.
 * @param vertex Vértice alvo da mudança.
 * @param new_color Nova cor proposta para o vértice.
 * @return int A diferença de conflitos (valores negativos indicam melhora).
 */
int compute_fitness_change(const Graph &graph, const Individual &indv, int vertex, int new_color);

/**
 * @brief Calcula a variação de fitness se dois vértices trocarem de cores entre si.
 * @param graph Estrutura do grafo.
 * @param indv Indivíduo base.
 * @param u Vértice 1.
 * @param v Vértice 2.
 * @return int A diferença de conflitos resultante do swap.
 */
int compute_swap_fitness_change(const Graph &graph, const Individual &indv, int u, int v);

/**
 * @brief Encontra o vértice com o maior número de conflitos (arestas violadas) na solução atual.
 * @param indv Indivíduo a ser analisado.
 * @param graph Estrutura do grafo.
 * @return int O ID do vértice mais conflituoso.
 */
int find_most_conflicted_vertex(const Individual &indv, const Graph &graph);

/**
 * @brief Encontra o segundo vértice mais conflituoso na solução atual.
 * @param indv Indivíduo a ser analisado.
 * @param graph Estrutura do grafo.
 * @param most_conflicted ID do vértice mais conflituoso (para ser ignorado na busca).
 * @return int O ID do segundo vértice mais conflituoso.
 */
int find_second_most_conflicted_vertex(Individual &indv, Graph &graph, int most_conflicted);

/**
 * @brief Gera um vizinho aleatório alterando a cor de um único vértice.
 * Copia o indivíduo base e aplica um movimento aleatório, atualizando o fitness de forma incremental.
 * @param new_indv Indivíduo que receberá a nova solução vizinha.
 * @param indv Indivíduo base.
 * @param new_fit Referência que receberá o fitness do vizinho.
 * @param fit Fitness da solução base.
 * @param graph Estrutura do grafo.
 * @param num_colors Número de cores disponíveis.
 * @param rng Gerador de números aleatórios.
 */
void explore_neighborhood(Individual& new_indv, const Individual& indv, Fitness& new_fit, const Fitness& fit, const Graph& graph, int num_colors, std::mt19937& rng);

/**
 * @brief Verifica rapidamente se um vértice possui pelo menos um vizinho com a mesma cor.
 * @param graph Estrutura do grafo.
 * @param indv Indivíduo (solução).
 * @param vertex_id Vértice a ser verificado.
 * @return true Se houver conflito.
 * @return false Se a cor do vértice for válida em relação aos seus vizinhos.
 */
bool vertex_has_conflicts(const Graph &graph, const Individual &indv, const int &vertex_id);

/**
 * @brief Retorna uma lista com os IDs de todos os vértices que estão em conflito.
 * @param graph Estrutura do grafo.
 * @param indv Indivíduo (solução).
 * @return std::vector<int> Vetor contendo os vértices com conflito.
 */
std::vector<int> get_conflicted_vertices(const Graph &graph, const Individual &indv);

/**
 * @brief Sorteia e retorna o ID de um vértice que esteja em conflito.
 * @param graph Estrutura do grafo.
 * @param indv Indivíduo (solução).
 * @param rng Gerador de números aleatórios.
 * @return int ID do vértice selecionado, ou -1 se não houver conflitos (solução ótima).
 */
int get_random_conflicted_vertex(const Graph &graph, const Individual &indv, std::mt19937 &rng);


#endif