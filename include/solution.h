#ifndef HEADER_SOLUTION
#define HEADER_SOLUTION

#include "graphs.h"
#include <iostream>
#include <vector>
#include <random>

using Individual = std::vector<int>;
using Fitness = int;

struct NeighborMove {
    int vertex_id;
    int new_color;
    int delta_E;
};

void evaluate_fitness(const Graph &graph, const Individual& indv, Fitness& fitness);
void evaluate_fitness_incremental(const Graph &graph, const Individual &indv, Fitness &fitness, std::vector<int> &conflicts_per_vertex);

// Método para gerar uma solução aleatória
void random_individual(int num_color, const Graph &graph, Individual& indv, std::mt19937 &rng);


// Método para copiar uma solução para outra
void copy_individual(const Individual &from, const Fitness& fit_from, Individual &to, Fitness& fit_to);

// Função para criar uma solução inicial
Individual initialize_individual(int num_color, const Graph &graph, std::mt19937 &rng);

// Método para exibir a solução
void individual_toString(const Individual& indv); 
void print_individual(const Individual& indv, const Fitness& fit);

int compute_fitness_change(const Graph &graph, const Individual &indv, int vertex, int new_color);
int compute_swap_fitness_change(const Graph &graph, const Individual &indv, int u, int v);
int find_most_conflicted_vertex(const Individual &indv, const Graph &graph);
int find_second_most_conflicted_vertex(Individual &indv, Graph &graph, int most_conflicted);

void explore_neighborhood(Individual& new_indv, const Individual& indv, Fitness& new_fit, const Fitness& fit, const Graph& graph, int num_colors, std::mt19937& rng);

bool vertex_has_conflicts(const Graph &graph, const Individual &indv, const int &vertex_id);
std::vector<int> get_conflicted_vertices(const Graph &graph, const Individual &indv);
int get_random_conflicted_vertex(const Graph &graph, const Individual &indv, std::mt19937 &rng);


#endif