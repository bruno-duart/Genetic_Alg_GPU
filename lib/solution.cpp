#include "solution.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <cstdlib> // Para usar rand()
#include <ctime>   // Para usar time()
#include <random>

// Método para avaliar a fitness de uma solução
// Assumimos grafo não-direcionado; cada conflito é contado duas vezes.
void evaluate_fitness(const Graph &graph, const Individual &indv, Fitness &fitness)
{
    fitness = 0;
    for (int node = 0; node < graph.getNumVertices(); node++)
    {
        for (auto &neighboor : graph.adjList[node])
        {
            if (indv[node] == indv[neighboor])
            {
                fitness++;
            }
        }
    }
    fitness /= 2;
}

void evaluate_fitness_incremental(const Graph &graph, const Individual &indv, Fitness &fitness, std::vector<int> &conflicts_per_vertex)
{
    int n{graph.getNumVertices()};
    conflicts_per_vertex.assign(n, 0);
    fitness = 0;

    for (int v{0}; v < n; ++v){
        for (int u : graph.getNeighbors(v)){
            if (indv[v] == indv[u]){
                conflicts_per_vertex[v]++;
            }
        }
    }

    for (int node = 0; node < graph.getNumVertices(); node++)
    {
        for (auto &neighboor : graph.adjList[node])
        {
            if (indv[node] == indv[neighboor])
            {
                fitness++;
            }
        }
    }
    fitness /= 2;
}

// Método para gerar uma solução aleatória
void random_individual(int num_color, const Graph &graph, Individual &indv, std::mt19937 &rng)
{
    for (int i = 0; i < graph.getNumVertices(); i++)
    {
        // indv[i] = (rand() % num_color) + 1;
        indv[i] = randint(0, num_color - 1, rng);
    }

    // evaluate_fitness(graph);
}

// Método para copiar uma solução para outra
void copy_individual(const Individual &from, const Fitness &fit_from, Individual &to, Fitness &fit_to)
{
    to = from;
    fit_to = fit_from;
}

// Função para criar uma solução inicial
Individual initialize_individual(int num_color, const Graph &graph, std::mt19937 &rng)
{
    Individual indv(graph.getNumVertices());
    random_individual(num_color, graph, indv, rng);
    return indv;
}

// Método para exibir a solução
void individual_toString(const Individual &indv)
{
    for (int el : indv)
        std::cout << el << " ";

    std::cout << std::endl;
}

void print_individual(const Individual &indv, const Fitness &fit)
{
    individual_toString(indv);
    std::cout << "fit: " << fit << std::endl;
}

int compute_fitness_change(const Graph &graph, const Individual &indv, int vertex, int new_color)
{
    int old_color = indv[vertex];
    int conflict_change = 0;

    // Percorre os vizinhos do vértice
    for (int neighbor : graph.getNeighbors(vertex))
    {
        if (indv[neighbor] == old_color)
        {
            // Removendo um conflito
            conflict_change--;
        }
        if (indv[neighbor] == new_color)
        {
            // Adicionando um conflito
            conflict_change++;
        }
    }

    return conflict_change;
}

int compute_swap_fitness_change(const Graph &graph, const Individual &indv, int u, int v)
{
    int delta{0};
    int color_u{indv[u]};
    int color_v{indv[v]};

    for (auto neighbor : graph.getNeighbors(u))
    {
        if (neighbor == v)
            continue;
        if (indv[neighbor] == color_u)
            delta--; // removendo conflito antigo
        if (indv[neighbor] == color_v)
            delta++; // possível novo conflito
    }

    for (auto neighbor : graph.getNeighbors(v))
    {
        if (neighbor == u)
            continue;
        if (indv[neighbor] == color_v)
            delta--; // removendo conflito antigo
        if (indv[neighbor] == color_u)
            delta++; // possível novo conflito
    }

    return delta;
}

int find_most_conflicted_vertex(const Individual &indv, const Graph &graph)
{
    int worst_vertex = -1;
    int max_conflicts = -1;

    for (int v = 0; v < graph.getNumVertices(); v++)
    {
        int conflicts = 0;
        for (int neighbor : graph.getNeighbors(v))
        {
            if (indv[v] == indv[neighbor])
            {
                conflicts++;
            }
        }

        if (conflicts > max_conflicts)
        {
            max_conflicts = conflicts;
            worst_vertex = v;
        }
    }

    return worst_vertex;
}

int find_second_most_conflicted_vertex(Individual &indv, Graph &graph, int most_conflicted)
{
    int second_worst_vertex = -1;
    int second_max_conflicts = -1;

    for (int v = 0; v < graph.getNumVertices(); v++)
    {
        if (v == most_conflicted)
            continue; // Skip the most conflicted vertex

        int conflicts = 0;
        for (int neighbor : graph.getNeighbors(v))
        {
            if (indv[v] == indv[neighbor])
            {
                conflicts++;
            }
        }

        if (conflicts > second_max_conflicts)
        {
            second_max_conflicts = conflicts;
            second_worst_vertex = v;
        }
    }

    return second_worst_vertex;
}

void explore_neighborhood(Individual &new_indv, const Individual &indv, Fitness &new_fit, const Fitness &fit, const Graph &graph, int num_colors, std::mt19937 &rng)
{
    // Copia a solução base
    copy_individual(indv, fit, new_indv, new_fit);
    // Encontra um movimento aleatório
    int vertex = randint(0, graph.getNumVertices() - 1, rng);
    int old_color = new_indv[vertex];
    int new_color = randint_diff(0, num_colors, old_color, rng);

    // Calcula a variação de fitness
    int delta_fit = compute_fitness_change(graph, indv, vertex, new_color);

    // Aplica o movimento e atualiza o fitness incrementalmente
    new_indv[vertex] = new_color;
    new_fit += delta_fit;
}

bool vertex_has_conflicts(const Graph &graph, const Individual &indv, const int &vertex_id)
{
    for (auto v : graph.getNeighbors(vertex_id))
    {
        if (indv[v] == indv[vertex_id])
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief Retorna uma lista de todos os vértices que estão em conflito.
 * * Um vértice está em conflito se tem pelo menos um vizinho com a mesma cor.
 * * @param graph O grafo a ser analisado.
 * @param indv A coloração atual (solução).
 * @return std::vector<int> Uma lista contendo os IDs dos vértices em conflito.
 */
std::vector<int> get_conflicted_vertices(const Graph &graph, const Individual &indv)
{
    std::vector<int> conflicted_vertices;
    // O loop começa em 0 e vai até < graph.getNumVertices()
    for (int v = 0; v < graph.getNumVertices(); v++)
    {
        // Verifica se o vértice v está em conflito com algum de seus vizinhos
        if (vertex_has_conflicts(graph, indv, v))
        {
            conflicted_vertices.push_back(v);
        }
    }

    return conflicted_vertices;
}

/**
 * @brief Retorna o ID de um vértice em conflito selecionado aleatoriamente.
 * * Se não houver vértices em conflito (solução é própria), retorna -1.
 * * @param graph O grafo.
 * @param indv A coloração atual (solução).
 * @param rng O gerador de números aleatórios para a seleção.
 * @return int O ID do vértice em conflito aleatório, ou -1 se não houver conflitos.
 */
int get_random_conflicted_vertex(const Graph &graph, const Individual &indv, std::mt19937 &rng)
{
    // 1. Obter todos os vértices em conflito
    std::vector<int> conflicted_vertices = get_conflicted_vertices(graph, indv);

    // 2. Verificar se há algum conflito
    if (conflicted_vertices.empty())
    {
        return -1; // Retorna -1 se a solução for própria (fitness = 0)
    }

    // 3. Selecionar um índice aleatório dentro do vetor de conflitos
    // O índice deve estar no intervalo [0, tamanho_do_vetor - 1]
    int random_index = randint(0, conflicted_vertices.size() - 1, rng);

    // 4. Retornar o vértice correspondente ao índice aleatório
    return conflicted_vertices[random_index];
}

