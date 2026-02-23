/**
 * @file utils.cpp
 * @brief Implementação das funções utilitárias de geração de números aleatórios.
 */
#include "utils.h"

#include <cstdlib> // Para usar rand()
#include <ctime>   // Para usar time()
#include <random>

int randint(int lower_bound, int upper_bound, 
    std::mt19937& rng)
{
    // std::uniform_int_distribution é inclusivo nos dois limites: [lower_bound, upper_bound]
    std::uniform_int_distribution<int> unif(lower_bound, upper_bound);

    return unif(rng);
}

int randint_diff(int min, int max, int avoid, 
    std::mt19937& rng)
{
    int num;
    // Laço "do-while" para forçar a geração de um novo número 
    // enquanto o número gerado colidir com a restrição 'avoid'.
    do
        num = randint(min, max, rng);
    while (num == avoid);
    return num;
}

double randdouble(double lower_bound, double upper_bound, 
    std::mt19937& rng)
{
    // std::uniform_real_distribution gera em [lower_bound, upper_bound)
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    
    double a_random_double = unif(rng);

    return a_random_double;
}

double generate_probability_value(std::mt19937& rng) {
    // Retorna probabilidade puramente no intervalo padrão de [0.0, 1.0)
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    return unif(rng);
}