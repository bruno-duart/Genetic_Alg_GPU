#include "utils.h"

#include <cstdlib> // Para usar rand()
#include <ctime>   // Para usar time()
#include <random>

int randint(int lower_bound, int upper_bound, 
    std::mt19937& rng)
{
    std::uniform_int_distribution<int> unif(lower_bound, upper_bound);

    return unif(rng);
}

int randint_diff(int min, int max, int avoid, 
    std::mt19937& rng)
{
    int num;
    do
        num = randint(min, max, rng);
    while (num == avoid);
    return num;
}

double randdouble(double lower_bound, double upper_bound, 
    std::mt19937& rng)
{
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    
    double a_random_double = unif(rng);

    return a_random_double;
}

/**
 * @brief Retorna um valor double, relativo à probabilidade [0.0, 1.0)
 * @param rng O gerador de números aleatórios para a seleção.
 * @return double o valor de probabilidade gerado.
 */
double generate_probability_value(std::mt19937& rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    return unif(rng);
}