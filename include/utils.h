/**
 * @file utils.h
 * @brief Funções utilitárias para geração de números aleatórios na CPU.
 * * Este arquivo contém as declarações de funções auxiliares que envelopam 
 * a biblioteca <random> do C++11, facilitando a geração de inteiros, 
 * números de ponto flutuante e probabilidades de forma padronizada.
 */
#ifndef HEADER_UTILS
#define HEADER_UTILS

#include <random>

/**
 * @brief Gera um número inteiro aleatório dentro de um intervalo [min, max].
 * A distribuição é uniforme e o intervalo é inclusivo nas duas extremidades.
 * @param min Limite inferior do intervalo.
 * @param max Limite superior do intervalo.
 * @param rng Referência para o gerador Mersenne Twister (std::mt19937).
 * @return int Um valor inteiro aleatório entre min e max.
 */
int randint(int min, int max, std::mt19937& rng);

/**
 * @brief Gera um número inteiro aleatório [min, max], garantindo que seja diferente de um valor específico.
 * Útil para operadores genéticos ou buscas locais onde se deseja escolher 
 * uma nova cor/elemento diferente da cor atual.
 * @param min Limite inferior do intervalo.
 * @param max Limite superior do intervalo.
 * @param avoid O valor que NÃO deve ser retornado.
 * @param rng Referência para o gerador Mersenne Twister.
 * @return int Um valor inteiro aleatório entre min e max, e diferente de avoid.
 */
int randint_diff(int min, int max, int avoid, std::mt19937& rng);

/**
 * @brief Gera um número de ponto flutuante (double) aleatório em um intervalo [lower_bound, upper_bound).
 * A distribuição é uniforme e o limite superior é geralmente exclusivo.
 * @param lower_bound Limite inferior do intervalo.
 * @param upper_bound Limite superior do intervalo.
 * @param rng Referência para o gerador Mersenne Twister.
 * @return double Um valor aleatório entre lower_bound e upper_bound.
 */
double randdouble(double lower_bound, double upper_bound, std::mt19937& rng);

/**
 * @brief Gera um valor de probabilidade entre 0.0 e 1.0 (exclusivo).
 * Função de atalho para simular sorteios estocásticos (como taxa de mutação).
 * @param rng Referência para o gerador Mersenne Twister.
 * @return double O valor de probabilidade gerado em [0.0, 1.0).
 */
double generate_probability_value(std::mt19937& rng);

#endif