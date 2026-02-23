#ifndef HEADER_UTILS
#define HEADER_UTILS

#include <random>

int randint(int min, int max, std::mt19937& rng);
int randint_diff(int min, int max, int avoid, std::mt19937& rng);
double randdouble(double lower_bound, double upper_bound, std::mt19937& rng);
double generate_probability_value(std::mt19937& rng);

#endif