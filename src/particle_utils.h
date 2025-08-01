#ifndef PARTICLE_UTILS_H
#define PARTICLE_UTILS_H
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include <string>
#include <functional>
#include <utility>
#include <stdexcept>

using namespace Eigen;

class TruncatedBetaDistribution {
    private:
        double alpha_, beta_;  // Shape parameters
        double a_, b_;        // Truncation bounds (0 ≤ a < b ≤ 1)
        std::mt19937_64 rng_; // Random number generator
        
        // Regularized incomplete beta function
        double ibeta(double x, double a, double b) {
            if (x < 0.0 || x > 1.0) {
                throw std::domain_error("x must be in [0, 1]");
            }
            
            // Continued fraction approximation (based on NR)
            const double eps = 1e-10;
            const int max_iter = 100;
            
            if (x == 0.0 || x == 1.0) {
                return x;
            }
            
            double bt = (x == 0.0 || x == 1.0) ? 0.0 :
                std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + 
                        a * std::log(x) + b * std::log(1.0 - x));
            
            if (x < (a + 1.0) / (a + b + 2.0)) {
                // Use continued fraction directly
                return bt * betacf(x, a, b) / a;
            } else {
                // Use continued fraction after symmetry transform
                return 1.0 - bt * betacf(1.0 - x, b, a) / b;
            }
        }
        
        // Continued fraction evaluation for incomplete beta function
        double betacf(double x, double a, double b) {
            const double eps = 1e-10;
            const int max_iter = 100;
            
            double qab = a + b;
            double qap = a + 1.0;
            double qam = a - 1.0;
            double c = 1.0;
            double d = 1.0 - qab * x / qap;
            if (std::abs(d) < eps) d = eps;
            d = 1.0 / d;
            double h = d;
            
            for (int m = 1; m <= max_iter; ++m) {
                int m2 = 2 * m;
                double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
                d = 1.0 + aa * d;
                if (std::abs(d) < eps) d = eps;
                c = 1.0 + aa / c;
                if (std::abs(c) < eps) c = eps;
                d = 1.0 / d;
                h *= d * c;
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
                d = 1.0 + aa * d;
                if (std::abs(d) < eps) d = eps;
                c = 1.0 + aa / c;
                if (std::abs(c) < eps) c = eps;
                d = 1.0 / d;
                double del = d * c;
                h *= del;
                if (std::abs(del - 1.0) < eps) break;
            }
            
            return h;
        }
        
    public:
        // Constructor with shape parameters and truncation bounds
        TruncatedBetaDistribution(double alpha, double beta, double a = 0.0, double b = 1.0, 
                                unsigned int seed = std::random_device{}())
            : alpha_(alpha), beta_(beta), a_(a), b_(b), rng_(seed) {
            if (alpha <= 0.0 || beta <= 0.0) {
                throw std::domain_error("alpha and beta must be positive");
            }
            if (a < 0.0 || a >= b || b > 1.0) {
                throw std::domain_error("truncation bounds must satisfy 0 ≤ a < b ≤ 1");
            }
        }
        
        // Generate a random sample from the truncated beta distribution
        double operator()() {
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            
            // Calculate CDF at truncation points
            double Fa = ibeta(a_, alpha_, beta_);
            double Fb = ibeta(b_, alpha_, beta_);
            
            // Inverse transform sampling
            double u = uniform(rng_);
            double Fu = Fa + u * (Fb - Fa);
            
            // Find x such that F(x) = Fu using Newton-Raphson
            double x = 0.5 * (a_ + b_); // Initial guess
            const double eps = 1e-10;
            const int max_iter = 100;
            
            for (int i = 0; i < max_iter; ++i) {
                double Fx = ibeta(x, alpha_, beta_);
                double fx = std::pow(x, alpha_ - 1.0) * std::pow(1.0 - x, beta_ - 1.0) / 
                           std::exp(std::lgamma(alpha_) + std::lgamma(beta_) - std::lgamma(alpha_ + beta_));
                
                double dx = (Fx - Fu) / fx;
                x -= dx;
                
                // Keep within bounds
                x = std::max(a_, std::min(b_, x));
                
                if (std::abs(dx) < eps * x) break;
            }
            
            return x;
        }
    };

struct hash_pair {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const
    {
        // Hash the first element
        size_t hash1 = std::hash<T1>{}(p.first);
        // Hash the second element
        size_t hash2 = std::hash<T2>{}(p.second);
        // Combine the two hash values
        return hash1
               ^ (hash2 + 0x9e3779b9 + (hash1 << 6)
                  + (hash1 >> 2));
    }
};

static double PI = 4*std::atan(1.0);
static long double SQRT1_2 = std::sqrt(0.5);
static std::uniform_real_distribution<long double> STD_UNIF;
static std::normal_distribution<long double> STD_NORMAL(0.0, 1.0);

VectorXd logit(const VectorXd& x);

void default_prior_vals(const int& n,
                        const int& d,
                        double& nu,
                        VectorXd& mu,
                        MatrixXd& sigma);

MatrixXi get_combs(const int& n, const int& j);

long double rbeta(const long double& a, const long double& b, std::mt19937_64& rng);

double sum(const std::vector<double>& arr);
long double sum(const std::vector<long double>& arr);
int rcat(std::vector<long double> weights, std::mt19937_64& rng);

long double rgamma(const long double& a, const long double& b, std::mt19937_64& rng);
long double rinvgamma(const long double& a, const long double& b, std::mt19937_64& rng);

long double log_dgamma(const long double& x, const long double& a, const long double& b);
long double log_dinvgamma_prod(const std::vector<long double>& x, const long double& a, const long double& b);
long double log_dinvgamma(const long double& x, const long double& a, const long double& b);
long double log_dnorm(const long double& x, const long double& mu, const long double& sd);

void remove_index(std::vector<int>& arr, const int& index);
void remove_index(std::vector<long double>& arr, const int& index);

void saveData(std::string fileName, MatrixXd  matrix);
MatrixXd openData(std::string fileToOpen);

#endif