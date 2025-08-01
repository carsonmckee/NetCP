#ifndef GIBBS_H
#define GIBBS_H
#include "particle_utils.h"
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>

using namespace Eigen;

class Gibbs{
    
    public:
        Gibbs(const MatrixXd& Y_, 
                      const long double& rho_upper_, 
                      const bool& use_distance_,
                      const MatrixXd& Z_,
                      const VectorXd& Y_pred_);
        
        MatrixXd Y;
        int T, d;
        int keep_ind;
        bool use_distance;
        MatrixXi A, U, tau;
        MatrixXd U_sum, W_q_acceptance, distances;
        MatrixXd C, A_sum, W, W_sum, g_params, g_params_sum;
        std::vector<MatrixXi> A_combs;
        VectorXd Y_pred;
        MatrixXd N_cps;
        
        long double tau_dist;
        long double est_pred_score;
        
        bool cache_log_pred;
        
        std::vector<std::unordered_map<std::pair<int, int>, long double, hash_pair>> log_pred_cache;
        
        long double max_post_val;
        MatrixXd A_map, W_map, U_map, g_map;
        
        long double rho_upper, rho;
        
        std::vector<std::vector<int>> dependent_series;
        
        std::vector<std::vector<int>> particles;
        std::vector<std::vector<long double>> weights;
        
        std::vector<long double> sorted_indices;
        
        VectorXd saved_val;
        VectorXd log_post_vals;

        void sample(const int& n_iter, const int& burn_in);
        
        void sample_hidde_states(std::mt19937_64& rng);

        void sample_A(std::mt19937_64& rng);
        void sample_A_distance(std::mt19937_64& rng);
        void sample_W(std::mt19937_64& rng);
        void sample_g_params(std::mt19937_64& rng);
        void sample_W_g_joint(std::mt19937_64& rng);
        long double check_is_map(const int& iter, const int& burn_in);
        
        long double predictive_sample(){
            long double prob_, pred_j, val = 1.0;
            for(int j=0; j<d; j++){
                prob_ = prob(T, j, j, tau(j, T-1), tau.col(T-1));
                pred_j = prob_*std::exp(log_pred(j, T, T-1)) + (1-prob_)*std::exp(log_pred(j, T, tau(j, T-1)));
                val *= pred_j;
            }
            return val;
        };
                
        long double transition(const int& t, const int& j, const int& next_tau, const int& prev_tau);
        long double f(const int& t, const int& j, const int& tau_j, const int& i, const int& tau_i, const VectorXi& tau_rest);
        long double prob(const int& t, const int& j, const int& i, const int& tau_i, const VectorXi& tau_rest);
        
        // segment data factor
        virtual long double log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
            std::cout << "method must be implemented by subclass!!\n";
            std::abort();
            return 1.0l;
        }
        
        virtual void sample_data_factor_params(const int& iter, const int& burn_in, std::mt19937_64& rng){
            std::cout << "method must be implemented by subclass!!\n";
            std::abort();
        }
        
        virtual void set_up_sampling(const int& n_iter, const int& burn_in){
            
        }

        // log predictive distribution
        long double log_pred(const int& j, const int& t, const int& tau);

        long double g_ij(const int& i, const int& j, const int& t);
        
};

class NormalMeanGibbs: public Gibbs{

    public:
        NormalMeanGibbs(const MatrixXd& Y_, 
                    const long double& rho_upper_,
                    const bool& use_distance_,
                    const MatrixXd& Z_, 
                    const VectorXd& means_, 
                    const VectorXd& variances_, 
                    const VectorXd& sigma2s_,
                    const VectorXd& Y_pred_) : Gibbs(Y_, rho_upper_, use_distance_, Z_, Y_pred_){
            means = means_;
            variances = variances_;
            sigma2s = sigma2s_;
            for(int j=0; j<Y.rows(); j++){
                std::vector<long double> Y_sumsj={0.0}, Y2_sumsj={0.0};
                Y_sums.push_back(Y_sumsj);
                Y2_sums.push_back(Y2_sumsj);
                for(int t=0; t<Y.cols(); t++){
                    Y_sums[j].push_back(Y_sums[j][t] + Y(j, t));
                    Y2_sums[j].push_back(Y2_sums[j][t] + Y(j, t)*Y(j, t));
                }
            }
        }
        VectorXd means, variances, sigma2s;
        std::vector<std::vector<long double>> Y_sums, Y2_sums;
        
        virtual long double log_data_factor(const int&j, const int& prev_ep, const int& next_ep);

        virtual void sample_data_factor_params(const int& iter, const int& burn_in, std::mt19937_64& rng){
            
        }
};

class LaplaceGibbs: public Gibbs{
    public:
        LaplaceGibbs(const MatrixXd& Y_, 
                const long double& rho_upper_, 
                const bool& use_distance_,
                const MatrixXd& Z_,
                const VectorXd& means_, 
                const VectorXd& alphas_, 
                const VectorXd& lambdas_, 
                const VectorXd& Y_pred_) : Gibbs(Y_, rho_upper_, use_distance_, Z_, Y_pred_){
            means = means_;
            alphas = alphas_;
            lambdas = lambdas_;
            b_sum = MatrixXd::Zero(d, T);
            for(int j=0; j<Y.rows(); j++){
                std::vector<long double> Y_abs_sumsj={0.0};
                Y_abs_sums.push_back(Y_abs_sumsj);
                for(int t=0; t<Y.cols(); t++){
                    Y_abs_sums[j].push_back(Y_abs_sums[j][t] + std::abs(Y(j, t)-means(j)));
                }
            }
        }
        VectorXd means, alphas, lambdas;
        std::vector<std::vector<long double>> Y_abs_sums;
        MatrixXd alpha_store, lambda_store;
        MatrixXd b_sum;

        long double prop_sd = 0.5;
        
        virtual void set_up_sampling(const int& n_iter, const int& burn_in){
            alpha_store = MatrixXd::Zero(n_iter-burn_in, d);
            lambda_store = MatrixXd::Zero(n_iter-burn_in, d);
        }
        
        virtual long double log_data_factor(const int&j, const int& prev_ep, const int& next_ep);
        
        virtual void sample_data_factor_params(const int& iter, const int& burn_in, std::mt19937_64& rng){
            long double prop_sd = 0.5;
            long double prop, curr, accept_prob;
            for(int j=0; j<d; j++){
                // first sample alpha
                for(int th=0; th<10; th++){
                    curr = alphas(j);
                    prop = curr + prop_sd * STD_NORMAL(rng);
                    if(prop < 0){
                        continue;
                    }
                    accept_prob = log_dgamma(prop, 0.001, 0.001) - log_dgamma(curr, 0.001, 0.001); 
                    
                    alphas(j) = prop;
                    for(int t=1; t<T; t++){
                        accept_prob += log_pred(j, t, tau(j, t));
                    }

                    alphas(j) = curr;
                    for(int t=1; t<T; t++){
                        accept_prob -= log_pred(j, t, tau(j, t));
                    }

                    if(STD_UNIF(rng) < std::exp(accept_prob)){
                        alphas(j) = prop;
                    } else {
                        alphas(j) = curr;
                    }
                }

                for(int th=0; th<10; th++){
                    curr = lambdas(j);
                    prop = curr + prop_sd * STD_NORMAL(rng);
                    if(prop < 0){
                        continue;
                    }
                    accept_prob = log_dgamma(prop, 0.001, 0.001) - log_dgamma(curr, 0.001, 0.001); 
                    
                    lambdas(j) = prop;
                    for(int t=1; t<T; t++){
                        accept_prob += log_pred(j, t, tau(j, t));
                    }

                    lambdas(j) = curr;
                    for(int t=1; t<T; t++){
                        accept_prob -= log_pred(j, t, tau(j, t));
                    }

                    if(STD_UNIF(rng) < std::exp(accept_prob)){
                        lambdas(j) = prop;
                    } else {
                        lambdas(j) = curr;
                    }
                }

            }
        }
};

class ARProcessGibbs: public Gibbs{

    public:

        ARProcessGibbs(const MatrixXd& Y_, 
                  const long double& rho_upper_, 
                  const bool& use_distance_,
                  const MatrixXd& Z_,
                  const MatrixXd& deltas_, 
                  const VectorXd& alphas_, 
                  const VectorXd& lambdas_, 
                  const VectorXd& Y_pred_) : 
                        Gibbs(Y_, rho_upper_, use_distance_, Z_, Y_pred_){
            
            for(int l=0; l<deltas_.cols(); l++){
                // deltas stores inverse diagonal
                deltas.push_back((1.0/deltas_.col(l).array()).matrix().asDiagonal());
            }
            alphas = alphas_;
            lambdas = lambdas_;
            
            L = deltas_.rows();
            T = Y.cols() - L;
            // Y = Y_.block(0, L, Y_.rows(), T);
            I_L = MatrixXd::Identity(L, L);
            Delta = MatrixXd::Zero(L, L);
            XY = VectorXd::Zero(L);
            
            // construct design matrices
            for(int j=0; j<d; j++){
                MatrixXd X(T, L);
                for(int t = L; t<Y.cols(); t++){
                    for(int l=1; l<=L; l++){
                        X(t-L, l-1) = Y(j, t-L);
                    }
                }
                Xs.push_back(X);
            }
            
            // construct sufficient statistic caches
            for(int j=0; j<d; j++){
                std::vector<MatrixXd> XX_sums_j = {MatrixXd::Zero(L, L)};
                std::vector<VectorXd> XY_sums_j = {VectorXd::Zero(L)};
                std::vector<long double> YY_sums_j = {0.0};
                for(int t=0; t < T; t++){
                    YY_sums_j.push_back(YY_sums_j[t] + Y(j, t+L)*Y(j, t+L));
                    MatrixXd XX = XX_sums_j[t] + Xs[j].row(t).transpose()*Xs[j].row(t);
                    XX_sums_j.push_back(XX);
                    VectorXd XY = XY_sums_j[t] + Xs[j].row(t).transpose()*Y(j, t+L);
                    XY_sums_j.push_back(XY);
                }
                XX_sums.push_back(XX_sums_j);
                XY_sums.push_back(XY_sums_j);
                YY_sums.push_back(YY_sums_j);
            }
            Y = Y_.block(0, L, Y_.rows(), T);
            T = T-1;
        }

        int L;
        std::vector<MatrixXd> Xs;
        std::vector<std::vector<MatrixXd>> XX_sums;
        std::vector<std::vector<VectorXd>> XY_sums;
        std::vector<std::vector<long double>> YY_sums;
        MatrixXd I_L;
        VectorXd alphas, lambdas, XY;
        std::vector<MatrixXd> deltas;

        // temp holders
        MatrixXd Delta;

        long double prop_sd = 0.2;

        virtual long double log_data_factor(const int&j, const int& prev_ep, const int& next_ep);
        
        virtual void sample_data_factor_params(const int& iter, const int& burn_in, std::mt19937_64& rng){
            
            
        }

};


long double sum_beta_lambdas(const Array<double, -1, 1>& betas, const Array<double, -1, 1>& lambda2s, const int& d);
long double sum_beta_lambdas(const Array<double, -1, 1>& betas, const long double& tau2, const int& d);

#endif