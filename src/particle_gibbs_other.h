#ifndef PARTICLE_GIBBS_OTHER_H
#define PARTICLE_GIBBS_OTHER_H
#include "particle_utils.h"
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <algorithm>
#include <string>

using namespace Eigen;

class ParticleGibbsOther{
    
    public:
        ParticleGibbsOther(const MatrixXd& Y_, 
                      const int& n_part_, 
                      const std::string& model_, 
                      const VectorXd& y_pred_);

        std::string model;
        MatrixXd Y;
        int n_parts;
        int T, d;
        int keep_ind;
        MatrixXi tau;
        MatrixXi U;
        MatrixXd U_sum;
        MatrixXd probs, prop_sd;
        VectorXd y_pred;

        bool cache_log_pred;
        std::vector<std::unordered_map<std::pair<int, int>, long double, hash_pair>> log_pred_cache;

        // for CCP models
        double nu;
        VectorXd mu;
        MatrixXd sigma, sigma_inv;

        std::vector<std::vector<int>> particles;
        std::vector<std::vector<long double>> weights;

        std::vector<long double> sorted_indices;

        long double pred_score_est;
        
        void sample(const int& n_iter, const int& burn_in, const int& seed_, const bool& verbose);

        void sample_probs(const int& iter, const int& burn_in, std::mt19937_64& rng);
        long double log_p_cond_global(const int& j);
        void sample_probs_global(std::mt19937_64& rng);
        long double log_p_cond(const long double& pit, const int& Uit, const VectorXd& logit_p_mu, const MatrixXd& sigma_inv);
        void sample_probs_non_global(std::mt19937_64& rng);

        void filter_forward(const int& j, std::mt19937_64& rng);
        void compute_particle_weights(const int& j, const int&t);
        void sample_backwards(const int& j, std::mt19937_64& rng);
        void resample_particles(const int& j,const int& t, std::mt19937_64& rng);

        long double transition(const int& t, const int& j, const int& next_tau, const int& prev_tau);
        
        void argsort(const std::vector<long double> &array);
        std::pair<long double, std::vector<int>> get_resample_inds(const int& t);
        void resample_particles_optimal(const int& j,const int& t, std::mt19937_64& rng);
        void strat_resample(const int& t, const long double& K, const std::vector<int>& inds, std::mt19937_64& rng);
        void cond_strat_resample(const int& t, const long double& K, const long double& n_to_sample, const std::vector<int>& inds, const int& to_keep, std::mt19937_64& rng);
        
        long double pred_score();

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
        
};

class NormalMeanOther: public ParticleGibbsOther{

    public:
        NormalMeanOther(const MatrixXd& Y_, 
                        const int& n_part_, 
                        const std::string& model_, 
                        const VectorXd& y_pred_,
                        const VectorXd& means_, 
                        const VectorXd& variances_, 
                        const VectorXd& sigma2s_) : ParticleGibbsOther(Y_, n_part_, model_, y_pred_){
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

class LaplaceOther: public ParticleGibbsOther{
    public:
        LaplaceOther(const MatrixXd& Y_, 
                const int& n_part_, 
                const std::string& model_, 
                const VectorXd& y_pred_,
                const VectorXd& means_, 
                const VectorXd& alphas_, 
                const VectorXd& lambdas_) : ParticleGibbsOther(Y_, n_part_, model_, y_pred_){
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
        
        long double prop_sd = 0.8;
        
        virtual void set_up_sampling(const int& n_iter, const int& burn_in){
            alpha_store = MatrixXd::Zero(n_iter-burn_in, d);
            lambda_store = MatrixXd::Zero(n_iter-burn_in, d);
        }
        
        virtual long double log_data_factor(const int&j, const int& prev_ep, const int& next_ep);
        
        virtual void sample_data_factor_params(const int& iter, const int& burn_in, std::mt19937_64& rng){
            
            for(int j=0; j<d; j++){
                // first sample laplace scale parameters 
                std::vector<long double> bs;
                long double b, b_inv_sum = 0.0;
                long double y_abs_sum = std::abs(Y(j, 0)-means(j));
                long double nij = 1.0;
                int prev_ep = -1;
                for(int t=1; t<=T; t++){
                    if((t == T) || (tau(j, t) == t-1)){ // dubious since tau(j, T) will cause seg error
                        // a cp 
                        // sample new_beta
                        b = rinvgamma(alphas(j) + nij, lambdas(j) + y_abs_sum, rng);
                        bs.push_back(b);
                        b_inv_sum += 1.0/b;
                        if(iter >= burn_in){
                            for(int s=(prev_ep+1); s<t; s++){
                                b_sum(j, s) += b;
                            }
                        }
                        prev_ep = t-1;
                        if(t < T){
                            y_abs_sum = std::abs(Y(j, t)-means(j));
                            nij = 1.0;
                        }
                    } else {
                        y_abs_sum += std::abs(Y(j, t)-means(j));
                        nij += 1.0;
                    }
                }
                long double n = (long double) bs.size();
                
                // sample alpha
                for(int th=0; th<10; th++){
                    long double prop_alpha = alphas(j) + STD_NORMAL(rng) * prop_sd;
                    if(prop_alpha > 0){
                        long double log_val = log_dinvgamma_prod(bs, prop_alpha, lambdas(j)) + log_dgamma(prop_alpha, 0.001, 0.001) + log_dnorm(alphas(j), prop_alpha, prop_sd);
                        log_val -= (log_dinvgamma_prod(bs, alphas(j), lambdas(j)) + log_dgamma(alphas(j), 0.001, 0.001) + log_dnorm(prop_alpha, alphas(j), prop_sd));
                        if(STD_UNIF(rng) < std::exp(log_val)){
                            alphas(j) = prop_alpha;
                        }
                    }
                }
                // sample lambda
                lambdas(j) = rgamma(0.001 + n*alphas(j), 0.001 + b_inv_sum, rng);

                if(iter >= burn_in){
                    alpha_store(iter-burn_in, j) = alphas(j);
                    lambda_store(iter-burn_in, j) = lambdas(j);
                }
            }

        }
};

class ARProcessOther: public ParticleGibbsOther{

    public:

        ARProcessOther(const MatrixXd& Y_, 
                  const int& n_part_, 
                  const std::string& model_, 
                  const VectorXd& y_pred_,
                  const MatrixXd& deltas_, 
                  const VectorXd& alphas_, 
                  const VectorXd& lambdas_) : 
                        ParticleGibbsOther(Y_, n_part_, model_, y_pred_){

            for(int l=0; l<deltas_.cols(); l++){
                // deltas stores inverse diagonal
                deltas.push_back((1.0/deltas_.col(l).array()).matrix().asDiagonal());
            }
            alphas = alphas_;
            lambdas = lambdas_;
            
            L = deltas_.rows();
            T = Y.cols() - L;
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
            Y = Y.block(0, L, Y.rows(), T);
            T-=1; // last col is the predictive column
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

        virtual long double log_data_factor(const int&j, const int& prev_ep, const int& next_ep);
        
        virtual void sample_data_factor_params(const int& iter, const int& burn_in, std::mt19937_64& rng){
            
        }

};

long double sum_beta_lambdas(const Array<double, -1, 1>& betas, const Array<double, -1, 1>& lambda2s, const int& d);
long double sum_beta_lambdas(const Array<double, -1, 1>& betas, const long double& tau2, const int& d);

#endif