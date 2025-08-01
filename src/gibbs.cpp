#include "gibbs.h"
#include <unordered_map>

using namespace std::chrono;
using namespace Eigen;

Gibbs::Gibbs(const MatrixXd& Y_, 
                             const long double& rho_upper_, 
                             const bool& use_distance_,
                             const MatrixXd& Z_,
                             const VectorXd& Y_pred_){
    
    Y = Y_;
    T = Y.cols();
    Y.conservativeResize(Y.rows(), Y.cols()+1);
    Y.col(T) = Y_pred_;
    d = Y.rows();
    rho_upper = rho_upper_;
    rho = 0.1;
    Y_pred = Y_pred_;
    use_distance = use_distance_;
    if(use_distance){
        distances = MatrixXd::Zero(d, d);
        for(int i=0; i<d; i++){
            for(int j=0; j<d; j++){
                // euclidean distance between series
                distances(i, j) = std::sqrt((Z_.row(i).array()-Z_.row(j).array()).square().sum());
            }
        }
        distances /= distances.maxCoeff();
        tau_dist = 1.0;
        std::cout << distances << std::endl;
    }
    
    U = MatrixXi::Zero(d, T-1);
    C = MatrixXd::Zero(d, T);
    U_sum = MatrixXd::Zero(d, T-1);
    W_q_acceptance = MatrixXd::Zero(d, d);
    
    // hidden state holder
    tau = -1*MatrixXi::Ones(d, T); // initialise with no change-points
    
    // g params
    g_params = 0.4*MatrixXd::Ones(d, d);
    g_params_sum = MatrixXd::Zero(d, d);
    for(int i=0; i<d; i++){
        g_params(i, i) = 0.02;
    }

    // Adjacency Matrix
    A = MatrixXi::Ones(d, d);
    A_sum = MatrixXd::Zero(d, d);
    for(int i=0; i<d; i++){
        A(i, i) = 1;
    }
    
    cache_log_pred = true;
    for(int i=0; i<d; i++){
        std::unordered_map<std::pair<int, int>, long double, hash_pair> cache;
        log_pred_cache.push_back(cache);
    }
    
    // keep track of series which depend on the series j (have an edge directed from j)
    for(int j=0; j<d; j++){
        std::vector<int> dep_series;
        for(int i=0; i<d; i++){
            if((i != j) & (A(j, i) == 1)){
                dep_series.push_back(i);
            }
        }
        dependent_series.push_back(dep_series);
    }

    // Weights matrix
    W = MatrixXd::Ones(d, d);
    W_sum = MatrixXd::Zero(d, d);
    
    // particles initialise each with a single particle
    for(int t=0; t<T; t++){
        particles.push_back({-1});
        weights.push_back({1.0});
    }
    
}

void Gibbs::sample(const int& n_iter, const int& burn_in){

    auto start = high_resolution_clock::now();
    unsigned int seed = (unsigned int) time(NULL);
    // unsigned int seed = 1;
    std::mt19937_64 rng{seed};
    est_pred_score = 0;
    set_up_sampling(n_iter, burn_in);
    N_cps = MatrixXd::Zero(d, n_iter-burn_in);
    log_post_vals = VectorXd::Zero(n_iter);
    saved_val = VectorXd::Zero(n_iter);

    for(int iter=0; iter<n_iter; iter++){
        
        std::cout << "Iter: " << iter << std::endl;
        sample_hidde_states(rng);

        U.fill(0);
        for(int j=0; j<d; j++){
            for(int t=0; t<T-1; t++){
                if(tau(j, t) != tau(j, t+1)){
                    U(j, t) = 1;
                }
            }
        }
        if(use_distance){
            sample_A_distance(rng);
        } else {
            sample_A(rng);
        }
        for(int j=0; j<d; j++){
            // std::vector<int> dep_series;
            dependent_series[j].clear();
            for(int i=0; i<d; i++){
                if((i != j) & (A(j, i) == 1)){
                    dependent_series[j].push_back(i);
                }
            }
        }

        sample_W_g_joint(rng);

        log_post_vals(iter) = check_is_map(iter, burn_in);
        saved_val(iter) = U(0, 582);
        // saved_val(iter) = U(0, 449);

        sample_data_factor_params(iter, burn_in, rng);
        
        if(iter >= burn_in){
            U_sum += U.cast<double>();
            A_sum += A.cast<double>();
            W_sum += W.cast<double>();

            N_cps.col(iter-burn_in) = U.rowwise().sum().cast<double>();

            g_params_sum += g_params;
            // std::cout << "pred sample = " << predictive_sample() << std::endl;
            est_pred_score += predictive_sample();
        }

    }
    
    U_sum /= (double) (n_iter - burn_in);
    A_sum /= (double) (n_iter - burn_in);
    W_sum /= (double) (n_iter - burn_in);
    g_params_sum /= (double) (n_iter - burn_in);
    est_pred_score /= (long double) (n_iter - burn_in);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << n_iter << " iterations in " << duration.count() << " ms (or " << duration.count()/1000000 << " seconds)\n";
}

int next_end_point(const RowVectorXi& tau, const int& t){
    for(int k=t+1; k<tau.size(); k++){
        if(tau(k) == k-1){
            return k-1;
        }
    }
    return tau.size()-1;
}

void Gibbs::sample_hidde_states(std::mt19937_64& rng){
    int last_ep, next_ep;
    
    long double cp_log_prob, no_cp_log_prob, prob_, max_;
    long double prev_probs;

    long double prior_seg_cp, prior_seg_no_cp;
    long double seg_cp_valid, seg_no_cp_valid;
    
    for(int j=0; j<d; j++){
        
        last_ep = -1;
        
        for(int t=1; t<T; t++){
            next_ep = next_end_point(tau.row(j), t);
            // set tau(j, t) = t-1 (and propagate all forward to next end point)
            for(int k=t; k<=next_ep; k++){
                tau(j, k) = t-1;
            }
            // two segments induced
            cp_log_prob = log_data_factor(j, last_ep, t-1) + log_data_factor(j, t-1, next_ep);
            // add prior component
            for(int k=last_ep+1; k<=std::min({next_ep+1, T-1}); k++){
                // std::cout << "k = " << k << std::endl; 
                if(k == 0){
                    continue;
                }
                for(int l=0; l<d; l++){
                    
                    if(tau(l, k) == k-1){
                        cp_log_prob += std::log(prob(k, l, l, tau(l, k-1), tau.col(k-1)));
                    } else {
                        cp_log_prob += std::log(1.0 - prob(k, l, l, tau(l, k-1), tau.col(k-1)));
                    }
                }
            }
            // set tau(j, k) = tau(j, t-1) (and propagate forward to next end point)
            for(int k=t; k<=next_ep; k++){
                tau(j, k) = tau(j, t-1);
            }
            // single segment
            no_cp_log_prob = log_data_factor(j, last_ep, next_ep);
            // add prior component
            for(int k=last_ep+1; k<=std::min({next_ep + 1, T-1}); k++){
                if(k == 0){
                    continue;
                }
                for(int l=0; l<d; l++){
                    if(tau(l, k) == k-1){
                        no_cp_log_prob += std::log(prob(k, l, l, tau(l, k-1), tau.col(k-1)));
                    } else {
                        no_cp_log_prob += std::log(1.0 - prob(k, l, l, tau(l, k-1), tau.col(k-1)));
                    }
                }
            }
            max_ = std::max({no_cp_log_prob, cp_log_prob});
            cp_log_prob -= max_;
            no_cp_log_prob -= max_;
            prob_ = std::exp(cp_log_prob) / (std::exp(cp_log_prob) + std::exp(no_cp_log_prob));
            
            if(STD_UNIF(rng) < prob_){
                // a cp
                for(int k=t; k<=next_ep; k++){
                    tau(j, k) = t-1;
                }
                last_ep = t-1;
            } else {
                // no cp
                
            }

        }
    }

}

long double Gibbs::check_is_map(const int& iter, const int& burn_in){
    long double log_val = 0.0;

    // add likelihood and hidden state prior component
    for(int j=0; j<d; j++){
        for(int t=1; t<T; t++){
            log_val += log_pred(j, t, tau(j, t)); // likelihood
            if(tau(j, t) != tau(j, t-1)){ // hidden state probabilities
                log_val += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
            } else {
                log_val += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
            }
        }
    }

    // add A prior component 
    long double log_rho = std::log(rho/2), log_1_rho = std::log(1-rho);
    for(int i=0; i<d; i++){
        for(int j=i+1; j<d; j++){
            if((A(i, j) == 1) || (A(j, i) == 1)){
                log_val += log_rho;
            } else {
                log_val += log_1_rho;
            }
        }
    }

    // add W prior component 
    for(int i=0; i<d; i++){
        for(int j=0; j<d; j++){
            log_val += log_dgamma(W(i, j), 1, 1);
        }
    }
    
    
    // g and rho components are uniform so can disregard
    
    if(iter >= burn_in){
        if((iter == burn_in) || (log_val > max_post_val)){
            max_post_val = log_val;
            U_map = U.cast<double>();
            A_map = A.cast<double>();
            W_map = W;
            g_map = g_params;
        }
    }
    return log_val;
}

long double Gibbs::transition(const int& t, const int& j, const int& next_tau, const int& prev_tau){
    if((next_tau != t-1) & (next_tau != prev_tau)){
        return 0.0;
    }
    long double val = f(t, j, next_tau, j, prev_tau, tau.col(t-1));
    if(t < T-1){
        for(int k=0; k<d; k++){
            if(k == j){
                continue;
            }
            val *= f(t+1, k, tau(k, t+1), j, next_tau, tau.col(t));
        }
    }
    return val;
}

long double Gibbs::f(const int& t, const int& j, const int& tau_j, const int& i, const int& tau_i, const VectorXi& tau_rest){
    long double prob_ = prob(t, j, i, tau_i, tau_rest);
    int tau_prev;
    if(i == j){
        tau_prev = tau_i;
    } else {
        tau_prev = tau_rest(j);
    }

    if(tau_j == t-1){
        return prob_;
    } else if(tau_j == tau_prev){
        return 1-prob_;
    } else {
        return 0.0;
    }
}

long double Gibbs::prob(const int& t, const int& j, const int& i, const int& tau_i, const VectorXi& tau_rest){
    long double p, val = 0, AW_sum=0;
    int prev;
    for(int k=0; k<d; k++){
        if(k == i){
            prev = tau_i;
        } else {
            prev = tau_rest(k);
        }
        p = A(k, j) * W(k, j) * (long double) ((prev > -1) || (k == j));
        val += p * g_ij(k, j, t-prev-1);
        // AW_sum += p;

        // or should AW_sum be simply 
        AW_sum += A(k, j) * W(k, j);
    }
    return val/AW_sum;
}

long double Gibbs::g_ij(const int& i, const int& j, const int& t){
    if(i == j){
        return g_params(i, j);
    } else {
        return g_params(i, j)*std::pow(1-g_params(i, j), t-1);
        // return dpois(t-1, g_params(i, j));
    }
}

void Gibbs::sample_A(std::mt19937_64& rng){

    long double p00, p01, p10, prob_, p_max, u, a=0, b=0;

    for(int i=0; i<d; i++){
        for(int j=i+1; j<d; j++){
            
            p00 = std::log(1-rho), p01 = std::log(rho/2), p10 = std::log(rho/2);
            // (0, 0)
            A(i, j) = 0;
            A(j, i) = 0;
            for(int t=1; t<T; t++){
                if(tau(j, t) != tau(j, t-1)){
                    p00 += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                } else {
                    p00 += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                }
                if(tau(i, t) != tau(i, t-1)){
                    p00 += std::log(prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                } else {
                    p00 += std::log(1 - prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                }
            }
            
            // (1, 0)
            A(i, j) = 1;
            A(j, i) = 0;
            for(int t=1; t<T; t++){
                if(tau(j, t) != tau(j, t-1)){
                    p10 += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                } else {
                    p10 += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                }
                if(tau(i, t) != tau(i, t-1)){
                    p10 += std::log(prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                } else {
                    p10 += std::log(1 - prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                }
            }
            
            // (1, 0)
            A(i, j) = 0;
            A(j, i) = 1;
            for(int t=1; t<T; t++){
                if(tau(j, t) != tau(j, t-1)){
                    p01 += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                } else {
                    p01 += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                }
                if(tau(i, t) != tau(i, t-1)){
                    p01 += std::log(prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                } else {
                    p01 += std::log(1 - prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                }
            }
            
            p_max = std::max({p00, p01, p10});
            
            p00 -= p_max;
            p01 -= p_max;
            p10 -= p_max;
            
            p00 = std::exp(p00);
            p10 = std::exp(p10);
            p01 = std::exp(p01);

            p00 /= (p00 + p10 + p01);
            p10 /= (p00 + p10 + p01);
            p01 /= (p00 + p10 + p01);
            
            u = STD_UNIF(rng);
            if(u < p00){
                A(i, j) = 0;
                A(j, i) = 0;
                b += 1;
            } else if((p00 <= u) & (u < p10)){
                A(i, j) = 1;
                A(j, i) = 0;
                a += 1;
            } else{
                A(i, j) = 0;
                A(j, i) = 1;
                a += 1;
            }
        }
    }
    
    TruncatedBetaDistribution truncated_beta(1 + a, 1 + b, 0, rho_upper);
    rho = truncated_beta();
}

void Gibbs::sample_A_distance(std::mt19937_64& rng){
    long double p00, p01, p10, prob_, p_max, u, a=0, b=0;
    long double dist;
    for(int i=0; i<d; i++){
        for(int j=i+1; j<d; j++){
            dist = std::exp(-tau_dist * distances(i, j));
            p00 = std::log(1-rho*dist), p01 = std::log(rho*dist/2), p10 = std::log(rho*dist/2);
            // (0, 0)
            A(i, j) = 0;
            A(j, i) = 0;
            for(int t=1; t<T; t++){
                if(tau(j, t) != tau(j, t-1)){
                    p00 += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                } else {
                    p00 += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                }
                if(tau(i, t) != tau(i, t-1)){
                    p00 += std::log(prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                } else {
                    p00 += std::log(1 - prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                }
            }
            
            // (1, 0)
            A(i, j) = 1;
            A(j, i) = 0;
            for(int t=1; t<T; t++){
                if(tau(j, t) != tau(j, t-1)){
                    p10 += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                } else {
                    p10 += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                }
                if(tau(i, t) != tau(i, t-1)){
                    p10 += std::log(prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                } else {
                    p10 += std::log(1 - prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                }
            }

            // (1, 0)
            A(i, j) = 0;
            A(j, i) = 1;
            for(int t=1; t<T; t++){
                if(tau(j, t) != tau(j, t-1)){
                    p01 += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                } else {
                    p01 += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                }
                if(tau(i, t) != tau(i, t-1)){
                    p01 += std::log(prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                } else {
                    p01 += std::log(1 - prob(t, i, i, tau(i, t-1), tau.col(t-1)));
                }
            }
            
            p_max = std::max({p00, p01, p10});
            
            p00 -= p_max;
            p01 -= p_max;
            p10 -= p_max;
            
            p00 = std::exp(p00);
            p10 = std::exp(p10);
            p01 = std::exp(p01);

            p00 /= (p00 + p10 + p01);
            p10 /= (p00 + p10 + p01);
            p01 /= (p00 + p10 + p01);
            
            u = STD_UNIF(rng);
            if(u < p00){
                A(i, j) = 0;
                A(j, i) = 0;
                b += 1;
            } else if((p00 <= u) & (u < p10)){
                A(i, j) = 1;
                A(j, i) = 0;
                a += 1;
            } else{
                A(i, j) = 0;
                A(j, i) = 1;
                a += 1;
            }
        }
    }

    // update rho and tau_dist
    long double curr_rho, curr_tau_dist, prop_rho, prop_tau_dist, acc_prob;
    for(int th=0; th<10; th++){
        curr_rho = rho;
        curr_tau_dist = tau_dist;
        prop_rho = curr_rho + 0.05*STD_NORMAL(rng);
        prop_tau_dist = curr_tau_dist + 0.5*STD_NORMAL(rng);

        if((prop_rho < 0) || (prop_rho > rho_upper) || (prop_tau_dist < 0)){
            continue;
        } 

        acc_prob = log_dgamma(prop_tau_dist, 1, 1) - log_dgamma(curr_tau_dist, 1, 1);
        for(int i=0; i<d; i++){
            for(int j=i+1; j<d; j++){
                if((A(i, j) == 1) || (A(j, i) == 1)){
                    acc_prob += std::log(prop_rho) - prop_tau_dist * distances(i, j);
                    acc_prob -= (std::log(curr_rho) - curr_tau_dist * distances(i, j));
                } else {
                    acc_prob += std::log(1-prop_rho*std::exp(-1*prop_tau_dist*distances(i, j)));
                    acc_prob -= std::log(1-curr_rho*std::exp(-1*curr_tau_dist*distances(i, j)));
                }
            }
        }

        if(STD_UNIF(rng) < std::exp(acc_prob)){
            rho = prop_rho;
            tau_dist = prop_tau_dist;
        }

    }
    
}

void Gibbs::sample_W(std::mt19937_64& rng){
    long double curr_W, prop_W, prop_sd=0.75;
    long double accept_prob;
    for(int j=0; j<d; j++){
        for(int i=0; i<d; i++){
            
            if(A(i, j) == 0){ // then no dependence on W(i, j) so just sample from prior
                W(i, j) = rgamma(1, 1, rng);
            } else {
                for(int th=0; th<1; th++){
                    curr_W = W(i, j);
                    prop_W = W(i, j) + prop_sd*STD_NORMAL(rng);
                    if(prop_W < 0){
                        continue;
                    }
                    accept_prob = log_dgamma(prop_W, 1, 1) - log_dgamma(curr_W, 1, 1);
                    
                    W(i, j) = prop_W;
                    for(int t=1; t<T; t++){
                        if(tau(j, t) != tau(j, t-1)){
                            accept_prob += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        } else {
                            accept_prob += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        }
                    }
    
                    W(i, j) = curr_W;
                    for(int t=1; t<T; t++){
                        if(tau(j, t) != tau(j, t-1)){
                            accept_prob -= std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        } else {
                            accept_prob -= std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        }
                    }
                    
                    if(STD_UNIF(rng) < std::exp(accept_prob)){
                        W(i, j) = prop_W;
                    } else {
                        W(i, j) = curr_W;
                    }
                }
            }
            
        }
    }
    // std::cout << W << std::endl;
}

void Gibbs::sample_W_g_joint(std::mt19937_64& rng){
    long double curr_W, prop_W, prop_sd_W=0.5;
    long double curr_g, prop_g, prop_sd_g=0.05;
    // long double curr_g, prop_g, prop_sd_g=1;
    long double accept_prob;
    for(int j=0; j<d; j++){
        for(int i=0; i<d; i++){
            
            if(A(i, j) == 0){ // then no dependence on W(i, j) or g_params(i, j) so just sample from prior
                W(i, j) = rgamma(1, 1, rng);
                g_params(i, j) = 1*STD_UNIF(rng);
            } else {
                for(int th=0; th<15; th++){
                    curr_W = W(i, j);
                    curr_g = g_params(i, j);
                    
                    prop_W = curr_W + prop_sd_W*STD_NORMAL(rng);
                    prop_g = curr_g + prop_sd_g*STD_NORMAL(rng);
                    
                    if((prop_W < 0) || (prop_g < 0) || (prop_g > 1)){
                        continue;
                    }

                    accept_prob = log_dgamma(prop_W, 1, 1) - log_dgamma(curr_W, 1, 1);
                    
                    W(i, j) = prop_W;
                    g_params(i, j) = prop_g;
                    for(int t=1; t<T; t++){
                        if(tau(j, t) != tau(j, t-1)){
                            accept_prob += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        } else {
                            accept_prob += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        }
                    }
                    
                    W(i, j) = curr_W;
                    g_params(i, j) = curr_g;
                    for(int t=1; t<T; t++){
                        if(tau(j, t) != tau(j, t-1)){
                            accept_prob -= std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        } else {
                            accept_prob -= std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        }
                    }
                    
                    if(STD_UNIF(rng) < std::exp(accept_prob)){
                        W(i, j) = prop_W;
                        g_params(i, j) = prop_g;
                    } else {
                        W(i, j) = curr_W;
                        g_params(i, j) = curr_g;
                    }
                }
            }
            
        }
    }
    // std::cout << W << std::endl;
}

void Gibbs::sample_g_params(std::mt19937_64& rng){
    
    long double curr_param, prop_param, prop_sd=0.05;
    long double accept_prob;
    for(int j=0; j<d; j++){
        for(int i=0; i<d; i++){

            if(A(i, j) == 0){ // then no dependence on W(i, j) so just sample from prior
                g_params(i, j) = STD_UNIF(rng);
            } else {
                for(int th=0; th<1; th++){
                    curr_param = g_params(i, j);
                    prop_param = g_params(i, j) + prop_sd*STD_NORMAL(rng);
                    if((prop_param < 0) || (prop_param > 1)){
                        // reject
                        continue;
                    }
                    accept_prob = 0;
                    
                    g_params(i, j) = prop_param;
                    for(int t=1; t<T; t++){
                        if(tau(j, t) != tau(j, t-1)){
                            accept_prob += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        } else {
                            accept_prob += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        }
                    }
                    
                    g_params(i, j) = curr_param;
                    for(int t=1; t<T; t++){
                        if(tau(j, t) != tau(j, t-1)){
                            accept_prob -= std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        } else {
                            accept_prob -= std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                        }
                    }
                    
                    if(STD_UNIF(rng) < std::exp(accept_prob)){
                        g_params(i, j) = prop_param;
                    } else {
                        g_params(i, j) = curr_param;
                    }
                }
            }
            
        }
    }
}

long double Gibbs::log_pred(const int& j, const int& t, const int& tau){
    if(cache_log_pred){
        long double log_df1, log_df2, log_pred;
        std::pair<int, int> key1 = {tau, t};
        std::pair<int, int> key2 = {tau, t-1};
        if(log_pred_cache[j].count(key1) > 0){
            log_df1 = log_pred_cache[j][key1];
        } else {
            log_df1 = log_data_factor(j, tau, t);
            log_pred_cache[j][key1] = log_df1;
        } 
        if(log_pred_cache[j].count(key2) > 0){
            log_df2 = log_pred_cache[j][key2];
        } else {
            log_df2 = log_data_factor(j, tau, t-1);
            log_pred_cache[j][key2] = log_df2;
        }
        log_pred = log_df1 - log_df2;
        return log_pred;
    } else {
        long double log_pred = log_data_factor(j, tau, t) - log_data_factor(j, tau, t-1);
        return log_pred;
    }
}

long double NormalMeanGibbs::log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
    std::pair<int, int> key = {prev_ep, next_ep};
    if(log_pred_cache[j].count(key) > 0){
        return log_pred_cache[j][key];
    } else {
        int start = prev_ep+1;
        int size = next_ep-prev_ep;
        long double n = (long double) size;
        long double y_sum = (Y_sums[j][size+start] - Y_sums[j][start]);
        long double t1 = -0.5*n*std::log(2*PI*sigma2s(j)) + 0.5*std::log(sigma2s(j)/(sigma2s(j)+n*variances(j)));
        long double cached2 = (Y2_sums[j][size+start] - Y2_sums[j][start]) - 2*y_sum + size*means(j);
        long double t2 = -0.5*cached2 / sigma2s(j); // use summaries
        cached2 = std::pow(y_sum - n*means(j), 2);
        long double t3 = 0.5*variances(j)*cached2 / (sigma2s(j)*(n*variances(j)+sigma2s(j)));
        log_pred_cache[j][key] = t1+t2+t3;
        return t1+t2+t3;
    }
    
}

long double LaplaceGibbs::log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
    int start = prev_ep+1;
    int size = next_ep-prev_ep;
    long double n = (long double) size;
    long double y_abs_sum = (Y_abs_sums[j][size+start] - Y_abs_sums[j][start]);
    long double alpha_post = alphas(j) + n;
    long double lambda_post = lambdas(j) + y_abs_sum;
    long double val = -n*std::log(2) + alphas(j)*std::log(lambdas(j)) - std::lgamma(alphas(j));
    val += std::lgamma(alpha_post) - alpha_post*std::log(lambda_post);
    return val;
}

long double ARProcessGibbs::log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
    int start = prev_ep+1;
    int size = next_ep-prev_ep;
    long double n = (long double) size;
    
    Delta.noalias() = (XX_sums[j][size+start] - XX_sums[j][start] + deltas[j]).llt().solve(I_L);
    long double alpha_n = alphas(j) + n/2.0;
    long double yy = YY_sums[j][size+start] - YY_sums[j][start];
    XY.noalias() = XY_sums[j][size+start] - XY_sums[j][start];
    long double beta_n = lambdas(j) + 0.5*(yy - (XY.transpose()*Delta*XY)(0, 0));

    long double val = -0.5*n*std::log(2*PI) + alphas(j)*std::log(lambdas(j)) + std::lgamma(alpha_n) + std::log(Delta.llt().matrixL().determinant()); // note not using 0.5* here as this is llt matrix
    val += -0.5*std::log(deltas[j].diagonal().prod()) - std::lgamma(alphas(j)) - alpha_n*std::log(beta_n);
    return val;
}
