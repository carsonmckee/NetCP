#include "particle_gibbs_other.h"
#include <unordered_map>
#include <chrono>

using namespace std::chrono;
using namespace Eigen;

ParticleGibbsOther::ParticleGibbsOther(const MatrixXd& Y_, 
                                    const int& n_part_, 
                                    const std::string& model_, 
                                    const VectorXd& y_pred_){
    
    Y = Y_;
    T = Y.cols();
    
    // extra columns holds values of Y to evaluate predicitive density at
    y_pred = y_pred_;
    Y.conservativeResize(Y.rows(), T+1);
    Y.col(T) = y_pred;
    
    d = Y.rows();
    n_parts = n_part_;
    model = model_;
    tau = -1*MatrixXi::Ones(d, T); // initialise with no change-points
    U = MatrixXi::Zero(d, T-1);
    U_sum = MatrixXd::Zero(d, T-1);

    for(int t=0; t<T; t++){
        particles.push_back({-1});
        weights.push_back({1.0});
    }
    
    cache_log_pred = true;
    for(int i=0; i<d; i++){
        std::unordered_map<std::pair<int, int>, long double, hash_pair> cache;
        log_pred_cache.push_back(cache);
    }

    if(model == "BH"){
        probs = 0.01*VectorXd::Ones(d);
    } else if(model == "GlobalCCP"){
        probs = 0.01*VectorXd::Ones(d);
        default_prior_vals(T, d, nu, mu, sigma);
        sigma_inv = sigma.llt().solve(MatrixXd::Identity(d, d));
        prop_sd = 0.05*VectorXd::Ones(d);
    } else if(model == "NonGlobalCCP"){
        probs = 0.01*MatrixXd::Ones(d, T); // last col is for predictive probabilities
        default_prior_vals(T, d, nu, mu, sigma);
        sigma_inv = sigma.llt().solve(MatrixXd::Identity(d, d));
        prop_sd = 0.05*VectorXd::Ones(d, T);
    } else {
        std::cout << "Unrecognised model!";
        std::abort();
    }

}

void ParticleGibbsOther::sample(const int& n_iter, const int& burn_in, const int& seed_, const bool& verbose){

    auto start = high_resolution_clock::now();

    unsigned int seed;
    if(seed_ == -1){
        seed = (unsigned int) time(NULL);
    } else {
        seed = (unsigned int) seed_;
    }
    std::mt19937_64 rng{seed};

    set_up_sampling(n_iter, burn_in);
    pred_score_est = 0.0;

    for(int iter=0; iter<n_iter; iter++){
        
        if(verbose){
            std::cout << "Iter: " << iter << std::endl;
        }
        
        // sampling the taus using particle gibbs
        for(int j=0; j<d; j++){
            filter_forward(j, rng);
            sample_backwards(j, rng);
        }
        
        // set U and design matrix from sampled taus
        for(int j=0; j<d; j++){
            for(int t=1; t<T; t++){
                if(tau(j, t-1) != tau(j, t)){
                    U(j, t-1) = 1;
                } else {
                    U(j, t-1) = 0;
                }
            }
        }
        
        sample_probs(iter, burn_in, rng);
        
        sample_data_factor_params(iter, burn_in, rng);

        if(iter >= burn_in){
            U_sum += U.cast<double>();
            pred_score_est += pred_score();
        }
        // std::cout << "finished storing\n";
    }
    
    U_sum /= (double) (n_iter - burn_in);
    
    pred_score_est = pred_score_est / (long double) (n_iter - burn_in);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Total: " << duration.count() << " ms (or " << duration.count()/1000000 << " seconds)\n";
}

void ParticleGibbsOther::argsort(const std::vector<long double> &array) {
    sorted_indices.resize(array.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });
}

std::pair<long double, std::vector<int>> ParticleGibbsOther::get_resample_inds(const int& t){
    // solves for kappa : sum min(particle[k]/kappa, 1) = n_parts

    std::vector<int> resample_inds;
    
    // sort particle indices 
    argsort(weights[t]);

    int N = weights[t].size();
    long double left_sum = 0.0L;
    long double right_sum = (long double) N;
    long double n_parts_dbl = (long double) n_parts;
    for(int i=0; i<(N-n_parts); i++){
        left_sum += weights[t][sorted_indices[i]];
        right_sum -= 1;
        resample_inds.push_back(sorted_indices[i]);
    }

    for(int i=(N-n_parts); i<N; i++){ // should this not be up to N-1?
        left_sum += weights[t][sorted_indices[i]];
        resample_inds.push_back(sorted_indices[i]);
        right_sum -= 1;
        if(((left_sum + weights[t][sorted_indices[i+1]]) / weights[t][sorted_indices[i+1]] + right_sum - 1) <= n_parts_dbl){
            break;
        }
        
    }
    
    std::sort(resample_inds.begin(), resample_inds.end());
    std::pair<long double, std::vector<int>> out = {left_sum, resample_inds};
    
    return out;
}

void ParticleGibbsOther::strat_resample(const int& t, const long double& K, const std::vector<int>& inds, std::mt19937_64& rng){

    std::vector<int> to_remove;
    long double U = STD_UNIF(rng)*K;
    for(const int& i : inds){
        U -= weights[t][i];
        if(U < 0){
            U += K;
            weights[t][i] = K;
        } else {
            to_remove.push_back(i);
        }
    }

    // remove indices
    for(int i=to_remove.size()-1; i>=0; i--){
        remove_index(weights[t], to_remove[i]);
        remove_index(particles[t], to_remove[i]);
    }
}

void ParticleGibbsOther::cond_strat_resample(const int& t, const long double& K, const long double& n_to_sample, const std::vector<int>& inds, const int& to_keep, std::mt19937_64& rng){
    
    long double w_sum = K * n_to_sample;
    std::vector<int> to_remove;
    long double U_copy;
    std::vector<long double> weights_copy = weights[t];
    if(w_sum < 1e-12){
        // just remove the smallest, have precision issue
        weights[t][to_keep] = K;
        for(int i=0; i<inds.size(); i++){
            if(i != to_keep){
                if(to_remove.size() < (inds.size() - n_to_sample)){
                    to_remove.push_back(inds[i]);
                } else {
                    weights[t][i] = K;
                }
            }
        }
    } else {
        std::vector<long double> cdf = {0.0};
        long double lower, upper;
        for(int i=0; i<inds.size(); i++){
            cdf.push_back(cdf[i] + weights[t][inds[i]]/w_sum);
            if(i == to_keep){
                lower = cdf[i];
                upper = cdf[i+1];
                break;
            }
        }
        
        long double U_star = lower + STD_UNIF(rng)*(upper - lower);
        long double U = U_star - std::floor(n_to_sample*U_star)/(n_to_sample);
        U_copy = U;
        for(const int& i : inds){
            U -= weights[t][i]/w_sum; // printing out the weights later after theyve been modified here
            if(U < 0){
                U += 1.0/n_to_sample;
                weights[t][i] = K;
            } else {
                to_remove.push_back(i);
            }
        }
        
        // sanity check, if above didnt work, just remove manually
        if(((inds.size() - (int) n_to_sample) - to_remove.size()) > 0){
            to_remove.clear();
            int n_to_remove = inds.size() - (int) n_to_sample;
            for(int i=0; i<inds.size(); i++){
                if(n_to_remove == 0){
                    break;
                } else if(i == to_keep){
                    continue;
                } else {
                    to_remove.push_back(inds[i]);
                    n_to_remove -= 1;
                }
            }
        }
    }

    // remove indices
    for(int i=to_remove.size()-1; i>=0; i--){
        remove_index(weights[t], to_remove[i]);
        remove_index(particles[t], to_remove[i]);
    }


}

void ParticleGibbsOther::resample_particles_optimal(const int& j, const int& t, std::mt19937_64& rng){

    std::pair<long double, std::vector<int>> out = get_resample_inds(t);
    long double w_sum = out.first;
    std::vector<int> resample_inds = out.second;

    int n_survived = weights[t].size() - resample_inds.size();
    int n_to_sample = n_parts - n_survived;

    if(n_to_sample > 0){
        int to_keep = -1;
        for(int i=0; i<resample_inds.size(); i++){
            if(resample_inds[i] == keep_ind){
                to_keep = i;
            }
        }

        long double K = w_sum / (long double) n_to_sample;

        if(to_keep == -1){
            // the conditional particle has already survived
            // so use regular stratified resampling
            strat_resample(t, K, resample_inds, rng);
        } else {
            // the conditional particle is in the set to be resampled
            // so use conditional stratified resampling
            cond_strat_resample(t, K, n_to_sample, resample_inds, to_keep, rng);
        }
    }
    
}

void ParticleGibbsOther::filter_forward(const int& j, std::mt19937_64& rng){

    for(int t=1; t<T; t++){
        compute_particle_weights(j, t);
        
        if(particles[t].size() > n_parts){
            // resample_particles(j, t, rng);
            resample_particles_optimal(j, t, rng);
            if(particles[t].size() > n_parts){
                std::cout << "more particles survived than should have!" << std::endl;
                std::cout << particles[t].size() << std::endl;
                std::abort();
            }
        }
    }
}

void ParticleGibbsOther::compute_particle_weights(const int& j, const int&t){
    // updates support from time t-1 and computes the normalized weights (for series j).
    long double weight_sum = 0.0;
    particles[t].resize(particles[t-1].size()); // don't want to resize here, just keep them all at a fixed size (n_parts+1);
    weights[t].resize(particles[t-1].size());
    long double prev_weight, weight;
    int particle;
    long double new_particle_weight = 0;
    long double new_log_pred = log_pred(j, t, t-1);

    for(int k=0; k<particles[t-1].size(); k++){
        // update particles which have remained the same (no cp)
        
        particle = particles[t-1][k];
        prev_weight = weights[t-1][k];
        
        weight = prev_weight * transition(t, j, particle, particle) * log_pred(j, t, particle);
        weight_sum += weight;
        particles[t][k] = particle;
        weights[t][k] = weight;

        // update the weight of the new particle (a cp)
        new_particle_weight += prev_weight*transition(t, j, t-1, particle)*new_log_pred;
    }
    weight_sum += new_particle_weight;

    // add new particle and weight
    particles[t].push_back(t-1);
    weights[t].push_back(new_particle_weight);
    
    // normalise weights
    for(int k=0; k<particles[t].size(); k++){
        weights[t][k] /= weight_sum;
    }
}

long double ParticleGibbsOther::log_p_cond_global(const int& j){
    long double val = 0.0;
    long double log_p = std::log(probs(j)), log_1_p = std::log(1-probs(j));
    for(int t=1; t<T; t++){
        if(tau(j, t) != tau(j, t-1)){
            val += log_p;
        } else {
            val += log_1_p;
        }
    }
    val -= (log_p + log_1_p);
    VectorXd logit_p_mu = logit(probs) - mu;
    val += -0.5*(nu+(long double)d) * std::log(1 + (logit_p_mu.transpose()*sigma_inv*logit_p_mu)(0, 0)/nu);
    return val;
}

void ParticleGibbsOther::sample_probs_global(std::mt19937_64& rng){
    long double curr_p, prop_prob, prop_p;
    for(int j=0; j<d; j++){
        for(int th=0; th<20; th++){
            curr_p = probs(j);
            prop_p = curr_p + prop_sd(j) * STD_NORMAL(rng);
            
            if((prop_p < 0) || (prop_p > 1)){
                // reject 
                probs(j) = curr_p;
            } else {
                probs(j) = prop_p;
                prop_prob = log_p_cond_global(j) + log_dnorm(curr_p, prop_p, prop_sd(j));
                probs(j) = curr_p;
                prop_prob -= (log_p_cond_global(j) + log_dnorm(prop_p, curr_p, prop_sd(j)));
                if(STD_UNIF(rng) < std::exp(prop_prob)){
                    // accept
                    probs(j) = prop_p;
                    // accept_rates.row(j) += 1;
                } else {
                    // reject
                    probs(j) = curr_p;
                }
            }
        }
    }
}

long double ParticleGibbsOther::log_p_cond(const long double& pit, const int& Uit, const VectorXd& logit_p_mu, const MatrixXd& sigma_inv){
    long double val = -std::log(pit);
    if(Uit){
        val += std::log(pit/(1-pit));
    }
    val += -0.5*(nu+(long double)d) * std::log(1 + (logit_p_mu.transpose()*sigma_inv*logit_p_mu)(0, 0)/nu);
    return val;
}

void ParticleGibbsOther::sample_probs_non_global(std::mt19937_64& rng){
    long double curr_p, prop_p, prop_prob;
    int Ujt;
    for(int t=0; t<T-1; t++){
        for(int j=0; j<d; j++){
            for(int th=0; th<20; th++){
                curr_p = probs(j, t);
                prop_p = curr_p + prop_sd(j) * STD_NORMAL(rng);
                if((prop_p < 0) || (prop_p > 1)){
                    // reject 
                    probs(j, t) = curr_p;
                } else {
                    if(tau(j, t) != tau(j, t+1)){
                        Ujt = 1;
                    } else {
                        Ujt = 0;
                    }
                    probs(j, t) = prop_p;
                    prop_prob = log_p_cond(prop_p, Ujt, (logit(probs.col(t)) - mu), sigma_inv) + log_dnorm(curr_p, prop_p, prop_sd(j));
                    probs(j, t) = curr_p;
                    prop_prob -= (log_p_cond(curr_p, Ujt, (logit(probs.col(t)) - mu), sigma_inv) + log_dnorm(prop_p, curr_p, prop_sd(j)));
                    if(STD_UNIF(rng) < std::exp(prop_prob)){
                        // accept
                        probs(j, t) = prop_p;
                        // accept_rates(j, t) += 1;
                    } else {
                        // reject
                        probs(j, t) = curr_p;
                    }
                }
            }
        }
    }
    
    // update predictive probs
    for(int j=0; j<d; j++){
        for(int th=0; th<20; th++){
            curr_p = probs(j, T-1);
            prop_p = curr_p + prop_sd(j) * STD_NORMAL(rng);
            if((prop_p < 0) || (prop_p > 1)){
                // reject 
                probs(j, T-1) = curr_p;
            } else {
                probs(j, T-1) = prop_p;
                prop_prob = log_p_cond(prop_p, 1, (logit(probs.col(T-1)) - mu), sigma_inv) + log_dnorm(curr_p, prop_p, prop_sd(j)) - std::log(prop_p);
                probs(j, T-1) = curr_p;
                prop_prob -= (log_p_cond(curr_p, 1, (logit(probs.col(T-1)) - mu), sigma_inv) + log_dnorm(prop_p, curr_p, prop_sd(j)) - std::log(curr_p));
                if(STD_UNIF(rng) < std::exp(prop_prob)){
                    // accept
                    probs(j, T-1) = prop_p;
                } else {
                    // reject
                    probs(j, T-1) = curr_p;
                }
            }
        }
    }
}

void ParticleGibbsOther::sample_probs(const int& iter, const int& burn_in, std::mt19937_64& rng){
    
    if(model == "BH"){
        int b;
        for(int j=0; j<d; j++){
            b = U.row(j).sum() + 1;
            probs(j) = rbeta(b, T+b+1, rng);
            // std::cout << "b= " << b <<  ", prob(j) = " << probs(j) << std::endl;
        }
    } else if (model == "GlobalCCP"){
        sample_probs_global(rng);
    } else {
        sample_probs_non_global(rng);
    }
}

long double ParticleGibbsOther::pred_score(){
    // first compute probs (different based on model)
    long double val = 1.0;
    long double val_j;
    if((model == "BH") || (model == "GlobalCCP")){
        for(int j=0; j<d; j++){
            val_j = probs(j) * log_pred(j, T, T-1) + (1-probs(j))*log_pred(j, T, tau(j, T-1));
            val *= val_j;
        }
    } else {
        for(int j=0; j<d; j++){
            val_j = probs(j, T-1) * log_pred(j, T, T-1) + (1-probs(j, T-1))*log_pred(j, T, tau(j, T-1));
            val *= val_j;
        }
    }

    return val;
}

void ParticleGibbsOther::sample_backwards(const int& j, std::mt19937_64& rng){
    int choice = rcat(weights[T-1], rng);
    tau(j, T-1) = particles[T-1][choice];
    long double weight_sum;
    int upper_ep = T-1;
    
    for(int t=T-2; t>=1; t--){
        weight_sum = 0;
        
        for(int k=0; k<particles[t].size(); k++){
            weights[t][k] *= transition(t+1, j, tau(j, t+1), particles[t][k]);
            weight_sum += weights[t][k];
        }
        
        for(int k=0; k<particles[t].size(); k++){
            weights[t][k] /= weight_sum;
        }
        
        choice = rcat(weights[t], rng);
        tau(j, t) = particles[t][choice];
        if(tau(j, t) == t-1){
            upper_ep = t-1;
        }
    }
}

void ParticleGibbsOther::resample_particles(const int& j, const int& t, std::mt19937_64& rng){
    // resample conditional on the current value tau(j, t);
    int keep_particle = tau(j, t);
    int choice, particle;
    std::discrete_distribution<int> dist(weights[t].begin(), weights[t].end());
    std::unordered_map<int, long double> new_weights;
    long double w = 1.0/(long double)n_parts;
    new_weights[keep_particle] = w;
    for(int i=0; i<n_parts-1; i++){
        choice = dist(rng);
        particle = particles[t][choice];
        if(new_weights.count(particle) == 0){
            new_weights[particle] = w;
        } else {
            new_weights[particle] += w;
        }
    }
    weights[t].clear();
    particles[t].clear();
    for(const std::pair<int, long double> pw : new_weights){
        particles[t].push_back(pw.first);
        weights[t].push_back(pw.second);
    }
}

long double ParticleGibbsOther::transition(const int& t, const int& j, const int& next_tau, const int& prev_tau){
    if((next_tau != t-1) & (next_tau != prev_tau)){
        return 0.0;
    }
    long double prob;
    if((model == "BH") || (model == "GlobalCCP")){
        prob = probs(j);
    } else {
        prob = probs(j, t-1);
    }
    if(next_tau == prev_tau){
        return 1-prob;
    } else {
        return prob;
    }
}

// log predictive distribution
long double ParticleGibbsOther::log_pred(const int& j, const int& t, const int& tau){
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
        return std::exp(log_pred);
    } else {
        long double log_pred = log_data_factor(j, tau, t) - log_data_factor(j, tau, t-1);
        return std::exp(log_pred);
    }
}

long double NormalMeanOther::log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
    int start = prev_ep+1;
    int size = next_ep-prev_ep;
    long double n = (long double) size;
    long double y_sum = (Y_sums[j][size+start] - Y_sums[j][start]);
    long double t1 = -0.5*n*std::log(2*PI*sigma2s(j)) + 0.5*std::log(sigma2s(j)/(sigma2s(j)+n*variances(j)));
    long double cached2 = (Y2_sums[j][size+start] - Y2_sums[j][start]) - 2*y_sum + n*means(j);
    long double t2 = -0.5*cached2 / sigma2s(j); // use summaries
    // cached2 = std::pow(y_sum - n*means(j), 2);
    cached2 = (y_sum - n*means(j))*(y_sum - n*means(j));
    long double t3 = 0.5*variances(j)*cached2 / (sigma2s(j)*(n*variances(j)+sigma2s(j)));
    return t1+t2+t3;
}

long double LaplaceOther::log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
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

long double ARProcessOther::log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
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