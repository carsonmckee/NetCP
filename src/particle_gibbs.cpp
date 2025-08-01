#include "particle_gibbs.h"
#include <unordered_map>

using namespace std::chrono;
using namespace Eigen;

std::pair<ArrayXd, ArrayXd> get_grid(const int& K, const long double& a, const long double& b){
    unsigned int seed = (unsigned int) time(NULL);
    // unsigned int seed = 1;
    std::mt19937_64 rng{seed};

    ArrayXd out = ArrayXd::Zero(K);
    ArrayXd diffs = ArrayXd::Zero(K-1);
    // sample us
    std::vector<long double> u(K);
    long double dk = 1.0/(long double)K;
    for(int k=0; k<K; k++){
        u[k] = k*dk + dk*STD_UNIF(rng);
    }
    // replace below with qgamma when using R cpp
    long double dx = 10e-8;
    long double v = 10e-50;
    int ind = 0;
    long double csum = 0.0;
    while(ind != K){
        csum += dx*std::exp(log_dgamma(v, a, b));
        if(csum > u[ind]){
            out(ind) = v;
            ind += 1;
        }
        v = v + dx;
    }
    diffs = out.segment(1, K-1) - out.segment(0, K-1);
    std::pair<ArrayXd, ArrayXd> val = {out, diffs};
    return val;
}

ParticleGibbs::ParticleGibbs(const MatrixXd& Y_, 
                             const int& n_parts_,
                             const long double& rho_upper_, 
                             const bool& use_distance_,
                             const MatrixXd& Z_,
                             const VectorXd& Y_pred_){
    
    Y = Y_;
    T = Y.cols();
    Y.conservativeResize(Y.rows(), Y.cols()+1);
    Y.col(T) = Y_pred_;
    d = Y.rows();
    n_parts = n_parts_;
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

void ParticleGibbs::sample(const int& n_iter, const int& burn_in){

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
        // sample hidden states via particle gibbs
        for(int j=0; j<d; j++){
            filter_forward(j, rng);
            sample_backwards(j, rng);
        }
        
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

long double ParticleGibbs::check_is_map(const int& iter, const int& burn_in){
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

void ParticleGibbs::argsort(const std::vector<long double> &array) {
    sorted_indices.resize(array.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });
}

std::pair<long double, std::vector<int>> ParticleGibbs::get_resample_inds(const int& t){
    // solves for kappa : sum min(particle[k]/kappa, 1) = n_parts
    
    std::vector<int> resample_inds;
    
    // sort particle indices 
    argsort(weights[t]); // not ideal O(n log n)

    int N = weights[t].size();
    long double left_sum = 0.0L;
    long double right_sum = (long double) N;
    long double n_parts_dbl = (long double) n_parts;
    for(int i=0; i<(N-n_parts); i++){
        left_sum += weights[t][sorted_indices[i]];
        right_sum -= 1;
        resample_inds.push_back(sorted_indices[i]);
    }

    for(int i=(N-n_parts); i<N; i++){
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

void ParticleGibbs::strat_resample(const int& t, const long double& K, const std::vector<int>& inds, std::mt19937_64& rng){

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

void ParticleGibbs::cond_strat_resample(const int& t, const long double& K, const long double& n_to_sample, const std::vector<int>& inds, const int& to_keep, std::mt19937_64& rng){
    std::vector<long double> cdf = {0.0};
    long double w_sum = K * n_to_sample;
    std::vector<int> to_remove;

    if(w_sum < 1e-12){ // if weight_sum is around this then we are almost out of precision
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
        long double U_copy = U;
        for(const int& i : inds){
            U -= weights[t][i]/w_sum; // printing out the weights later after theyve been modified here
            if(U < 0){
                U += 1.0/n_to_sample;
                weights[t][i] = K;
            } else {
                to_remove.push_back(i);
            }
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
    
    // remove indices
    for(int i=to_remove.size()-1; i>=0; i--){
        remove_index(weights[t], to_remove[i]);
        remove_index(particles[t], to_remove[i]);
    }
    
}

void ParticleGibbs::resample_particles_optimal(const int& j, const int& t, std::mt19937_64& rng){

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

void ParticleGibbs::filter_forward(const int& j, std::mt19937_64& rng){

    for(int t=1; t<T; t++){
        compute_particle_weights(j, t);
        
        if(particles[t].size() > n_parts){
            resample_particles_optimal(j, t, rng);
            
            if(particles[t].size() > n_parts){
                std::cout << "More particles survived than should have! Possible precision issues." << std::endl;
                int n_to_remove = n_parts - particles[t].size();
                std::cout << particles[t].size() << std::endl;
                
                std::abort();
            }
        }
    }
}

/*
// optimized version
void ParticleGibbs::compute_particle_weights(const int& j, const int&t){

    // updates support from time t-1 and computes the normalized weights (for series j).
    long double weight_sum = 0.0;
    particles[t].resize(particles[t-1].size()); // don't want to resize here, just keep them all at a fixed size (n_parts+1);
    weights[t].resize(particles[t-1].size());
    long double prev_weight, weight;
    int particle;
    long double new_particle_weight = 0;
    long double new_log_pred = std::exp(log_pred(j, t, t-1));

    long double transition_t_change = prob(t, j, j, t-1, tau.col(t-1));
    long double transition_t_no_change = 1 - transition_t_change;
    long double transition_t1_change = 1.0;
    if((dependent_series[j].size() > 0) & (t < T-1)){
        for(const int& i: dependent_series[j]){
            if(tau(i, t+1) != tau(i, t)){
                transition_t1_change *= prob(t+1, i, j, t-1, tau.col(t));
            } else {
                transition_t1_change *= (1 - prob(t+1, i, j, t-1, tau.col(t)));
            }
        }
    }
    
    for(int k=0; k<particles[t-1].size(); k++){
        // update particles which have remained the same (no cp)
        // no self exciting behaviour so no dependence on prev particle in transition.
        // only need to update the effect of the currnet particle at t+1 in other series which are connected to j
        particle = particles[t-1][k];
        prev_weight = weights[t-1][k];
        
        weight = prev_weight*transition_t_no_change*std::exp(log_pred(j, t, particle));
        if((dependent_series[j].size() > 0) & (t < T-1) ){
            for(const int& i: dependent_series[j]){
                // each of these prob statements involves a sum of length d when 
                // only need to sum over edge set
                if(tau(i, t+1) != tau(i, t)){
                    weight *= prob(t+1, i, j, particle, tau.col(t));
                } else {
                    weight *= (1 - prob(t+1, i, j, particle, tau.col(t)));
                }
            }
        }
        weight_sum += weight;
        particles[t][k] = particle;
        weights[t][k] = weight;
        
        // update the weight of the new particle (a cp)
        new_particle_weight += prev_weight*transition_t_change*transition_t1_change*new_log_pred;
        
        if(tau(j, t) == particle){
            // particle to keep when doing resampling
            keep_ind = k;
        }

    }
    weight_sum += new_particle_weight;

    // add new particle and weight
    particles[t].push_back(t-1);
    weights[t].push_back(new_particle_weight);

    // particle to keep when doing resampling
    if(tau(j, t) == t-1){
        keep_ind = particles[t].size() - 1;
    }
    
    // normalise weights
    for(int k=0; k<particles[t].size(); k++){
        weights[t][k] /= weight_sum;
    }
}
*/

// optimized version 2
void ParticleGibbs::compute_particle_weights(const int& j, const int&t){

    // updates support from time t-1 and computes the normalized weights (for series j).
    long double weight_sum = 0.0;
    particles[t].resize(particles[t-1].size()); // don't want to resize here, just keep them all at a fixed size (n_parts+1);
    weights[t].resize(particles[t-1].size());
    long double prev_weight, weight;
    int particle;
    long double new_particle_weight = 0;
    long double new_log_pred = std::exp(log_pred(j, t, t-1));
    
    long double transition_t_change = prob(t, j, j, t-1, tau.col(t-1));
    long double transition_t_no_change = 1 - transition_t_change;
    long double transition_t1_change = 1.0;

    ArrayXd transition_t1_no_change_unnorm(dependent_series.size());

    if((dependent_series[j].size() > 0) & (t < T-1)){
        for(const int& i: dependent_series[j]){

            if(tau(i, t+1) != tau(i, t)){
                transition_t1_change *= prob(t+1, i, j, t-1, tau.col(t));
            } else {
                transition_t1_change *= (1 - prob(t+1, i, j, t-1, tau.col(t)));
            }
        }
    }
    
    for(int k=0; k<particles[t-1].size(); k++){
        // update particles which have remained the same (no cp)
        // no self exciting behaviour so no dependence on prev particle in transition.
        // only need to update the effect of the currnet particle at t+1 in other series which are connected to j
        particle = particles[t-1][k];
        prev_weight = weights[t-1][k];
        
        weight = prev_weight*transition_t_no_change*std::exp(log_pred(j, t, particle));
        if((dependent_series[j].size() > 0) & (t < T-1) ){
            for(const int& i: dependent_series[j]){
                // each of these prob statements involves a sum of length d when 
                // only need to sum over edge set
                // can replace the below probs with a scheme that only adds subtracts the relevant 
                // part of the probability then normalizes
                
                // subtract previous value of g_ji(t+1-particle-1) * (long double) (particle > -1) : stored as g_prev[ind]
                // add curr_value of g_ji(t+1-particle-1) * (long double) (particle > -1)
                // update weight with the normalized probability
                if(tau(i, t+1) != tau(i, t)){
                    weight *= prob(t+1, i, j, particle, tau.col(t));
                } else {
                    weight *= (1 - prob(t+1, i, j, particle, tau.col(t)));
                }
            }
        }
        weight_sum += weight;
        particles[t][k] = particle;
        weights[t][k] = weight;
        
        // update the weight of the new particle (a cp)
        new_particle_weight += prev_weight*transition_t_change*transition_t1_change*new_log_pred;
        
        if(tau(j, t) == particle){
            // particle to keep when doing resampling
            keep_ind = k;
        }

    }
    weight_sum += new_particle_weight;

    // add new particle and weight
    particles[t].push_back(t-1);
    weights[t].push_back(new_particle_weight);

    // particle to keep when doing resampling
    if(tau(j, t) == t-1){
        keep_ind = particles[t].size() - 1;
    }
    
    // normalise weights
    for(int k=0; k<particles[t].size(); k++){
        weights[t][k] /= weight_sum;
    }
}


/*
void ParticleGibbs::compute_particle_weights(const int& j, const int&t){

    // need to do optimized version of this
    // -> only one component of the probability is changing with each particle
    // -> only need to sum over components i with A(i, j) = 1

    // updates support from time t-1 and computes the normalized weights (for series j).
    long double weight_sum = 0.0;
    particles[t].resize(particles[t-1].size()); // don't want to resize here, just keep them all at a fixed size (n_parts+1);
    weights[t].resize(particles[t-1].size());
    long double prev_log_weight, weight;
    int particle;
    long double new_particle_weight = 0;
    long double new_log_pred = std::exp(log_pred(j, t, t-1));

    for(int k=0; k<particles[t-1].size(); k++){
        // update particles which have remained the same (no cp)
        // no self exciting behaviour so no dependence on prev particle in transition.
        // only need to update the effect of the currnet particle at t+1 in other series which are connected to j
        particle = particles[t-1][k];
        prev_log_weight = weights[t-1][k];
        
        weight = prev_log_weight*transition(t, j, particle, particle)*std::exp(log_pred(j, t, particle));
        weight_sum += weight;
        particles[t][k] = particle;
        weights[t][k] = weight;
        
        // update the weight of the new particle (a cp)
        new_particle_weight += prev_log_weight*transition(t, j, t-1, particle)*new_log_pred;
        
        if(tau(j, t) == particle){
            keep_ind = k;
        }

    }
    weight_sum += new_particle_weight;

    // add new particle and weight
    particles[t].push_back(t-1);
    weights[t].push_back(new_particle_weight);

    if(tau(j, t) == t-1){
        keep_ind = particles[t].size() - 1;
    }
    
    // normalise weights
    for(int k=0; k<particles[t].size(); k++){
        weights[t][k] /= weight_sum;
    }
}
*/

long double ParticleGibbs::transition(const int& t, const int& j, const int& next_tau, const int& prev_tau){
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

long double ParticleGibbs::f(const int& t, const int& j, const int& tau_j, const int& i, const int& tau_i, const VectorXi& tau_rest){
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

long double ParticleGibbs::prob(const int& t, const int& j, const int& i, const int& tau_i, const VectorXi& tau_rest){
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

void ParticleGibbs::sample_backwards(const int& j, std::mt19937_64& rng){
    int choice = rcat(weights[T-1], rng);
    tau(j, T-1) = particles[T-1][choice];
    long double weight_sum;
    int upper_ep = T-1;
    
    // only need to consider particle which have non zero probability of going from t -> t+1
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

void ParticleGibbs::resample_particles(const int& j, const int& t, std::mt19937_64& rng){
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

long double dpois(const long double& x, const long double& lam){
    long double val = x*std::log(lam) - lam - std::lgamma(x+1);
    return std::exp(val);
}

long double ParticleGibbs::g_ij(const int& i, const int& j, const int& t){
    if(i == j){
        return g_params(i, j);
    } else {
        return g_params(i, j)*std::pow(1-g_params(i, j), t-1);
        // return dpois(t-1, g_params(i, j));
    }
}

void ParticleGibbs::sample_A(std::mt19937_64& rng){

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

void ParticleGibbs::sample_A_distance(std::mt19937_64& rng){
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

/*
void ParticleGibbs::sample_A(std::mt19937_64& rng){
    
    long double p0, p1, prob_, p_max, u, a=0, b=0;

    for(int i=0; i<d; i++){
        for(int j=0; j<d; j++){
            if(i == j){
                continue;
            }

            p0 = std::log(1-rho), p1 = std::log(rho);
            // (0)
            A(i, j) = 0;
            for(int t=1; t<T; t++){
                if(tau(j, t) != tau(j, t-1)){
                    p0 += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                } else {
                    p0 += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                }
            }
            
            // (1)
            A(i, j) = 1;
            for(int t=1; t<T; t++){
                if(tau(j, t) != tau(j, t-1)){
                    p1 += std::log(prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                } else {
                    p1 += std::log(1 - prob(t, j, j, tau(j, t-1), tau.col(t-1)));
                }
            }
            
            p_max = std::max({p0, p1});
            
            p0 -= p_max;
            p1 -= p_max;
            
            p0 = std::exp(p0);
            p1 = std::exp(p1);

            p0 /= (p0 + p1);
            p1 /= (p0 + p1);
            
            u = STD_UNIF(rng);
            if(u < p0){
                A(i, j) = 0;
                b += 1;
            } else{
                A(i, j) = 1;
                a += 1;
            }
        }
    }

    // update rho
    rho = rbeta(1 + a, 1 + b, rng);
    while(rho > rho_upper){
        rho = rbeta(1 + a, 1 + b, rng);
    }
    std::cout << A.sum() - d << ", " << rho << std::endl;

}
*/

void ParticleGibbs::sample_W(std::mt19937_64& rng){
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

void ParticleGibbs::sample_W_g_joint(std::mt19937_64& rng){
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

void ParticleGibbs::sample_g_params(std::mt19937_64& rng){
    
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

long double ParticleGibbs::log_pred(const int& j, const int& t, const int& tau){
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

long double NormalMean::log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
    int start = prev_ep+1;
    int size = next_ep-prev_ep;
    long double n = (long double) size;
    long double y_sum = (Y_sums[j][size+start] - Y_sums[j][start]);
    long double t1 = -0.5*n*std::log(2*PI*sigma2s(j)) + 0.5*std::log(sigma2s(j)/(sigma2s(j)+n*variances(j)));
    long double cached2 = (Y2_sums[j][size+start] - Y2_sums[j][start]) - 2*y_sum + size*means(j);
    long double t2 = -0.5*cached2 / sigma2s(j); // use summaries
    cached2 = std::pow(y_sum - n*means(j), 2);
    long double t3 = 0.5*variances(j)*cached2 / (sigma2s(j)*(n*variances(j)+sigma2s(j)));
    return t1+t2+t3;
}

long double Laplace::log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
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

long double ARProcess::log_data_factor(const int&j, const int& prev_ep, const int& next_ep){
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

long double StateSpace::log_data_factor(const int& j, const int& prev_ep, const int& next_ep){

    long double mu_mean = mu_means(j);
    long double mu_var = mu_variances(j);
    long double a = alphas(j);
    long double b = betas(j);
    long double s2 = sigma2s(j);
    
    long double tau = 0.1*0.1;

    long double val, int_val, prev_int_val;
    long double x_kk;
    long double P_kk;
    long double S_k, K_k, x_kk1, P_kk1, y_k;
    long double integral = 0;
    
    for(int h=0; h<tau_grid.size(); h++){
        tau = tau_grid(h);
        val = log_dgamma(tau, a, b);

        x_kk = mu_mean;
        P_kk = mu_var;
        S_k, K_k, x_kk1, P_kk1, y_k;
        for(int k=prev_ep+1; k<next_ep; k++){
            x_kk1 = x_kk;
            P_kk1 = P_kk + tau;
            S_k = P_kk1 + s2;
            K_k = P_kk1 / S_k;
            P_kk = (1-K_k) * P_kk1;
            y_k = Y(j, k) - x_kk1;
            x_kk = x_kk1 + K_k*y_k;
            val += log_dnorm(Y(j, k), x_kk1, std::sqrt(S_k));
        }
        
        int_val = std::exp(val);
        if(h > 0){
            integral += 0.5*(tau_grid(h) - tau_grid(h-1)) * (prev_int_val + int_val);
        }
        prev_int_val = int_val;
    }
    /*
    val = 0.0;

    x_kk = mu_mean;
    P_kk = mu_var;
    for(int k=prev_ep+1; k<next_ep; k++){
        x_kk1 = x_kk;
        P_kk1 = P_kk + tau;
        S_k = P_kk1 + s2;
        K_k = P_kk1 / S_k;
        P_kk = (1-K_k) * P_kk1;
        y_k = Y(j, k) - x_kk1;
        x_kk = x_kk1 + K_k*y_k;
        val += log_dnorm(Y(j, k), x_kk1, std::sqrt(S_k));
    }
    */
    
    return std::log(integral);
}