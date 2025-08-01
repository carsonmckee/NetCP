#include "particle_gibbs.h"
#include "particle_gibbs_other.h"

// to compile: 
// g++ -std=c++11 -I "C:/Users/k2259011/OneDrive - King's College London/Documents/Code/EigenLibs" -Ofast -DEIGEN_NO_DEBUG -Wno-deprecated-declarations -o main.exe src/main.cpp src/particle_gibbs.cpp src/particle_utils.cpp src/particle_gibbs_other.cpp

// g++ -std=c++11 -Ofast -DEIGEN_NO_DEBUG -Wno-deprecated-declarations -o main.exe src/main.cpp src/particle_gibbs.cpp src/particle_utils.cpp
// g++ -std=c++11 -I "C:/Users/k2259011/OneDrive - King's College London/Documents/Code/EigenLibs" -Wno-deprecated-declarations -o main.exe src/main.cpp src/particle_gibbs.cpp src/particle_utils.cpp

int main(){
    MatrixXd data = openData("C:/Users/k2259011/OneDrive - King's College London/Documents/particle_mcmc/cp_dat.csv");
    // MatrixXd data = openData("/Users/carsonmckee/rates.csv");
    // data.transposeInPlace();
    int d = data.rows();
    std::cout << "(" << d << ", " << data.cols() << ")" << std::endl;
    
    double sd_base = 3;
    double sd_error = 0.5;
    
    NormalMean sampler(data, 
                       150,
                       0.2,
                       false,
                       MatrixXd::Zero(d, d),
                       VectorXd::Zero(d), 
                       sd_base*sd_base*VectorXd::Ones(d),
                       sd_error*sd_error*VectorXd::Ones(d), 
                       VectorXd::Zero(d));
    
    
    /*
    NormalMeanOther sampler(data, 
                            150, 
                            "GlobalCCP", 
                            VectorXd::Zero(d), 
                            VectorXd::Zero(d), 
                            sd_base*sd_base*VectorXd::Ones(d),
                            sd_error*sd_error*VectorXd::Ones(d));
    */
    
    /*
    ARProcessOther sampler(data, 
                            150, 
                            "NonGlobalCCP", 
                            VectorXd::Zero(d), 
                            1*MatrixXd::Ones(1, d),
                            1*VectorXd::Ones(d),
                            1*VectorXd::Ones(d));
    sampler.sample(5000, 100, -1, false);
    */
    
    /*
    int L = 1;
    ARProcess sampler(
                    data, 
                    150, 
                    0.2, 
                    false,
                    MatrixXd::Zero(d, d),
                    1*MatrixXd::Ones(L, d),
                    1*VectorXd::Ones(d),
                    1*VectorXd::Ones(d), 
                    VectorXd::Zero(d));
    */
    sampler.sample(100, 0);
    
    std::cout << (sampler.A_sum .array() > 0.5) << std::endl << std::endl;
    std::cout << sampler.W_sum << std::endl << std::endl;
    std::cout << sampler.g_params_sum << std::endl << std::endl;
    
    saveData("C:/Users/k2259011/OneDrive - King's College London/Documents/adj.csv", sampler.A_sum.array());
    saveData("C:/Users/k2259011/OneDrive - King's College London/Documents/n_cps.csv", sampler.N_cps.array());
    saveData("C:/Users/k2259011/OneDrive - King's College London/Documents/log_post.csv", sampler.log_post_vals.array());
    saveData("C:/Users/k2259011/OneDrive - King's College London/Documents/cp_map.csv", sampler.U_map.array());
    saveData("C:/Users/k2259011/OneDrive - King's College London/Documents/adj_map.csv", sampler.A_map.array());
    saveData("C:/Users/k2259011/OneDrive - King's College London/Documents/W_map.csv", sampler.W_map.array());
    
    saveData("C:/Users/k2259011/OneDrive - King's College London/Documents/cp_out.csv", sampler.U_sum.array());
    // std::cout << "finished!\n";
    
    
    // std::cout << "finished!\n";
    

    return 0;
}