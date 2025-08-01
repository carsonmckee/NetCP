#include "../src/particle_gibbs.h"

// to compile: 
// g++ -std=c++11 -I "../src" -Ofast -DEIGEN_NO_DEBUG -Wno-deprecated-declarations -o eeg.exe eeg.cpp ../src/particle_gibbs.cpp ../src/particle_utils.cpp
// to run (seocnd number is the identifier for the MCMC output): ./eeg.exe 100

int main(int argc, char *argv[]){

    std::string id = argv[1];

    MatrixXd data = openData("eeg_dat.csv");
    int d = data.rows();
    std::cout << "(" << d << ", " << data.cols() << ")" << std::endl;
    
    int L = 1; // AR(1)
    ARProcess sampler(
                    data, 
                    500,
                    0.2, 
                    false,
                    MatrixXd::Zero(5, 5),
                    1*MatrixXd::Ones(L, d),
                    1*VectorXd::Ones(d),
                    1*VectorXd::Ones(d), 
                    VectorXd::Zero(d));
    
    sampler.sample(20000, 2000);
    
    std::cout << sampler.A_sum << std::endl << std::endl;
    std::cout << sampler.W_sum << std::endl << std::endl;
    std::cout << sampler.g_params_sum << std::endl << std::endl;
    
    saveData("results/out_" + id + ".csv", sampler.U_sum.array());
    saveData("results/W_" + id + ".csv", sampler.W_sum.array());
    saveData("results/adj_" + id + ".csv", sampler.A_sum.array());
    saveData("results/n_cps_" + id + ".csv", sampler.A_sum.array());
    
    saveData("results/out_map_" + id + ".csv", sampler.U_map.array());
    saveData("results/W_map_" + id + ".csv", sampler.W_map.array());
    saveData("results/adj_map_" + id + ".csv", sampler.A_map.array());
    
    std::cout << "finished\n";
    return 0;
}