#include "../src/gibbs.h"
#include "../src/particle_gibbs.h"

// to compile: 
// g++ -std=c++11 -I "../src" -Ofast -DEIGEN_NO_DEBUG -Wno-deprecated-declarations -o gibbs_main.exe ../src/gibbs_main.cpp ../src/particle_gibbs.cpp ../src/gibbs.cpp ../src/particle_utils.cpp

int main(){
    MatrixXd data = openData("cp_dat.csv");
    int d = data.rows();
    std::cout << "(" << d << ", " << data.cols() << ")" << std::endl;
    
    double sd_base = 3;
    double sd_error = 0.5;
    int n_iter = 20000;
    int burn_in = 0;
    
    NormalMeanGibbs gibbs(data, 
                       0.2,
                       false,
                       MatrixXd::Zero(d, d),
                       VectorXd::Zero(d), 
                       sd_base*sd_base*VectorXd::Ones(d),
                       sd_error*sd_error*VectorXd::Ones(d), 
                       VectorXd::Zero(d));
    gibbs.sample(n_iter, burn_in);
    
    NormalMean pg50(data, 
        50,
        0.2,
        false,
        MatrixXd::Zero(d, d),
        VectorXd::Zero(d), 
        sd_base*sd_base*VectorXd::Ones(d),
        sd_error*sd_error*VectorXd::Ones(d), 
        VectorXd::Zero(d));
    pg50.sample(n_iter, burn_in);
    
    NormalMean pg100(data, 
        100,
        0.2,
        false,
        MatrixXd::Zero(d, d),
        VectorXd::Zero(d), 
        sd_base*sd_base*VectorXd::Ones(d),
        sd_error*sd_error*VectorXd::Ones(d), 
        VectorXd::Zero(d));
    pg100.sample(n_iter, burn_in);
    
    saveData("gibbs_cp_out.csv", gibbs.U_sum.array());
    saveData("pg50_cp_out.csv", pg50.U_sum.array());    
    saveData("pg100_cp_out.csv", pg100.U_sum.array());
    
    saveData("gibbs_saved.csv", gibbs.saved_val.array());
    saveData("pg50_saved.csv", pg50.saved_val.array());    
    saveData("pg100_saved.csv", pg100.saved_val.array());
    

    std::cout << "finished!\n";
    
    return 0;
}