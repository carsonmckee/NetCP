#include "particle_gibbs.h"
#include "particle_utils.h"
#include "particle_gibbs_other.h"
#include <string>
#include <iostream>

// Runs simulation study on KCL CREATE cluster.
// On a particular task we supply the model name and task number

// We have 5 scenarios, 50 datasets for each and T=500 observations in each dataset.
// Model must be re-run on each dataset for eact t=1, ..., T
// If we bucket 10 runs on each cpu, we get 5 * 50 * (500 / 10) = 12500 cpu tasks
// Must then take task id (integer) and work out which:
// 1. Scenario number (1-5) 
// 2. Dataset (1-50)
// 3. Which indices to run for (start, end)
// remember all starts at zero!

// Then run sampler for 4000 iterations ( + 500 burn_in) with 150 particles

// run below from sim_study directory (local)
// g++ -std=c++11 -I "C:/Users/k2259011/OneDrive - King's College London/Documents/Code/EigenLibs" -Ofast -DEIGEN_NO_DEBUG -Wno-deprecated-declarations -o run_task_normal.exe src/run_grid_sim_normal.cpp src/particle_gibbs.cpp src/particle_utils.cpp src/particle_gibbs_other.cpp

// run below from sim_study directory on grid (ensure eigen is loaded first)
// g++ -std=c++11 -Ofast -DEIGEN_NO_DEBUG -Wno-deprecated-declarations -o run_task_normal.exe src/run_grid_sim_normal.cpp src/particle_gibbs.cpp src/particle_utils.cpp src/particle_gibbs_other.cpp

// to transfer files to and from grid:

// to: scp -o MACs=hmac-sha2-512 -r code create:/users/k21168451/

// from: scp -o MACs=hmac-sha2-512 -r create:/users/k21168451/sim_study/results_clean code/sim_study/results_clean

/*
#!/bin/bash -l
#SBATCH --job-name=sim_normal
#SBATCH --time=0-15:00
#SBATCH --output=log_normal/log.%A.%a.txt
#SBATCH --array=0-12499:1
##SBATCH --array=5645,5646,5647,5648,5649,6686,6687,6688,6689,6690
cd /users/k21168451/sim_study
ind=$SLURM_ARRAY_TASK_ID

/usr/bin/time -v ./run_task_normal.exe $ind BH 4500 500

/usr/bin/time -v ./run_task_normal.exe $ind GlobalCCP 4500 500

/usr/bin/time -v ./run_task_normal.exe $ind NonGlobalCCP 4500 500

/usr/bin/time -v ./run_task_normal.exe $ind NetCP 4500 500
*/

int main(int argc, char *argv[]){

    int task_id = std::stoi(argv[1]);
    std::string model_name = argv[2];
    int n_iter = std::stoi(argv[3]);
    int burn_in = std::stoi(argv[4]);

    std::cout << "Task number: " << task_id << std::endl;
    std::cout << "Model name: " << model_name << std::endl;
    
    // work out scenario and dataset numbers 
    int scenario_number = task_id / (50 * (500 / 10));
    int dataset_number = (task_id % (50 * (500 / 10))) / (500 / 10);
    int slice_number = (task_id % (50 * (500 / 10))) % (500 / 10);
    int start_number = 10 * slice_number + 1;
    int end_number = 10 * (slice_number + 1) + 1;
    std::cout << "Scenario number = " << scenario_number << std::endl; 
    std::cout << "Dataset number = " << dataset_number << std::endl; 
    std::cout << "Slice number = " << slice_number << std::endl; 
    std::cout << "Slice = [" << start_number << ", " << end_number << ")" << std::endl; 

    std::string data_path = "sim_study/sim_data/normal_mean/scenario_" + std::to_string(scenario_number+1) + "_" + std::to_string(dataset_number+1) + ".csv";

    std::cout << "Data path = " << data_path << std::endl;

    MatrixXd Y, data = openData(data_path);
    VectorXd Y_pred;
    std::cout << data.rows() << ", " << data.cols() << std::endl;

    ArrayXd y_pred_vals(10);

    int d = 4;

    // start_running_model for each index
    for(int t=start_number; t<end_number; t++){
        Y = data.leftCols(t); // up to col t-1 (run model on this)
        Y_pred = data.col(t); // col t (ask model for posterior predictive density at this)
        double sd_base = 3;
        double sd_error = 0.5;
        
        if(model_name == "NetCP"){
            NormalMean sampler(Y, 
                150,
                0.2,
                false,
                MatrixXd::Zero(d, d),
                VectorXd::Zero(d), 
                sd_base*sd_base*VectorXd::Ones(d),
                sd_error*sd_error*VectorXd::Ones(d), 
                Y_pred);
            
            sampler.sample(n_iter, burn_in);
            y_pred_vals(t-start_number) = sampler.est_pred_score;
            // save one of the posterior adjacency matrices for the link test
            if(slice_number == 49){
                std::string results_path = "sim_study/sim_results/normal_mean/adj/" + model_name + "_" + std::to_string(scenario_number+1) + "_" + std::to_string(dataset_number+1) + ".csv";
                saveData(results_path, sampler.A_sum);
            }
            
        } else if(model_name == "BH"){
            NormalMeanOther sampler(Y, 
                                    150, 
                                    "BH", 
                                    Y_pred,
                                    VectorXd::Zero(d), 
                                    sd_base*sd_base*VectorXd::Ones(d),
                                    sd_error*sd_error*VectorXd::Ones(d));
            
            sampler.sample(n_iter, burn_in, -1, false);
            y_pred_vals(t-start_number) = sampler.pred_score_est;
        } else if (model_name == "GlobalCCP"){
            NormalMeanOther sampler(Y, 
                                    150, 
                                    "GlobalCCP", 
                                    Y_pred,
                                    VectorXd::Zero(d), 
                                    sd_base*sd_base*VectorXd::Ones(d),
                                    sd_error*sd_error*VectorXd::Ones(d));
            
            sampler.sample(n_iter, burn_in, -1, false);
            y_pred_vals(t-start_number) = sampler.pred_score_est;
        } else if (model_name == "NonGlobalCCP"){
            NormalMeanOther sampler(Y, 
                                    150, 
                                    "NonGlobalCCP", 
                                    Y_pred,
                                    VectorXd::Zero(d), 
                                    sd_base*sd_base*VectorXd::Ones(d),
                                    sd_error*sd_error*VectorXd::Ones(d));
            
            sampler.sample(n_iter, burn_in, -1, false);
            y_pred_vals(t-start_number) = sampler.pred_score_est;
        } else {
            std::cout << "Model name not recognised: " << model_name << std::endl;
            std::abort();
        }
        
    }

    std::string results_path = "sim_study/sim_results/normal_mean/" + model_name + "_" + std::to_string(scenario_number+1) + "_" + std::to_string(dataset_number+1) + "_" + std::to_string(start_number) + "-" + std::to_string(end_number) + ".csv";
    
    saveData(results_path, y_pred_vals);
    
    std::cout << "Results saved to path: " << results_path << std::endl;
}