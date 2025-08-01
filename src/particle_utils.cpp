#include "particle_utils.h"

VectorXd logit(const VectorXd& x){
    return (x.array()/(1-x.array())).log().matrix();
}

void default_prior_vals(const int& n,
                        const int& d,
                        double& nu,
                        VectorXd& mu,
                        MatrixXd& sigma){
    // fills nu, mu and sigma with default values as specified in Quinlan et al. (2024)
    nu = 3.0;
    mu = (1.0/(long double)n) * VectorXd::Ones(d);
    long double r0 = 0.5;
    long double sigma20 = ((long double) n - 1.0)/ (long double) (n*n*n);
    sigma = sigma20 * ((1.0-r0)*MatrixXd::Identity(d, d) + r0*MatrixXd::Ones(d, d));

    MatrixXd D_inv = (1 / (mu.array()*(1-mu.array()))).matrix().asDiagonal();
    mu = (mu.array()/(1-mu.array())).log().matrix();
    // mu = logit(mu);
    sigma = ((nu-2) / nu) * D_inv * sigma * D_inv;
}

long double rbeta(const long double& a, const long double& b, std::mt19937_64& rng){
    std::gamma_distribution<long double> dist1(a, 1);
    std::gamma_distribution<long double> dist2(b, 1);
    
    long double X = dist1(rng);
    long double Y = dist2(rng);
    long double samp = X/(X+Y);

    return samp;
}

double sum(const std::vector<double>& arr){
    double v = 0;
    for(int i=0; i<arr.size(); i++){
        v += arr[i];
    }
    return v;
}

long double sum(const std::vector<long double>& arr){
    long double v = 0;
    for(int i=0; i<arr.size(); i++){
        v += arr[i];
    }
    return v;
}

void remove_index(std::vector<int>& arr, const int& index){

    for(int i = index; i < arr.size()-1; i++){
        arr[i] = arr[i+1];
    }
    
    arr.pop_back();
}

void remove_index(std::vector<long double>& arr, const int& index){

    for(int i = index; i < arr.size()-1; i++){
        arr[i] = arr[i+1];
    }
    
    arr.pop_back();
}

int rcat(std::vector<long double> weights, std::mt19937_64& rng){
    long double u = STD_UNIF(rng);
    long double csum = weights[0];
    int c = 0;
    while(u > csum){
        c += 1;
        csum += weights[c];
        if(c >= weights.size()){
            std::cout << "overflow!! csum = " << csum << std::endl;
            std::abort();
        }
    }
    return c;
}

MatrixXi get_combs(const int& n, const int& j){
    std::vector<std::vector<int>> out = {{}};
    for(int i=0; i<n; i++){
        if(i == j){
            for(int k=0; k<out.size(); k++){
                out[k].push_back(1);
            }
        } else {
            int w = out.size();
            std::vector<int> copy;
            for(int k=0; k<w; k++){
                copy = out[k];
                copy.push_back(0);
                out.push_back(copy);
                out[k].push_back(1);
            }
        }
    }
    MatrixXi Q = MatrixXi::Zero(out.size(), n);
    for(int i=0; i<out.size(); i++){
        for(int j=0; j<n; j++){
            Q(i, j) = out[i][j];
        }
    }
    return Q;
}

void saveData(std::string fileName, MatrixXd  matrix){
    //https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
 
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();
    }
}

Eigen::MatrixXd openData(std::string fileToOpen){
    std::vector<double> matrixEntries;
    std::ifstream matrixDataFile(fileToOpen);
    std::string matrixRowString;
    std::string matrixEntry;
    
    int matrixRowNumber = 0;
    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.
 
        while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; //update the column numbers
    }
    return Map<Matrix<double, -1, -1, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

long double rgamma(const long double& a, const long double& b, std::mt19937_64& rng){
    std::gamma_distribution<long double> dist(a, 1.0/b);
    return dist(rng);
}

long double rinvgamma(const long double& a, const long double& b, std::mt19937_64& rng){
    return 1.0/rgamma(a, b, rng);
}

long double log_dgamma(const long double& x, const long double& a, const long double& b){
    return a*std::log(b) - std::lgamma(a) + (a-1)*std::log(x) - b*x;
}

long double log_dinvgamma(const long double& x, const long double& a, const long double& b){
    return a*std::log(b) - std::lgamma(a) - (a+1)*std::log(x) - b/x;
}

long double log_dinvgamma_prod(const std::vector<long double>& x, const long double& a, const long double& b){
    long double val = 0.0;
    for(int i=0; i<x.size(); i++){
        val += log_dinvgamma(x[i], a, b);
    }
    return val;
}

long double log_dnorm(const long double& x, const long double& mu, const long double& sd){
    return -0.5*std::log(2*PI*sd*sd) - 0.5*(x-mu)*(x-mu)/(sd*sd);
}