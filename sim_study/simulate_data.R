library(LaplacesDemon)

prob_jt <- function(j, t, x_prev, A, W, params){
  prob <- 0
  d <- length(x_prev)
  AW_sum <- sum(A[,j]*W[, j])
  for(i in 1:d){
    if(i == j){
      prob <- prob + A[i, j] * W[i, j] * params[i, j]
    } else {
      prob <- prob + A[i, j] * W[i, j] * (x_prev[i] > 0) * dgeom(t-x_prev[i]-2, params[i, j])
    }
  }
  prob/AW_sum
}

sim_cps_prior <- function(n, A, W, params){
  d <- nrow(A)
  U <- matrix(0, nrow=d, ncol=n)
  probs <- matrix(0, nrow=d, ncol=n)
  x_prev <- rep(0, d)
  
  for(t in 1:(n-1)){
    for(j in 1:d){
      pjt <- prob_jt(j, t+1, x_prev, A, W, params)
      probs[j, t+1] <- pjt
      # if(j == 1){
      if(0){
        if(t %% 50 == 0){
          U[j, t+1] <- 1
        }
      } else {
        if(runif(1) < pjt){
          U[j, t+1] <- 1
        }
      }
    }
    for(j in 1:d){
      if(U[j, t+1] == 1){
        x_prev[j] <- t
      }
    } 
  }
  U
  # list(cps = U, probs = probs)
}

sim_cps_simultaneous <- function(n, d, g1, g2, p1, p2){
  
  U <- matrix(0, nrow=d, ncol=n)
  
  cps1 <- c(0, rbinom(n-1, 1, p1))
  cps2 <- c(0, rbinom(n-1, 1, p2))
  
  for(i in g1){
    U[i, ] <- cps1
  }
  for(i in g2){
    U[i, ] <- cps2
  }
  U
}

sim_cps_random <- function(n, d, p){
  U <- matrix(0, nrow=d, ncol=n)
  
  for(i in 1:d){
    U[i, ] <- c(0, rbinom(n-1, 1, p))
  }
  U
}

sim_cps <- function(n, scenario){
  if(scenario == 1){
    A <- matrix(c(1, 1, 0, 0,
                  0, 1, 1, 0, 
                  0, 0, 1, 1,
                  0, 0, 0, 1), 4, 4, byrow=T)
    W <- matrix(5, 4, 4)
    diag(W) <- 1
    params <- matrix(0.6, 4, 4)
    diag(params) <- c(1/40) # 1/40
    return(sim_cps_prior(n, A, W, params))
  } else if(scenario == 2){
    A <- matrix(c(1, 1, 0, 0,
                  0, 1, 0, 0, 
                  0, 0, 1, 1,
                  0, 0, 0, 1), 4, 4, byrow=T)
    W <- matrix(5, 4, 4)
    diag(W) <- 1
    params <- matrix(0.6, 4, 4)
    diag(params) <- c(1/40)
    return(sim_cps_prior(n, A, W, params))
  } else if(scenario == 3){
    g1 <- c(1, 2, 3, 4)
    g2 <- c()
    p1 <- 1/40
    p2 <- 1/40
    return(sim_cps_simultaneous(n, 4, g1, g2, p1, p2))
  } else if(scenario == 4){
    g1 <- c(1, 2)
    g2 <- c(3, 4)
    p1 <- 1/40
    p2 <- 1/40
    return(sim_cps_simultaneous(n, 4, g1, g2, p1, p2))
  } else if(scenario == 5){
    return(sim_cps_random(n, 4, 1/40))
  }
}

sim_cps_normal_mean <- function(n, scenario, sd_base, sd_error){
  cps <- sim_cps(n, scenario)
  Y <- matrix(0, nrow=4, ncol=n)
  curr_means <- rnorm(4, 0, sd_base)
  Y[,1] <- rnorm(4, curr_means, sd_error)
  for(t in 2:n){
    for(j in 1:4){
      if(cps[j, t]){
        curr_means[j] <- rnorm(1, 0, sd_base)
      }
    }
    Y[,t] <- rnorm(4, curr_means, sd_error)
  }
  list(dat=Y, cps=cps)
}

sim_cps_ar_process <- function(n, scenario){
  
  L <- 1
  
  cps <- sim_cps(n, scenario)
  Y <- matrix(0, nrow=4, ncol=n+L)
  
  Y[, 1:L] <- rnorm(4*L, 0, 0.5)
  
  for(j in 1:4){
    curr_sd <- rinvgamma(1, 1, 1)
    curr_sd <- 0.5
    curr_phi <- 0.75
    sds <- c(0.3, 1, 2, 3)
    phis <- c(-0.8, 0.8, -0.8, 0.8)
    k <- length(sds)
    
    c <- 1
    for(i in (L+1):(n+L)){
      # if(i %% (50 + 2*j) == 0){
      if(cps[j, i-L]){
        # cp occured
        c <- ((c+1) %% k) 
        # c <- inds[-c][rcat(1, c(0.5, 0.5))]
        curr_phi <- phis[c+1]
        curr_sd <- sds[c+1]
        # curr_phi <- runif(1, -0.9, 0.9)
        # curr_sd <- sqrt(rinvgamma(1, 1, 1))
      }
      Y[j, i] <- Y[j, i-1] * curr_phi + rnorm(1, 0, curr_sd)
    }
  }
  list(dat=Y, cps=cps)
}

sim_scenarios_normal <- function(n, n_data_sets, sd_base, sd_error, base_path){
  for(i in 1:5){
    for(j in 1:n_data_sets){
      dat <- sim_cps_normal_mean(n, i, sd_base, sd_error)$dat
      # print(dim(dat))
      path <- paste(base_path, "/scenario_", i, "_", j, ".csv", sep="")
      write.table(dat, path, col.names=F, row.names=F, sep=",")
    }
  }
}

sim_scenarios_ar <- function(n, n_data_sets, base_path){
  for(i in 1:5){
    for(j in 1:n_data_sets){
      dat <- sim_cps_ar_process(n, i)$dat
      path <- paste(base_path, "/scenario_", i, "_", j, ".csv", sep="")
      write.table(dat, path, col.names=F, row.names=F, sep=",")
    }
  }
}

compute_co_lag <- function(U, lag){
  d <- nrow(U)
  n <- ncol(U)
  out <- matrix(0, d, d)
  for(i in 1:d){
    for(t in 1:(n-lag-1)){
      if(U[i, t]){
        out[i, ] <- out[i, ] + apply(U[, (t+1):(t+lag+1)], 1, sum)
      }
    }
  }
  s <- apply(U, 1, sum)
  sweep(out, 1, s, FUN = '/')
}

# run below to generate and save sim study datasets
set.seed(123)
sd_base <- 3
sd_error <- 0.5
sim_scenarios_normal(502, 50, sd_base, sd_error, "sim_data/normal_mean")
sim_scenarios_ar(503, 50, "sim_data/ar_process")
