# Functions to fit ROSE random forests
#' @importFrom mgcv gam
#' @import rpart

# Fit ROSE random forests for J=1
rose_forest_Jgtr1 <- function(data_train, data_theta=data_theta, J, Z=Z, max.depth, num.trees, min.node.size, replace, sample.fraction) {

  rose_forest <- create_rose_forest()

  num_for_dat <- nrow(data_train)
  num_pick <- floor(sample.fraction*num_for_dat)

  w_theta_rf_j <- list()
  for (j in 1:J) w_theta_rf_j[[j]] <- matrix(0,dim(data_theta)[1],num.trees)
  for (fsts in seq_len(num.trees)) {
    set.seed(fsts) # Reproducibility of rose forest
    # Trees
    selection <- sample(seq_len(num_for_dat), num_pick, replace=replace)
    DTdata_sub <- data_train[selection,]#bootstrap

    w_fit <- leaves <- w_fit_where <- w_fit_j_where_1ton_indexed <- list()
    num_leaves <- numeric(J)
    for (j in seq_len(J)) {
      w_fit[[j]] <- rpart(as.formula(paste0("cbind(d_theta_psi_",j,", psi_sq_",j,") ~ ",paste(Z,collapse="+"), collapse="")), data = DTdata_sub, method = rose_forest, cp=0, minbucket=min.node.size, maxdepth=max.depth)
      leaves[[j]] <- which(w_fit[[j]]$frame$var=="<leaf>")
      num_leaves[j] <- length(leaves[[j]])
      #w_fit[[j]]$frame$yval2[leaves[j],2] # sum of  xi1^2 ep^2  over leaves_1
      #w_fit[[j]]$frame$yval2[leaves[j],1] # sum of  xi1^2       over leaves_1
      w_fit_where[[j]] <- w_fit[[j]]$where
      w_fit_j_where_1ton_indexed[[j]] <- match(w_fit_where[[j]], leaves[[j]]) # THIS TRANSFORMS THE LIST OF DATA CORRESPONDING TO leaves1 nodes OT BEING INDEXED BY 1,2,3,4...
    }

    F_j_jj <- phi_j <- list()
    for (j in seq_len(J)) {
      for (jj in seq_len(J)) {
        if (j==jj) {
          diag_mat_F_j_jj <- w_fit[[j]]$frame$yval2[leaves[[j]],2]
          if (length(diag_mat_F_j_jj)==1) {
            F_j_jj[[paste0(j,",",j)]] <- diag_mat_F_j_jj
          } else {
            F_j_jj[[paste0(j,",",j)]] <- diag(diag_mat_F_j_jj)
          }
        } else if (j<jj) {
          xij_xijj_epsq <- DTdata_sub[[paste0("psi_",j)]] * DTdata_sub[[paste0("psi_",jj)]]
          F_cross_j_jj <- matrix(0,num_leaves[j],num_leaves[jj])
          for (i in seq_len(num_pick)) {
            F_cross_j_jj[w_fit_j_where_1ton_indexed[[j]][i], w_fit_j_where_1ton_indexed[[jj]][i]] <-
              F_cross_j_jj[w_fit_j_where_1ton_indexed[[j]][i], w_fit_j_where_1ton_indexed[[jj]][i]] + xij_xijj_epsq[i]
          }
          F_j_jj[[paste0(j,",",jj)]] <- F_cross_j_jj
          F_j_jj[[paste0(jj,",",j)]] <- t(F_cross_j_jj)
        }
      }
      # Generate phi vector
      phi_j[[j]] <- w_fit[[j]]$frame$yval2[leaves[[j]],1]
    }

    F_block_rows <- list()
    for (j in seq_len(J)) {
      F_matrices_j <- F_j_jj[grep(paste0("^",j,","), names(F_j_jj))]
      combined_row_j <- do.call(cbind, F_matrices_j)
      F_block_rows[[j]] <- combined_row_j
    }
    Fall <- do.call(rbind, F_block_rows)

    phiall <- unlist(phi_j)

    opweightsall <- solve(Fall, phiall)

    opweights <- list()
    tot_num_leaves <- 0
    for (j in seq_len(J)) {
      opweights[[j]] <- opweightsall[(tot_num_leaves+1):(tot_num_leaves+num_leaves[j])]
      tot_num_leaves <- tot_num_leaves + num_leaves[j]
    }
    # Overwrite original decision trees with rose evalutations
    fresh_w_fit <- w_fit_theta <- list()
    for (j in seq_len(J)) {
      fresh_w_fit[[j]] <- w_fit[[j]]
      fresh_w_fit[[j]]$frame$yval[leaves[[j]]] <- opweights[[j]]
      w_fit_theta[[j]] <- predict(fresh_w_fit[[j]], newdata=data_theta, type="vector")
      w_theta_rf_j[[j]][,fsts] <- w_fit_theta[[j]]
    }

  }

  w_theta <- list()
  for (j in seq_len(J)) {
    w_theta[[j]] <- rowMeans(w_theta_rf_j[[j]])
  }

  return(w_theta)
}

# Fit ROSE random forests plus for J>1
rose_forest_J1 <- function(data_train, data_theta=data_theta, J, Z=Z, max.depth, num.trees, min.node.size, replace, sample.fraction) {

  rose_forest <- create_rose_forest()

  num_for_dat <- nrow(data_train)
  num_pick <- floor(sample.fraction*num_for_dat)

  w_theta_rf_num <- w_theta_rf_den <- matrix(0,nrow(data_theta),num.trees)
  for (b in seq_len(num.trees)) {
    DTdata_sub <- data_train[sample(seq_len(num_for_dat), num_pick, replace=replace),]#bootstrap
    w_fit <- rpart(as.formula(paste("cbind(d_theta_psi_1, psi_sq_1) ~ ",paste(Z,collapse="+"), collapse="")), data = DTdata_sub, method = rose_forest, cp=0, minbucket=min.node.size, maxdepth=max.depth)
    w_theta_fit <- predict(w_fit, data_theta, type="matrix")
    w_theta_rf_num[,b] <- w_theta_fit[,1]
    w_theta_rf_den[,b] <- w_theta_fit[,2]
  }
  w_theta <- rowMeans(w_theta_rf_num)/rowMeans(w_theta_rf_den)

  return(w_theta)
}

