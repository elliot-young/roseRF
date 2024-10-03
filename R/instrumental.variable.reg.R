#' ROSE random forest estimator for the partially linear instrumental variable model
#'
#' @param y_formula a two-sided formula object describing the regression model for \eqn{E[Y|Z]}.
#' @param y_learner a string specifying the regression method to fit the regression of \eqn{Y} on \eqn{Z} as given by \code{y_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param y_pars a list containing hyperparameters for the \code{y_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param x_formula a two-sided formula object describing the regression model for \eqn{E[X|Z]}.
#' @param x_learner a string specifying the regression method to fit the regression of \eqn{X} on \eqn{Z} as given by \code{x_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param x_pars a list containing hyperparameters for the \code{y_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param IV1_formula a two-sided formula object for the model \eqn{E[V_1|Z]}.
#' @param IV1_learner a string specifying the regression method for \eqn{E[V_1(X,Z)|Z]} estimation.
#' @param IV1_pars a list containing hyperparameters for the \code{IV1_learner} chosen.
#' @param IV2_formula a two-sided formula object for the model \eqn{E[V_2|Z]}. Default is no formula / regression (i.e. \eqn{J=1})
#' @param IV2_learner a string specifying the regression method for \eqn{E[V_2(X,Z)|Z]} estimation.
#' @param IV2_pars a list containing hyperparameters for the \code{IV2_learner} chosen.
#' @param IV3_formula a two-sided formula object for the model \eqn{E[V_3(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1}).
#' @param IV3_learner a string specifying the regression method for \eqn{E[V_3(X,Z)|Z]} estimation.
#' @param IV3_pars a list containing hyperparameters for the \code{IV3_learner} chosen.
#' @param IV4_formula a two-sided formula object for the model \eqn{E[V_4(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1})
#' @param IV4_learner a string specifying the regression method for \eqn{E[V_4(X,Z)|Z]} estimation.
#' @param IV4_pars a list containing hyperparameters for the \code{IV4_learner} chosen.
#' @param IV5_formula a two-sided formula object for the model \eqn{E[V_5(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1})
#' @param IV5_learner a string specifying the regression method for \eqn{E[V_5(X,Z)|Z]} estimation.
#' @param IV5_pars a list containing hyperparameters for the \code{IV5_learner} chosen.
#' @param data a data frame containing the variables for the partially linear model.
#' @param K the number of folds used for \eqn{K}-fold cross-fitting. Default is 5.
#' @param S the number of repeats to mitigate the randomness in the estimator on the sample splits used for \eqn{K}-fold cross-fitting. Default is 5.
#' @param max.depth Maximum depth parameter used for ROSE random forests. Default is 5.
#' @param num.trees Number of trees used for a single ROSE random forest. Default is 50.
#' @param min.node.size Minimum node size of a leaf in each tree. Default is \code{max(10,ceiling(0.01 (K-1)/K nrow(data)))}.
#' @param replace Whether sampling for a single random tree are performed with (bootstrap) or without replacement. Default is \code{TRUE} (i.e. bootstrap).
#' @param sample.fraction Proportion of data used for each random tree. Default is 0.8.
#'
#' @return A list containing:
#'   \describe{
#'     \item{\code{theta}}{The estimator of \eqn{\theta_0}.}
#'     \item{\code{stderror}}{Huber robust estimate of the standard error of the \eqn{\theta_0}-estimator.}
#'     \item{\code{coefficients}}{Table of \eqn{\theta_0} coefficient estimator, standard error, z-value and p-value.}
#'   }
#'
#' @importFrom caret createFolds
#' @importFrom stats as.formula median
#' @export
roseRF_pliv <- function(y_formula, y_learner, y_pars=list(),
                        x_formula, x_learner, x_pars=list(),
                        IV1_formula=NA, IV1_learner=NA, IV1_pars=list(),
                        IV2_formula=NA, IV2_learner=NA, IV2_pars=list(),
                        IV3_formula=NA, IV3_learner=NA, IV3_pars=list(),
                        IV4_formula=NA, IV4_learner=NA, IV4_pars=list(),
                        IV5_formula=NA, IV5_learner=NA, IV5_pars=list(),
                        data, K=5, S=1,
                        max.depth=10, num.trees = 500, min.node.size = max(10,ceiling(0.01*(K-1)/K*nrow(data))), replace = TRUE, sample.fraction = 0.8) {

  #Bookkeeping
  Z <- unique(c(all.vars(y_formula)[-1],all.vars(x_formula)[-1]))
  rose_forest <- create_rose_forest()

  theta_hat_S <- V_hat_S <- numeric(S)
  for (s in seq_len(S)) {
    set.seed(s)
    cv_folds <- createFolds(data[,1], K)
    res_y_on_z_theta_k <- res_x_on_z_theta_k <- res_iv1_on_z_theta_k <- list()

    fold_sizes <- unlist(lapply(cv_folds,length))
    w_fres_ik <- list()
    for (k in seq_len(K)) {
      cv_fold <- cv_folds[[k]]
      data_nuisance <- data[-cv_fold,]
      data_theta <- data[cv_fold,]

      fit_y_reg <- regress_nuisance(y_formula, y_learner, data_nuisance, data_theta)
      res_y_on_z_nuisance <- fit_y_reg$residuals_nuisance
      res_y_on_z_theta <- fit_y_reg$residuals_theta
      res_y_on_z_theta_k[[k]] <- res_y_on_z_theta

      fit_x_reg <- regress_nuisance(x_formula, x_learner, data_nuisance, data_theta)
      res_x_on_z_nuisance <- fit_x_reg$residuals_nuisance
      res_x_on_z_theta <- fit_x_reg$residuals_theta
      res_x_on_z_theta_k[[k]] <- res_x_on_z_theta

      fit_iv1_reg <- regress_nuisance(IV1_formula, IV1_learner, data_nuisance, data_theta)
      res_iv1_on_z_nuisance <- fit_iv1_reg$residuals_nuisance
      res_iv1_on_z_theta <- fit_iv1_reg$residuals_theta
      res_iv1_on_z_theta_k[[k]] <- res_iv1_on_z_theta

      # Intial theta_k estimate
      theta_k <- sum(res_y_on_z_nuisance*res_iv1_on_z_nuisance)/sum(res_x_on_z_nuisance*res_iv1_on_z_nuisance)

      # Regressions for additional instruments
      d_theta_eps_nuisance <- - res_x_on_z_nuisance
      eps_nuisance <- res_y_on_z_nuisance - theta_k * res_x_on_z_nuisance
      eps_sq_nuisance <- eps_nuisance^2
      M_formulae = list(IV1_formula, IV2_formula, IV3_formula, IV4_formula, IV5_formula)
      M_learners = list(IV1_learner, IV2_learner, IV3_learner, IV4_learner, IV5_learner)
      M_parses = list(IV1_pars, IV2_pars, IV3_pars, IV4_pars, IV5_pars)
      J = sum(unlist(lapply(M_formulae, function(x) deparse(x)!="NA")))
      d_theta_psi_nuisance_j <- psi_sq_nuisance_j <- psi_nuisance_j <- res_f_on_z_theta_j <- list()
      for (j in seq_len(J)) {
        # Check in jth F-regression already done (i.e. whether its x-regression)
        # Do F-regression if necessary
        f_formula = M_formulae[[j]]
        f_learner = M_learners[[j]]
        f_pars = M_parses[[j]]
        if (j==1){
          res_fj_on_z_nuisance <- res_iv1_on_z_nuisance
          res_fj_on_z_theta <- res_iv1_on_z_theta
        } else {
          fit_fj_reg <- regress_nuisance(f_formula, f_learner, data_nuisance, data_theta, f_pars)
          res_fj_on_z_nuisance <- fit_fj_reg$residuals_nuisance
          res_fj_on_z_theta <- fit_fj_reg$residuals_theta
        }
        d_theta_psi_nuisance_j[[j]] <- res_fj_on_z_nuisance * d_theta_eps_nuisance
        psi_sq_nuisance_j[[j]] <- res_fj_on_z_nuisance^2 * eps_sq_nuisance
        psi_nuisance_j[[j]] <- res_fj_on_z_nuisance * eps_nuisance
        res_f_on_z_theta_j[[j]] <- res_fj_on_z_theta
      }

      forest_data <- data_nuisance
      for (j in seq_len(J)) {
        forest_data[[paste0("d_theta_psi_",j)]]=d_theta_psi_nuisance_j[[j]]
        forest_data[[paste0("psi_sq_",j)]]=psi_sq_nuisance_j[[j]]
        forest_data[[paste0("psi_",j)]]=psi_nuisance_j[[j]]
      }
      if (J==1) {
        w_theta_j_fit  <- rose_forest_J1(data_train=forest_data, data_theta=data_theta, J=J, Z=Z, max.depth=max.depth, num.trees=num.trees, min.node.size=min.node.size, replace=replace, sample.fraction=sample.fraction)
        w_fres_ik[[k]] <- w_theta_j_fit*res_f_on_z_theta_j[[1]]
      } else {
        w_theta_j_fit  <- rose_forest_Jgtr1(data_train=forest_data, data_theta=data_theta, J=J, Z=Z, max.depth=max.depth, num.trees=num.trees, min.node.size=min.node.size, replace=replace, sample.fraction=sample.fraction)
        w_fres_ik[[k]] <- sapply(Map(function(x1,x2) x1*x2, w_theta_j_fit, res_f_on_z_theta_j),sum)
      }

    }

    # Special for linear scores (so can calculate all analytically in one step)

    d_theta_psi <- - unlist(w_fres_ik)*unlist(res_x_on_z_theta_k)
    sum_d_theta_psi <- sum(d_theta_psi)
    psi_b <- - unlist(w_fres_ik)*unlist(res_y_on_z_theta_k)
    sum_psi_b <- sum(psi_b)
    theta_hat <- sum_psi_b / sum_d_theta_psi
    psi <- - psi_b + theta_hat * d_theta_psi
    psi_sq <- psi^2
    V_hat <- nrow(data) * sum(psi_sq) / sum_d_theta_psi^2

    theta_hat_S[s] <- theta_hat
    V_hat_S[s] <- V_hat
  }

  # Aggregate over S repetitions
  se <- sqrt(median(V_hat_S/nrow(data)+(theta_hat_S-median(theta_hat_S))^2))
  thet <- median(theta_hat_S)
  zval <- tval <- thet/se
  tab <- cbind(Estimate = thet,
               StdErr = se,
               z.value = zval,
               p.value = 2*pnorm(-abs(zval)))
  rownames(tab) <- all.vars(x_formula)[1]
  colnames(tab) <- c("Estimate", "Std. Err", "z value", "Pr(>|z|)")
  res <- structure(list(theta=thet, stderror=se, coefficients=tab, model=list(model.type="pliv")), class="roseforest")
  res$coefficients

  return(res)
}

