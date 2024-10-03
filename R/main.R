#' ROSE random forest estimator for the partially linear model
#'
#' Estimates the parameter of interest \eqn{\theta_0} in the partially linear model
#' \deqn{E[Y|X,Z] = X\theta_0 + f_0(Z),} which can be reposed in terms of
#' the `nuisance functions' \eqn{(E[Y|X], E[X|Z])} as \deqn{E[Y|X,Z]-E[Y|Z] = (X-E[X|Z])\theta_0.}
#'
#' The estimator of interest \eqn{\theta_0} solves the estimating equation
#' \deqn{\sum_{i}\psi(Y_i,X_i,Z_i; \theta,\hat{\eta}(Z),\hat{w}(Z)) = 0,}
#' \deqn{\psi(Y,X,Z;\theta,\eta_0,w) := \Big(\sum_{j=1}^J w_j(Z) \big( M_j(X,Z) - E[M_j(X,Z)|Z] \big) \Big) \Big( \big(Y-E[Y|Z]\big)-\big(X-E[X|Z]\big)\theta \Big),}
#' \deqn{\eta_0 := \big(E[Y|Z=\cdot], E[X|Z=\cdot]\big),}
#' where \eqn{M_1(X,Z),\ldots,M_J(X,Z)} denotes user-chosen functions of \eqn{(X,Z)}
#' and \eqn{w(Z)=\big(w_1(Z),\ldots,w_J(Z)\big)} denotes weights estimated via ROSE random forests.
#' The recommended default takes \eqn{J=1} and \eqn{M_1(X,Z)=X}; if taking \eqn{J\geq 2} we recommend care
#' in checking the applicability and appropriateness of any additional user-chosen
#' regression tasks.
#'
#' The parameter of interest \eqn{\theta_0} is estimated using a DML2 / \eqn{K}-fold cross-fitting
#' framework, to allow for arbitrary (\eqn{n^{1/4}}-consistent) learners for \eqn{\hat{\eta}} i.e. solving
#' the estimating equation
#' \deqn{\sum_{k}\sum_{I_k}\psi(Y_i,X_i,Z_i; \theta,\hat{\eta}^{(k)}(Z),\hat{w}^{(k)}(Z)) = 0,}
#' where \eqn{I_1,\ldots,I_K} denotes a partition of the index set for the datapoints \eqn{(Y_i,X_i,Z_i)},
#' \eqn{\hat{\eta}^{(k)}} denotes an estimator for \eqn{\eta_0} trained on the data indexed by
#' \eqn{I_k^c}, and \eqn{\hat{w}^{(k)}} denotes a ROSE random forest (again trained on the data
#' indexed by \eqn{I_k^c}).
#'
#' @param y_formula a two-sided formula object describing the model for \eqn{E[Y|Z]}.
#' @param y_learner a string specifying the regression method to fit the regression of \eqn{Y} on \eqn{Z} as given by \code{y_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param y_pars a list containing hyperparameters for the \code{y_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param x_formula a two-sided formula object describing the model for \eqn{E[X|Z]}.
#' @param x_learner a string specifying the regression method to fit the regression of \eqn{X} on \eqn{Z} as given by \code{x_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param x_pars a list containing hyperparameters for the \code{y_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param M1_formula a two-sided formula object for the model \eqn{E[M_1(X,Z)|Z]}. Default is \eqn{M_1(X,Z)=X}.
#' @param M1_learner a string specifying the regression method for \eqn{E[M_1(X,Z)|Z]} estimation.
#' @param M1_pars a list containing hyperparameters for the \code{M1_learner} chosen.
#' @param M2_formula a two-sided formula object for the model \eqn{E[M_2(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1})
#' @param M2_learner a string specifying the regression method for \eqn{E[M_2(X,Z)|Z]} estimation.
#' @param M2_pars a list containing hyperparameters for the \code{M2_learner} chosen.
#' @param M3_formula a two-sided formula object for the model \eqn{E[M_3(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1}).
#' @param M3_learner a string specifying the regression method for \eqn{E[M_3(X,Z)|Z]} estimation.
#' @param M3_pars a list containing hyperparameters for the \code{M3_learner} chosen.
#' @param M4_formula a two-sided formula object for the model \eqn{E[M_4(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1})
#' @param M4_learner a string specifying the regression method for \eqn{E[M_4(X,Z)|Z]} estimation.
#' @param M4_pars a list containing hyperparameters for the \code{M4_learner} chosen.
#' @param M5_formula a two-sided formula object for the model \eqn{E[M_5(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1})
#' @param M5_learner a string specifying the regression method for \eqn{E[M_5(X,Z)|Z]} estimation.
#' @param M5_pars a list containing hyperparameters for the \code{M5_learner} chosen.
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
roseRF_plm <- function(y_formula, y_learner, y_pars=list(),
                       x_formula, x_learner, x_pars=list(),
                       M1_formula=x_formula, M1_learner=x_learner, M1_pars=x_pars,
                       M2_formula=NA, M2_learner=NA, M2_pars=list(),
                       M3_formula=NA, M3_learner=NA, M3_pars=list(),
                       M4_formula=NA, M4_learner=NA, M4_pars=list(),
                       M5_formula=NA, M5_learner=NA, M5_pars=list(),
                       data, K=5, S=1,
                       max.depth=10, num.trees = 500, min.node.size = max(10,ceiling(0.01*(K-1)/K*nrow(data))), replace = TRUE, sample.fraction = 0.8) {

  #Bookkeeping
  Z <- unique(c(all.vars(y_formula)[-1],all.vars(x_formula)[-1]))
  rose_forest <- create_rose_forest()

#  if (!any(deparse(x_formula) %in% lapply(additional_regressions,deparse)))
#    additional_regressions <- append(additional_regressions,list(x_formula,x_learner),after=0)

  theta_hat_S <- V_hat_S <- numeric(S)
  for (s in seq_len(S)) {
    set.seed(s)
    cv_folds <- createFolds(data[,1], K)
    res_y_on_z_theta_k <- res_x_on_z_theta_k <- list()

    fold_sizes <- unlist(lapply(cv_folds,length))
    w_fres_ik <- list()
    for (k in seq_len(K)) {
      cv_fold <- cv_folds[[k]]
      data_nuisance <- data[-cv_fold,]
      data_theta <- data[cv_fold,]

      fit_y_reg <- regress_nuisance(y_formula, y_learner, data_nuisance, data_theta, y_pars)
      res_y_on_z_nuisance <- fit_y_reg$residuals_nuisance
      res_y_on_z_theta <- fit_y_reg$residuals_theta
      res_y_on_z_theta_k[[k]] <- res_y_on_z_theta

      fit_x_reg <- regress_nuisance(x_formula, x_learner, data_nuisance, data_theta, x_pars)
      res_x_on_z_nuisance <- fit_x_reg$residuals_nuisance
      res_x_on_z_theta <- fit_x_reg$residuals_theta
      res_x_on_z_theta_k[[k]] <- res_x_on_z_theta

      # Intial theta_k estimate
      theta_k <- sum(res_y_on_z_nuisance*res_x_on_z_nuisance)/sum(res_x_on_z_nuisance^2)

      # Generate ROSE random forest weights for each j=1:J
      d_theta_eps_nuisance <- - res_x_on_z_nuisance
      eps_nuisance <- res_y_on_z_nuisance - theta_k * res_x_on_z_nuisance
      eps_sq_nuisance <- eps_nuisance^2
      M_formulae = list(M1_formula, M2_formula, M3_formula, M4_formula, M5_formula)
      M_learners = list(M1_learner, M2_learner, M3_learner, M4_learner, M5_learner)
      M_parses = list(M1_pars, M2_pars, M3_pars, M4_pars, M5_pars)
      J = sum(unlist(lapply(M_formulae, function(x) deparse(x)!="NA")))
      d_theta_psi_nuisance_j <- psi_sq_nuisance_j <- psi_nuisance_j <- res_f_on_z_theta_j <- list()
      for (j in seq_len(J)) {
        # Check in jth F-regression already done (i.e. whether its x-regression)
        # Do F-regression if necessary
        f_formula <- M_formulae[[j]]
        f_learner <- M_learners[[j]]
        f_pars = M_parses[[j]]
        if (f_formula==x_formula & f_learner==x_learner ){
          res_fj_on_z_nuisance <- res_x_on_z_nuisance
          res_fj_on_z_theta <- res_x_on_z_theta
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
        forest_data[[paste0("d_theta_psi_",j)]] <-d_theta_psi_nuisance_j[[j]]
        forest_data[[paste0("psi_sq_",j)]] <- psi_sq_nuisance_j[[j]]
        forest_data[[paste0("psi_",j)]] <- psi_nuisance_j[[j]]
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
  res <- structure(list(theta=thet, stderror=se, coefficients=tab, model=list(model.type="plm")), class="roseforest")
  res$coefficients
  return(res)
}




#' ROSE random forest estimator for the generalised partially linear model
#'
#' Estimates the parameter of interest \eqn{\theta_0} in the generalised partially linear model
#' \deqn{g(E[Y|X,Z]) = X\theta_0 + f_0(Z),} for some (strictly increasing, differentiable) link function \eqn{g}, which can be reposed in terms of
#' the `nuisance functions' \eqn{(E[X|Z], E[g(E[Y|X,Z])|Z])} as \deqn{g\big(E[Y|X,Z])-E[g(E[Y|X,Z])|Z]\big) = (X-E[X|Z])\theta_0.}
#'
#' The estimator of interest \eqn{\theta_0} solves the estimating equation
#' \deqn{\sum_{i}\psi(Y_i,X_i,Z_i; \theta,\hat{\eta}(Z),\hat{w}(Z)) = 0,}
#' \deqn{\psi(Y,X,Z;\theta,\eta_0,w) := \Big(\sum_{j=1}^J w_j(Z) \big( M_j(X,Z) - E[M_j(X,Z)|Z] \big) \Big) g'\big(\mu(X,Z;\theta,\eta_0) \big) \big(Y-\mu(X,Z;\theta,\eta_0)\big) ,}
#' \deqn{\mu(X,Z;\theta,\eta_0) := g^{-1}\big(E[g(E[Y|X,Z])|Z] + (X-E[X|Z])\theta\big),}
#' \deqn{\eta_0 := \big(E[Y|Z=\cdot], E[X|Z=\cdot]\big),}
#' where \eqn{M_1(X,Z),\ldots,M_J(X,Z)} denotes user-chosen functions of \eqn{(X,Z)}
#' and \eqn{w(Z)=\big(w_1(Z),\ldots,w_J(Z)\big)} denotes weights estimated via ROSE random forests.
#' The recommended default takes \eqn{J=1} and \eqn{M_1(X,Z)=X}; if taking \eqn{J\geq 2} we recommend care
#' in checking the applicability and appropriateness of any additional user-chosen
#' regression tasks.
#'
#' The parameter of interest \eqn{\theta_0} is estimated using a DML2 / \eqn{K}-fold cross-fitting
#' framework, to allow for arbitrary (\eqn{n^{1/4}}-consistent) learners for \eqn{\hat{\eta}} i.e. solving
#' the estimating equation
#' \deqn{\sum_{k}\sum_{I_k}\psi(Y_i,X_i,Z_i; \theta,\hat{\eta}^{(k)}(Z),\hat{w}^{(k)}(Z)) = 0,}
#' where \eqn{I_1,\ldots,I_K} denotes a partition of the index set for the datapoints \eqn{(Y_i,X_i,Z_i)},
#' \eqn{\hat{\eta}^{(k)}} denotes an estimator for \eqn{\eta_0} trained on the data indexed by
#' \eqn{I_k^c}, and \eqn{\hat{w}^{(k)}} denotes a ROSE random forest (again trained on the data
#' indexed by \eqn{I_k^c}).
#'
#' @param y_on_xz_formula a two-sided formula object describing the model for \eqn{E[Y|X,Z]} (regressing \eqn{Y} on \eqn{(X,Z)}).
#' @param y_on_xz_learner a string specifying the regression method to fit the regression as given by \code{y_on_xz_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param y_on_xz_pars a list containing hyperparameters for the \code{y_on_xz_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param Gy_on_z_formula a two-sided formula object describing the model for \eqn{E[g(E[Y|X,Z])|Z]} (regressing \eqn{g(\hat{E}[Y|X,Z])} on \eqn{Z}).
#' @param Gy_on_z_learner a string specifying the regression method to fit the regression as given by \code{Gy_on_z_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param Gy_on_z_pars a list containing hyperparameters for the \code{Gy_on_z_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param x_formula a two-sided formula object describing the model for \eqn{E[X|Z]}.
#' @param x_learner a string specifying the regression method to fit the regression of \eqn{X} on \eqn{Z} as given by \code{x_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param x_pars a list containing hyperparameters for the \code{x_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param M1_formula a two-sided formula object for the model \eqn{E[M_1(X,Z)|Z]}. Default is \eqn{M_1(X,Z)=X}.
#' @param M1_learner a string specifying the regression method for \eqn{E[M_1(X,Z)|Z]} estimation.
#' @param M1_pars a list containing hyperparameters for the \code{M1_learner} chosen.
#' @param M2_formula a two-sided formula object for the model \eqn{E[M_2(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1})
#' @param M2_learner a string specifying the regression method for \eqn{E[M_2(X,Z)|Z]} estimation.
#' @param M2_pars a list containing hyperparameters for the \code{M2_learner} chosen.
#' @param M3_formula a two-sided formula object for the model \eqn{E[M_3(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1}).
#' @param M3_learner a string specifying the regression method for \eqn{E[M_3(X,Z)|Z]} estimation.
#' @param M3_pars a list containing hyperparameters for the \code{M3_learner} chosen.
#' @param M4_formula a two-sided formula object for the model \eqn{E[M_4(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1})
#' @param M4_learner a string specifying the regression method for \eqn{E[M_4(X,Z)|Z]} estimation.
#' @param M4_pars a list containing hyperparameters for the \code{M4_learner} chosen.
#' @param M5_formula a two-sided formula object for the model \eqn{E[M_5(X,Z)|Z]}. Default is no formula / regression (i.e. \eqn{J=1})
#' @param M5_learner a string specifying the regression method for \eqn{E[M_5(X,Z)|Z]} estimation.
#' @param M5_pars a list containing hyperparameters for the \code{M5_learner} chosen.
#' @param link link function (\eqn{g}). Options include \code{identity}, \code{log}, \code{sqrt}, \code{logit}, \code{probit}. Default is \code{identity}.
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
#' @importFrom stats uniroot as.formula median
#' @export
roseRF_gplm <- function(y_on_xz_formula, y_on_xz_learner, y_on_xz_pars=list(),
                        Gy_on_z_formula, Gy_on_z_learner, Gy_on_z_pars=list(),
                        x_formula, x_learner, x_pars=list(),
                        M1_formula=x_formula, M1_learner=x_learner, M1_pars=x_pars,
                        M2_formula=NA, M2_learner=NA, M2_pars=list(),
                        M3_formula=NA, M3_learner=NA, M3_pars=list(),
                        M4_formula=NA, M4_learner=NA, M4_pars=list(),
                        M5_formula=NA, M5_learner=NA, M5_pars=list(),
                        link="identity", data, K=5, S=1, max.depth=10, num.trees = 500, min.node.size = max(10,ceiling(0.01*(K-1)/K*nrow(data))), replace = TRUE, sample.fraction = 0.8) {

  G_link <- create_link(link)
  G_response <- all.vars(Gy_on_z_formula)[1]
  response <- all.vars(y_on_xz_formula)[1]
  Z <- unique(c(all.vars(Gy_on_z_formula)[-1],all.vars(x_formula)[-1]))

  theta_hat_S <- V_hat_S <- numeric(S)
  for (s in seq_len(S)) {
    set.seed(s)
    cv_folds <- createFolds(data[,1], K)
    y_theta_k <- fitted_Gy_on_z_theta_k <- res_x_on_z_theta_k <- list()

    fold_sizes <- unlist(lapply(cv_folds,length))
    w_fres_ik <- list()
    for (k in seq_len(K)) {
      cv_fold <- cv_folds[[k]]
      data_nuisance <- data[-cv_fold,]
      data_theta <- data[cv_fold,]

      fit_y_on_xz <- regress_nuisance(y_on_xz_formula, y_on_xz_learner, data_nuisance, data_theta, y_on_xz_pars)
      G_pseudo_res <- G_link$linkfun(G_link$muspaceproj(fit_y_on_xz$fitted_nuisance)) ##################
      G_pseudo_df_nuisance <- data_nuisance
      if (G_response %in% colnames(G_pseudo_df_nuisance)) G_pseudo_df_nuisance <- G_pseudo_df_nuisance[,!(colnames(G_pseudo_df_nuisance)%in%G_response)]
      G_pseudo_df_nuisance <- cbind(G_vals=G_pseudo_res, G_pseudo_df_nuisance)
      colnames(G_pseudo_df_nuisance)[1] <- G_response
      fit_Gy_reg <- regress_nuisance(Gy_on_z_formula, Gy_on_z_learner, G_pseudo_df_nuisance, data_theta, Gy_on_z_pars)
      fitted_Gy_on_z_nuisance <- fit_Gy_reg$fitted_nuisance
      fitted_Gy_on_z_theta <- fit_Gy_reg$fitted_theta
      fitted_Gy_on_z_theta_k[[k]] <- fitted_Gy_on_z_theta
      y_theta_k[[k]] <- data_theta[,response]

      fit_x_reg <- regress_nuisance(x_formula, x_learner, data_nuisance, data_theta, x_pars)
      res_x_on_z_nuisance <- fit_x_reg$residuals_nuisance
      res_x_on_z_theta <- fit_x_reg$residuals_theta
      res_x_on_z_theta_k[[k]] <- res_x_on_z_theta

      # Inital theta_hat_k estimate
      eval_theta_k <- function(theta) {
        mu <- G_link$linkinv( res_x_on_z_nuisance*theta + fitted_Gy_on_z_nuisance )
        eps <- G_link$linkderiv(mu) * ( data_nuisance[,response] - mu ) #G_link$linkderiv doesn't exist
        score <- sum( res_x_on_z_nuisance * eps )
        return(score)
      }
      if (k==1) {
        lower_bd <- -1; upper_bd <- 1
        while (eval_theta_k(lower_bd)*eval_theta_k(upper_bd)>0) {
          lower_bd <- 10*lower_bd
          upper_bd <- 10*upper_bd
        }
        lower_bd <- 10*lower_bd
        upper_bd <- 10*upper_bd
      }
      theta_k <- uniroot(eval_theta_k,interval=c(lower_bd,upper_bd))$root

      mu_nuisance <- G_link$linkinv( res_x_on_z_nuisance*theta_k + fitted_Gy_on_z_nuisance )
      d_theta_eps_nuisance <- - res_x_on_z_nuisance
      eps_nuisance <- G_link$linkderiv(mu_nuisance) * ( data_nuisance[,response] - mu_nuisance )
      eps_sq_nuisance <- eps_nuisance^2

      # Generate rose forest weights for each j=1:J
      M_formulae <- list(M1_formula, M2_formula, M3_formula, M4_formula, M5_formula)
      M_learners <- list(M1_learner, M2_learner, M3_learner, M4_learner, M5_learner)
      M_parses <- list(M1_pars, M2_pars, M3_pars, M4_pars, M5_pars)
      J = sum(unlist(lapply(M_formulae, function(x) deparse(x)!="NA")))
      d_theta_psi_nuisance_j <- psi_sq_nuisance_j <- psi_nuisance_j <- res_f_on_z_theta_j <- list()
      for (j in seq_len(J)) {
        # Check in jth F-regression already done (i.e. whether its x-regression)
        # Do F-regression if necessary
        f_formula <- M_formulae[[j]]
        f_learner <- M_learners[[j]]
        f_pars <- M_parses[[j]]
        if (f_formula==x_formula & f_learner==x_learner){
          res_fj_on_z_nuisance <- res_x_on_z_nuisance
          res_fj_on_z_theta <- res_x_on_z_theta
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
        list_overj_wfres <- Map(function(x1,x2) x1*x2, w_theta_j_fit, res_f_on_z_theta_j)
        w_fres_ik[[k]] <- sapply(seq_along(list_overj_wfres[[1]]), function(i) {
          sum(sapply(list_overj_wfres, `[`, i))
          })
      }
    }

    # Construct function to evaluate score at a theta value
    eval_theta_full <- function(theta) {
      mu <- G_link$linkinv( unlist(res_x_on_z_theta_k)*theta + unlist(fitted_Gy_on_z_theta_k) )
      eps <- G_link$linkderiv(mu) * ( unlist(y_theta_k) - mu )
      score <- sum( unlist(w_fres_ik) * eps )
      return(score)
    }
    theta_hat <- uniroot(eval_theta_full,interval=c(lower_bd,upper_bd))$root

    mu_theta <- G_link$linkinv( unlist(res_x_on_z_theta_k)*theta_hat + unlist(fitted_Gy_on_z_theta_k) )
    eps <- G_link$linkderiv(mu_theta) * ( unlist(y_theta_k) - mu_theta )
    psi <- unlist(w_fres_ik) * eps
    psi_sq <- psi^2
    d_theta_eps <- - unlist(res_x_on_z_theta_k)
    d_theta_psi <- unlist(w_fres_ik) * d_theta_eps
    V_hat <- nrow(data) * sum(psi_sq) / (sum(d_theta_psi))^2
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
  res <- structure(list(theta=thet, stderror=se, coefficients=tab, model=list(model.type="gplm", link=link)), class="roseforest")
  res$coefficients

  return(res)
}

