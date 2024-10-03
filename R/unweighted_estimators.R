#' Unweighted (baseline) estimator for the partially linear model
#'
#' Estimates the parameter of interest \eqn{\theta_0} in the partially linear regression model
#' \deqn{E[Y|X,Z] = X\theta_0 + f_0(Z),} as in \code{roseRF_plm} but without
#' any weights i.e. \eqn{J=1} and \eqn{w_1\equiv 1}.
#'
#' @param y_formula a two-sided formula object describing the regression model for \eqn{E[Y|Z]}.
#' @param y_learner a string specifying the regression method to fit the regression of \eqn{Y} on \eqn{Z} as given by \code{y_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param y_pars a list containing hyperparameters for the \code{y_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param x_formula a two-sided formula object describing the regression model for \eqn{E[X|Z]}.
#' @param x_learner a string specifying the regression method to fit the regression of \eqn{X} on \eqn{Z} as given by \code{x_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param x_pars a list containing hyperparameters for the \code{y_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param data a data frame containing the variables for the partially linear model.
#' @param K the number of folds used for \eqn{K}-fold cross-fitting. Default is 5.
#' @param S the number of repeats to mitigate the randomness in the estimator on the sample splits used for \eqn{K}-fold cross-fitting. Default is 5.
#'
#' @return A list containing:
#'   \describe{
#'     \item{\code{theta}}{The estimator of \eqn{\theta_0}.}
#'     \item{\code{stderror}}{Huber robust estimate of the standard error of the \eqn{\theta_0}-estimator.}
#'     \item{\code{coefficients}}{Table of \eqn{\theta_0} coefficient estimator, standard error, z-value and p-value.}
#'   }
#' @importFrom caret createFolds
#' @importFrom stats as.formula median
#' @export
unweighted_plm <- function(y_formula, y_learner, y_pars=list(),
                            x_formula, x_learner, x_pars=list(),
                            data, K=5, S=1) {
  res <- roseRF_plm(y_formula=y_formula, y_learner=y_learner, y_pars=y_pars,
                         x_formula=x_formula, x_learner=x_learner, x_pars=x_pars,
                         data=data, K=K, S=S,
                         min.node.size = nrow(data) )
  return(res)
}


#' Unweighted (baseline) estimator for the generalised partially linear model
#'
#' Estimates the parameter of interest \eqn{\theta_0} in the generalised partially linear regression model
#' \deqn{g(E[Y|X,Z]) = X\theta_0 + f_0(Z),} as in \code{roseRF_gplm} but without
#' any weights i.e. \eqn{J=1} and \eqn{w_1\equiv 1}.
#'
#' @param y_on_xz_formula a two-sided formula object describing the regression model for \eqn{E[Y|X,Z]} (regressing \eqn{Y} on \eqn{(X,Z)}).
#' @param y_on_xz_learner a string specifying the regression method to fit the regression as given by \code{y_on_xz_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param y_on_xz_pars a list containing hyperparameters for the \code{y_on_xz_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param Gy_on_z_formula a two-sided formula object describing the regression model for \eqn{E[g(E[Y|X,Z])|Z]} (regressing \eqn{g(\hat{E}[Y|X,Z])} on \eqn{Z}).
#' @param Gy_on_z_learner a string specifying the regression method to fit the regression as given by \code{Gy_on_z_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param Gy_on_z_pars a list containing hyperparameters for the \code{Gy_on_z_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param x_formula a two-sided formula object describing the regression model for \eqn{E[X|Z]}.
#' @param x_learner a string specifying the regression method to fit the regression of \eqn{X} on \eqn{Z} as given by \code{x_formula} (e.g. \code{randomforest, xgboost, neuralnet, gam}).
#' @param x_pars a list containing hyperparameters for the \code{x_learner} chosen. Default is an empty list, which performs hyperparameter tuning.
#' @param link link function (\eqn{g}). Options include \code{identity}, \code{log}, \code{sqrt}, \code{logit}, \code{probit}. Default is \code{identity}.
#' @param data a data frame containing the variables for the partially linear model.
#' @param K the number of folds used for \eqn{K}-fold cross-fitting. Default is 5.
#' @param S the number of repeats to mitigate the randomness in the estimator on the sample splits used for \eqn{K}-fold cross-fitting. Default is 5.
#'
#' @return A list containing:
#'   \describe{
#'     \item{\code{theta}}{The estimator of \eqn{\theta_0}.}
#'     \item{\code{stderror}}{Huber robust estimate of the standard error of the \eqn{\theta_0}-estimator.}
#'     \item{\code{coefficients}}{Table of \eqn{\theta_0} coefficient estimator, standard error, z-value and p-value.}
#'   }
#' @importFrom caret createFolds
#' @importFrom stats as.formula median
#' @export
unweighted_gplm <- function(y_on_xz_formula, y_on_xz_learner, y_on_xz_pars=list(),
                            Gy_on_z_formula, Gy_on_z_learner, Gy_on_z_pars=list(),
                            x_formula, x_learner, x_pars=list(),
                            link="identity", data, K=5, S=1) {
  res <- roseRF_gplm(y_on_xz_formula=y_on_xz_formula, y_on_xz_learner=y_on_xz_learner, y_on_xz_pars=y_on_xz_pars,
                          Gy_on_z_formula=Gy_on_z_formula, Gy_on_z_learner=Gy_on_z_learner, Gy_on_z_pars=Gy_on_z_pars,
                          x_formula=x_formula, x_learner=x_learner, x_pars=x_pars,
                          data=data, K=K, S=S,
                          link=link, min.node.size = nrow(data) )
  return(res)
}

