# Functions to carry out regressions for nuisance functions.
#' @importFrom mgcv gam
#' @importFrom glmnet cv.glmnet
#' @importFrom mlr makeRegrTask
#' @importFrom ranger ranger
#' @importFrom tuneRanger tuneRanger
#' @importFrom grf probability_forest
#' @importFrom stats lm predict sigma
#' @importFrom utils capture.output
#' @import xgboost
#' @import keras
#' @import mlr
#' @import ParamHelpers

regress_nuisance <- function (f_formula, f_learner, data_nuisance, data_theta, reg_pars=list()) {
  response <- all.vars(f_formula)[1]
  Z <- all.vars(f_formula)[-1]
  switch(f_learner, gam = {
    nuisance_f_fit <- gam(f_formula, data=data_nuisance)
    f_fitted_nuisance <- nuisance_f_fit$fitted
    f_residuals_nuisance <- data_nuisance[,response] - f_fitted_nuisance
    f_fitted_theta <- predict(nuisance_f_fit, data_theta)
    f_residuals_theta <- data_theta[,response] - f_fitted_theta
  }, randomforest = {
    stuff <- data_nuisance[, (names(data_nuisance) %in% c(response, Z))]
    f_task <- makeRegrTask(data=stuff, target=response) # used to be "response"
    if (length(reg_pars)==0) {
      res <- tuneRanger(f_task, iters = 70, iters.warmup = 30, time.budget = NULL, num.threads = NULL, num.trees = 500, parameters = list(replace = FALSE, respect.unordered.factors="order"), tune.parameters = c("min.node.size"), show.info=FALSE)
      reg_pars <- res$recommended.pars
      reg_pars$num.trees=50
    }
    nuisance_f_fit <- ranger(f_formula, data_nuisance, max.depth=reg_pars$max.depth, min.node.size=reg_pars$min.node.size, num.trees=reg_pars$num.trees)
    f_fitted_nuisance <- predict(nuisance_f_fit, data_nuisance[,Z,drop=FALSE])$predictions
    f_residuals_nuisance <- data_nuisance[,response] - f_fitted_nuisance
    f_fitted_theta <- predict(nuisance_f_fit, data_theta[,Z,drop=FALSE])$predictions
    f_residuals_theta <- data_theta[,response] - f_fitted_theta
  }, probabilityforest = {
    if (!all( data_nuisance[,response] %in% c(0,1) )) {
      stop("Error: Responses for probabilityforest must be only 0 or 1.")
    } else {
      if (length(reg_pars)==0) {
        nuisance_f_fit <- probability_forest(X=data_nuisance[,Z], Y=as.factor(data_nuisance[,response]))
      } else {
        nuisance_f_fit <- probability_forest(X=data_nuisance[,Z], Y=as.factor(data_nuisance[,response]), min.node.size=reg_pars$min.node.size, num.trees=reg_pars$num.trees)
      }
      f_fitted_nuisance <- predict(nuisance_f_fit, data_nuisance[,Z,drop=FALSE])$predictions[,2]
      f_residuals_nuisance <- data_nuisance[,response] - f_fitted_nuisance
      f_fitted_theta <- predict(nuisance_f_fit, data_theta[,Z,drop=FALSE])$predictions[,2]
      f_residuals_theta <- data_theta[,response] - f_fitted_theta
    }
  }, lm = {
    nuisance_f_fit <- lm(f_formula, data_nuisance)
    f_fitted_nuisance <- nuisance_f_fit$fitted
    f_residuals_nuisance <- data_nuisance[,response] - f_fitted_nuisance
    f_fitted_theta <- predict(nuisance_f_fit, data_theta)
    f_residuals_theta <- data_theta[,response] - f_fitted_theta
  }, lasso = {
    nuisance_f_fit <- cv.glmnet(as.matrix(data_nuisance[,Z]), as.matrix(data_nuisance[,response]))
    f_fitted_nuisance <- predict(nuisance_f_fit, newx = as.matrix(data_nuisance[,Z]), s = "lambda.min")
    f_residuals_nuisance <- data_nuisance[,response] - f_fitted_nuisance
    colnames(f_residuals_nuisance) <- NULL
    colnames(f_fitted_nuisance) <- NULL
    f_fitted_theta <- predict(nuisance_f_fit, newx = as.matrix(data_theta[,Z]), s = "lambda.min")
    f_residuals_theta <- data_theta[,response] - f_fitted_theta
    colnames(f_residuals_theta) <- NULL
    colnames(f_fitted_theta) <- NULL
  }, ridge = {
    nuisance_f_fit <- cv.glmnet(as.matrix(data_nuisance[,Z]), as.matrix(data_nuisance[,response]), alpha=0)
    f_fitted_nuisance <- predict(nuisance_f_fit, newx = as.matrix(data_nuisance[,Z]), s = "lambda.min")
    f_residuals_nuisance <- data_nuisance[,response] - f_fitted_nuisance
    colnames(f_residuals_nuisance) <- NULL
    colnames(f_fitted_nuisance) <- NULL
    f_fitted_theta <- predict(nuisance_f_fit, newx = as.matrix(data_theta[,Z]), s = "lambda.min")
    f_residuals_theta <- data_theta[,response] - f_fitted_theta
    colnames(f_residuals_theta) <- NULL
    colnames(f_fitted_theta) <- NULL
  }, xgboost = {
    if(length(reg_pars)==0) {
      data_nuisance_here <- data_nuisance[,c(Z,response)]
      data_theta_here <- data_theta[,c(Z,response)]
      task <- makeRegrTask(data=data_nuisance_here, target=response)
      lrn <- makeLearner("regr.xgboost",predict.type = "response", par.vals = list(verbose = 0))
      lrn$par.vals <- list( objective="reg:squarederror", eval_metric="rmse", nrounds=5000L, max_depth=4)
      params <- makeParamSet(
        makeNumericParam("eta",lower = 0.01,upper = 1.0)
      )
      rdesc <- makeResampleDesc("CV",iters=5L)
      ctrl <- makeTuneControlRandom(maxit = 20L)
      capture.output({
        best_model <- tuneParams(learner=lrn,task=task,resampling=rdesc,par.set=params, control=ctrl, show.info = FALSE)
      }, file = NULL)
      reg_pars <- best_model$x
      reg_pars$nrounds=5000
      reg_pars$max_depth=4
      print(reg_pars)
    }
    dattrain <- xgb.DMatrix(data = data.matrix(data_nuisance[,Z]), label=data_nuisance[,response])
    nuisance_f_fit <- xgb.train(data = dattrain, nrounds=reg_pars$nrounds, max_depth=reg_pars$max_depth, eta=reg_pars$eta)
    f_fitted_nuisance <- predict(nuisance_f_fit, dattrain)
    f_residuals_nuisance <- data_nuisance[,response] - f_fitted_nuisance
    dattheta <- xgb.DMatrix(data = data.matrix(data_theta[,Z]), label=data_theta[,response])#
    f_fitted_theta <- predict(nuisance_f_fit, dattheta)
    f_residuals_theta <- data_theta[,response] - f_fitted_theta
    #print("done a single boosting regression")
  }, neuralnet = {
    nuisance_f_fit <- keras_model_sequential(list(
      layer_dense(units = 20, activation = "relu", input_shape = length(Z)),
      layer_dense(units = 20, activation = "relu"),
      layer_dense(units = 1, activation = "linear")
    ))
    compile( nuisance_f_fit, loss = "mean_squared_error", optimizer = "sgd", metrics = "mean_squared_error")
    fit(nuisance_f_fit, as.matrix(data_nuisance[,Z]), data_nuisance[,response], epochs = 20, batch_size = 10, validation_split = 0.2)
    f_fitted_nuisance <- nuisance_f_fit %>% predict(as.matrix(data_nuisance[,Z]))
    f_residuals_nuisance <- data_nuisance[,response] - f_fitted_nuisance
    f_fitted_theta <- predict(nuisance_f_fit, as.matrix(data_theta[,Z]))
    f_residuals_theta <- data_theta[,response] - f_fitted_theta
  }, stop("Error: Invalid learner! learners must be randomforest, xgboost, neuralnet, gam, lasso, ridge, or lm."))
  return(list(residuals_nuisance=f_residuals_nuisance, fitted_nuisance=f_fitted_nuisance, residuals_theta=f_residuals_theta, fitted_theta=f_fitted_theta))
}

