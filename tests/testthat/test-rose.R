# Tests

# Generate example data (gplm - single covariate)
set.seed(111)
n <- 2000
z <- runif(n,-3,3)
x <- rnorm(n, cos(z), 2+tanh(z))
y <- rnorm(n, 0.1*x+sin(z), 2+tanh(z))
rdf_for_plm <- data.frame(z=z, x=x, y=y)
rdf_for_log_gplm <- data.frame(z=z, x=x, y=exp(y))
expit <- function(x) exp(x)/(1+exp(x))
rdf_for_logit_gplm <- data.frame(z=z, x=x, y=rbinom(n,1,expit(y)))

test_that("rose forests works for PLM with gam learners", {
  tst <- roseRF_plm(y_formula = y~s(z,bs="cs"), y_learner = "gam",
                         x_formula = x~s(z,bs="cs"), x_learner = "gam",
                         data=rdf_for_plm, K=2, S=2, max.depth=5, num.trees=20)
  expect_type(tst, "list")
})

test_that("rose forests works for PLM with random forest learners", {
  tst <- roseRF_plm(y_formula = y~z, y_learner = "randomforest", y_pars = list(max.depth=10, min.node.size=20, num.trees=20),
                         x_formula = x~z, x_learner = "randomforest", x_pars = list(max.depth=10, min.node.size=20, num.trees=20),
                         data=rdf_for_plm, K=2, S=2, max.depth=5, num.trees=20)
  expect_type(tst, "list")
})

test_that("unweighted PLM estimator works with gam learners", {
  tst <- unweighted_plm(y_formula = y~s(z,bs="cs"), y_learner = "gam",
                         x_formula = x~s(z,bs="cs"), x_learner = "gam",
                         data=rdf_for_plm, K=2, S=2)
  expect_type(tst, "list")
})

test_that("rose forests works for GPLM (log link) with gam learners", {
  tst <- roseRF_gplm(y_on_xz_formula = y~s(x,bs="cs")+s(z,bs="cs"), y_on_xz_learner = "gam",
                          Gy_on_z_formula = y~s(z,bs="cs"), Gy_on_z_learner = "gam",
                          x_formula = x~s(z,bs="cs"), x_learner = "gam",
                         link="log", data=rdf_for_log_gplm, K=2, S=2, max.depth=10, num.trees = 20, min.node.size=2, replace=FALSE, sample.fraction=0.1)
  expect_type(tst, "list")
})

test_that("rose forests works for GPLM (log link) with random forest learners", {
  tst <- roseRF_gplm(y_on_xz_formula = y~x+z, y_on_xz_learner = "randomforest", y_on_xz_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          Gy_on_z_formula = y~z, Gy_on_z_learner = "randomforest", Gy_on_z_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          x_formula = x~z, x_learner = "randomforest", x_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          link="log", data=rdf_for_log_gplm, K=2, S=2, max.depth=10, num.trees = 20, min.node.size=2, replace=FALSE, sample.fraction=0.1)
  expect_type(tst, "list")
})

test_that("rose forests works for GPLM with sqrt link", {
  tst <- roseRF_gplm(y_on_xz_formula = y~x+z, y_on_xz_learner = "randomforest", y_on_xz_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          Gy_on_z_formula = y~z, Gy_on_z_learner = "randomforest", Gy_on_z_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          x_formula = x~z, x_learner = "randomforest", x_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          link="sqrt", data=rdf_for_log_gplm, K=2, S=2, max.depth=10, num.trees = 20, min.node.size=2, replace=FALSE, sample.fraction=0.1)
  expect_type(tst, "list")
})

test_that("rose forests works for GPLM with logit link", {
  tst <- roseRF_gplm(y_on_xz_formula = y~x+z, y_on_xz_learner = "probabilityforest", y_on_xz_pars = list(min.node.size=20, num.trees=50),
                          Gy_on_z_formula = y~z, Gy_on_z_learner = "randomforest", Gy_on_z_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          x_formula = x~z, x_learner = "randomforest", x_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          link="logit", data=rdf_for_logit_gplm, K=2, S=2, max.depth=10, num.trees = 20, min.node.size=2, replace=FALSE, sample.fraction=0.1)
  expect_type(tst, "list")
})

test_that("rose forests works for GPLM with probit link", {
  tst <- roseRF_gplm(y_on_xz_formula = y~x+z, y_on_xz_learner = "probabilityforest", y_on_xz_pars = list(min.node.size=20, num.trees=50),
                          Gy_on_z_formula = y~z, Gy_on_z_learner = "randomforest", Gy_on_z_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          x_formula = x~z, x_learner = "randomforest", x_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          link="probit", data=rdf_for_logit_gplm, K=2, S=2, max.depth=10, num.trees = 20, min.node.size=2, replace=FALSE, sample.fraction=0.1)
  expect_type(tst, "list")
})

test_that("rose forests works for GPLM with cloglog link", {
  tst <- roseRF_gplm(y_on_xz_formula = y~x+z, y_on_xz_learner = "probabilityforest", y_on_xz_pars = list(min.node.size=20, num.trees=50),
                          Gy_on_z_formula = y~z, Gy_on_z_learner = "randomforest", Gy_on_z_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          x_formula = x~z, x_learner = "randomforest", x_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          link="cloglog", data=rdf_for_logit_gplm, K=2, S=2, max.depth=10, num.trees = 20, min.node.size=2, replace=FALSE, sample.fraction=0.1)
  expect_type(tst, "list")
})

test_that("rose forests works for GPLM with cauchit link", {
  tst <- roseRF_gplm(y_on_xz_formula = y~x+z, y_on_xz_learner = "probabilityforest", y_on_xz_pars = list(min.node.size=20, num.trees=50),
                          Gy_on_z_formula = y~z, Gy_on_z_learner = "randomforest", Gy_on_z_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          x_formula = x~z, x_learner = "randomforest", x_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                          link="cauchit", data=rdf_for_logit_gplm, K=2, S=2, max.depth=10, num.trees = 20, min.node.size=2, replace=FALSE, sample.fraction=0.1)
  expect_type(tst, "list")
})

test_that("unweighted GPLM estimator works with random forest learners", {
  tst <- unweighted_gplm(y_on_xz_formula = y~x+z, y_on_xz_learner = "randomforest", y_on_xz_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                         Gy_on_z_formula = y~z, Gy_on_z_learner = "randomforest", Gy_on_z_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                         x_formula = x~z, x_learner = "randomforest", x_pars = list(max.depth=10, min.node.size=20, num.trees=50),
                        link="log", data=rdf_for_log_gplm, K=2, S=2)
  expect_type(tst, "list")
})


# Generate example data (gplm - multiple covariates)
set.seed(111)
n <- 2000
z1 <- runif(n,-3,3)
z2 <- rnorm(n)
z3 <- rcauchy(n)
x <- rnorm(n, cos(z1), 2+tanh(z2))
y <- rnorm(n, 0.1*x+sin(z2), 2+tanh(z3))
rdf_for_plm <- data.frame(z1=z1, z2=z2, z3=z3, x=x, y=y)
test_that("rose forests works for PLM with gam learners with multiple covariates", {
  tst <- roseRF_plm(y_formula = y~s(z1,bs="cs")+s(z2,bs="cs")+s(z3,bs="cs"), y_learner = "gam",
                         x_formula = x~s(z1,bs="cs")+s(z2,bs="cs")+s(z3,bs="cs"), x_learner = "gam",
                         data=rdf_for_plm, K=5, S=5, max.depth=2, num.trees=10)
  expect_type(tst, "list")
})

test_that("rose forests works for PLM with randomforest learners with multiple covariates", {
  tst <- roseRF_plm(y_formula = y~z1+z2+z3, y_learner = "randomforest", y_pars = list(max.depth=10, min.node.size=20, num.trees=20),
                         x_formula = x~z1+z2+z3, x_learner = "randomforest", x_pars = list(max.depth=10, min.node.size=20, num.trees=20),
                         data=rdf_for_plm, K=2, S=2, max.depth=5, num.trees=20)
  expect_type(tst, "list")
})


# Generate example data (instrumental variable regression)
set.seed(111)
n <- 2000
IV1 <- rnorm(n)
IV2 <- rnorm(n)
Z <- rnorm(n)
U <- rnorm(n)
X <- tanh(IV1) + tanh(IV2) + expit(U) + rnorm(n)
Y <- X + tanh(Z) + 1*expit(U) + (0.2+1*expit(2*Z+X)) * rnorm(n)
rdf_for_iv <- data.frame(Z=Z, IV1=IV1, IV2=IV2, X=X, Y=Y)

test_that("rose forests works for PLIV with one instrument", {
  tst <- roseRF_pliv(y_formula = Y~Z, y_learner = "gam",
                          x_formula = X~Z, x_learner = "gam",
                          IV1_formula = IV1~Z, IV1_learner = "gam",
                          data=rdf_for_iv, K=2, S=2, max.depth=5, num.trees=5)
  expect_type(tst, "list")
})
test_that("rose forests works for PLIV with one instrument", {
  tst <- roseRF_pliv(y_formula = Y~Z, y_learner = "gam",
                          x_formula = X~Z, x_learner = "gam",
                          IV1_formula = IV1~Z, IV1_learner = "gam",
                          IV2_formula = IV2~Z, IV2_learner = "gam",
                          data=rdf_for_iv, K=2, S=2, max.depth=5, num.trees=5)
  expect_type(tst, "list")
})
