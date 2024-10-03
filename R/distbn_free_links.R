# Distribution-free link functions for distribution-free generalised linear models.
#' @importFrom stats dnorm qnorm dcauchy pcauchy qcauchy

create_link <- function (link = "identity") {
  okLinks <- c("identity", "log", "sqrt", "inverse", "1/mu^2", "logit", "probit", "cloglog", "cauchit")
  if (link %in% okLinks)
    G_link <- make.link.with.diff(link)
  else
    stop(paste0("Link ",link," is not available; available links are: ",paste(okLinks,collapse=", "),"."))
  return(G_link)
}

make.link.with.diff <- function (link) {
  switch(link, logit = {
    linkfun <- function(mu) log(mu/(1-mu))
    linkinv <- function(eta) {
      thresh <- -log(.Machine$double.eps/(1-.Machine$double.eps))
      eta <- pmin(pmax(eta, -thresh), thresh)
      1/(1+exp(-eta))
      }
    linkderiv <- function(mu) pmax(1/(mu*(1-mu)),100)
    muspaceproj <- function(mu) pmin(0.99,pmax(0.01,mu))
    mu.eta <- function(eta) 1/(1+exp(-eta))
    valideta <- function(eta) TRUE
  }, probit = {
    linkfun <- function(mu) qnorm(mu)
    linkinv <- function(eta) {
      thresh <- -qnorm(.Machine$double.eps)
      eta <- pmin(pmax(eta, -thresh), thresh)
      pnorm(eta)
    }
    linkderiv <- function(mu) 1/dnorm(qnorm(mu))
    muspaceproj <- function(mu) pmin(0.99,pmax(0.01,mu))
    mu.eta <- function(eta) pmax(dnorm(eta), .Machine$double.eps)
    valideta <- function(eta) TRUE
  }, cauchit = {
    linkfun <- function(mu) qcauchy(mu)
    linkinv <- function(eta) {
      thresh <- -qcauchy(.Machine$double.eps)
      eta <- pmin(pmax(eta, -thresh), thresh)
      pcauchy(eta)
    }
    linkderiv <- function(mu) 1/dcauchy(qcauchy(mu))
    muspaceproj <- function(mu) pmin(0.99,pmax(0.01,mu))
    mu.eta <- function(eta) pmax(dcauchy(eta), .Machine$double.eps)
    valideta <- function(eta) TRUE
  }, cloglog = {
    linkfun <- function(mu) log(-log(1 - mu))
    linkinv <- function(eta) pmax(pmin(-expm1(-exp(eta)),
                                       1 - .Machine$double.eps), .Machine$double.eps)
    linkderiv <- function(mu) -1/((1-mu)*log(1-mu))
    muspaceproj <- function(mu) pmin(0.99,pmax(0.01,mu))
    mu.eta <- function(eta) {
      eta <- pmin(eta, 700)
      pmax(exp(eta) * exp(-exp(eta)), .Machine$double.eps)
    }
    valideta <- function(eta) TRUE
  }, identity = {
    linkfun <- function(mu) mu
    linkinv <- function(eta) eta
    linkderiv <- function(mu) rep(1,length(mu))
    muspaceproj <- function(mu) mu
    mu.eta <- function(eta) rep.int(1, length(eta))
    valideta <- function(eta) TRUE
  }, log = {
    linkfun <- function(mu) log(mu)
    linkinv <- function(eta) pmax(exp(eta), .Machine$double.eps)
    linkderiv <- function(mu) 1/mu
    muspaceproj <- function(mu) pmax(mu,1e-4)
    mu.eta <- function(eta) pmax(exp(eta), .Machine$double.eps)
    valideta <- function(eta) TRUE
  }, sqrt = {
    linkfun <- function(mu) sqrt(mu)
    linkinv <- function(eta) eta^2
    linkderiv <- function(mu) 1/(2*sqrt(mu))
    muspaceproj <- function(mu) pmax(mu,0)
    mu.eta <- function(eta) 2 * eta
    valideta <- function(eta) all(is.finite(eta)) && all(eta>0)
  }, `1/mu^2` = {
    linkfun <- function(mu) 1/mu^2
    linkinv <- function(eta) 1/sqrt(eta)
    linkderiv <- function(mu) -2/mu^3
    muspaceproj <- function(mu) mu
    mu.eta <- function(eta) -1/(2 * eta^1.5)
    valideta <- function(eta) all(is.finite(eta)) && all(eta>0)
  }, inverse = {
    linkfun <- function(mu) 1/mu
    linkinv <- function(eta) 1/eta
    linkderiv <- function(mu) -1/mu^2
    muspaceproj <- function(mu) mu
    mu.eta <- function(eta) -1/(eta^2)
    valideta <- function(eta) all(is.finite(eta)) && all(eta!=0)
  }, stop(gettextf("%s link not recognised", sQuote(link)),
          domain = NA))
  structure(list(linkfun = linkfun, linkinv = linkinv, linkderiv = linkderiv, muspaceproj=muspaceproj, mu.eta = mu.eta,
                 valideta = valideta, name = link), class = "link-gplm")
}

