#' Summary for a rose random forest fitted object
#' @description Prints a \code{rose} object fitted by the functions \code{rose_...} in \code{rose}.
#' @param object a fitted \code{rose} object fitted by \code{rose_...}.
#' @param ... additional arguments
#' @importFrom stats printCoefmat pt pnorm
#' @export
summary.roseforest <- function(object, ...) {
  cat("\n",pres.txt.res(object)$mod.txt,", estimated using rose random forests",sep="")
  cat("\nLinear Coefficient:\n")
  object_incl_confint <- object$coefficients
  LB=formatC(signif(object_incl_confint[1]-qnorm(0.975)*object_incl_confint[2], digits=3), digits=3, format="fg", flag="#")
  UB=formatC(signif(object_incl_confint[1]+qnorm(0.975)*object_incl_confint[2], digits=3), digits=3, format="fg", flag="#")
  printCoefmat(object_incl_confint, P.values=TRUE, has.Pvalue=TRUE)
  invisible(object)
}

#' Print for a rose random forest fitted object
#' @description This is a method that prints a useful summary of aspects of a \code{rose} object fitted by the functions \code{rose_...} in \code{rose}.
#' @param x a fitted \code{rose} object fitted by \code{rose_...}.
#' @param ... additional arguments
#' @export
print.roseforest <- function(x, ...) {
  cat("\n",pres.txt.res(x)$mod.txt,", estimated using rose random forests",sep="")
  cat("\nLinear Coefficient:\n")
  LB <- x$coefficients[1]-qnorm(0.975)*x$coefficients[2]
  UB <- x$coefficients[1]+qnorm(0.975)*x$coefficients[2]
  print(cbind(x$coefficients, "CI Lower"=LB, "CI Upper"=UB), digits=5)
  invisible(x)
}

# To present text results
pres.txt.res <- function (res) {
  switch(res$model$model.type,
         plm = mod.txt <- "Partially linear model",
         gplm = mod.txt <- paste0("Generalised partially linear model (link = ",res$model$link,")"),
         pliv = mod.txt <- "Partially linear instrumental variable model",
         stop(gettextf("%s link not recognised", sQuote(res$model)),
              domain = NA))
  return(list(mod.txt = mod.txt))
}

