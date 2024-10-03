#' @importFrom rpart rpart

# Functions for sandwich tree splitting rules (user-specified splitting rule in rpart)
itemp_rf <- function(y, offset, parms, wt) {
  if (ncol(y) != 2) {
    stop("Matrix of response must be a 2 column matrix")
  }
  if (!missing(parms) && length(parms) > 0){
    warning("parameter argument ignored")
  }
  if (length(offset)) y <- y - offset
  sfun <- function(weigh, avar, ylevel, digits) {
    paste(" xisq=", format(signif(weigh[,1], digits)), " xiepsq=", format(signif(weigh[,2], digits)), ", AsymVar=", format(signif(avar, digits)), sep='')
  }
  environment(sfun) <- .GlobalEnv
  list(y = y, parms = NULL, numresp = 2, numy = 2, summary=sfun)
}
etemp_rf <- function(y, wt, parms) {
  sum_xisq <- sum(y[,1])
  sum_xiepsq <- sum(y[,2])
  pos_scaling <- 1e10 # set to 0 to get unweighted
  avar <- pos_scaling*length(y[,1]) - sum_xisq*sum_xisq/sum_xiepsq
  list(label = cbind(sum_xisq, sum_xiepsq), deviance = avar)
}
stemp_rf <- function(y, wt, x, parms, continuous) {
  n <- dim(y)[1]
  if (continuous) {
    # Continuous x variable
    total_temp_xisq <- sum(y[,1])
    total_temp_xiepsq <- sum(y[,2])
    left_temp_xisq <- cumsum(y[,1])[-n]
    left_temp_xiepsq <- cumsum(y[,2])[-n]
    right_temp_xisq <- total_temp_xisq - left_temp_xisq
    right_temp_xiepsq <- total_temp_xiepsq - left_temp_xiepsq
    lavar <- left_temp_xisq * left_temp_xisq / left_temp_xiepsq
    ravar <- right_temp_xisq * right_temp_xisq / right_temp_xiepsq
    goodness <- lavar + ravar - total_temp_xisq*total_temp_xisq/total_temp_xiepsq #????????
    list(goodness = goodness, direction = rep(1,length(x)-1))
  } else {
    # Categorical X variable
    ux <- sort(unique(x))
    total_temp_xisq <- tapply(y[,1], x, sum)
    total_temp_xiepsq <- tapply(y[,2], x, sum)
    wei <- total_temp_xisq/total_temp_xiepsq
    ord <- order(wei)
    n <- length(ord)

    left_temp_xisq <- cumsum(total_temp_xisq[ord])[-n]
    left_temp_xiepsq <- cumsum(total_temp_xiepsq[ord])[-n]
    right_temp_xisq <- sum(total_temp_xisq) - left_temp_xisq
    right_temp_xiepsq <- sum(total_temp_xiepsq) - left_temp_xiepsq
    lavar <- left_temp_xisq * left_temp_xisq / left_temp_xiepsq
    ravar <- right_temp_xisq * right_temp_xisq / right_temp_xiepsq
    goodness <- lavar + ravar - sum(total_temp_xisq)*sum(total_temp_xisq)/sum(total_temp_xiepsq)
    list(goodness=goodness, direction = ux[ord])
  }
}

create_rose_forest <- function() {
  ulist <- list(eval = etemp_rf, split = stemp_rf, init = itemp_rf)
  return(ulist)
}

