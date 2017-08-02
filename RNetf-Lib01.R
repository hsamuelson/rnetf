#
# Henry Samuelson 1/17/17
#

# R Library for muilti-subclassificantion machine learning




#### MAIN NN ALG ####
neuralnet <-
  function (formula, data, hidden = 1, threshold = 0.01, stepmax = 1e+05, 
            rep = 1, startweights = NULL, learningrate.limit = NULL, 
            learningrate.factor = list(minus = 0.5, plus = 1.2), learningrate = NULL, 
            lifesign = "none", lifesign.step = 1000, algorithm = "rprop+", 
            err.fct = "sse", act.fct = "logistic", linear.output = TRUE, 
            exclude = NULL, constant.weights = NULL, likelihood = FALSE) 
  {
    call <- match.call()
    options(scipen = 100, digits = 10)
    result <- varify.variables(data, formula, startweights, learningrate.limit, 
                               learningrate.factor, learningrate, lifesign, algorithm, 
                               threshold, lifesign.step, hidden, rep, stepmax, err.fct, 
                               act.fct)
    data <- result$data
    formula <- result$formula
    startweights <- result$startweights
    learningrate.limit <- result$learningrate.limit
    learningrate.factor <- result$learningrate.factor
    learningrate.bp <- result$learningrate.bp
    lifesign <- result$lifesign
    algorithm <- result$algorithm
    threshold <- result$threshold
    lifesign.step <- result$lifesign.step
    hidden <- result$hidden
    rep <- result$rep
    stepmax <- result$stepmax
    model.list <- result$model.list
    matrix <- NULL
    list.result <- NULL
    result <- generate.initial.variables(data, model.list, hidden, 
                                         act.fct, err.fct, algorithm, linear.output, formula)
    covariate <- result$covariate
    response <- result$response
    err.fct <- result$err.fct
    err.deriv.fct <- result$err.deriv.fct
    act.fct <- result$act.fct
    act.deriv.fct <- result$act.deriv.fct
    for (i in 1:rep) {
      if (lifesign != "none") {
        lifesign <- display(hidden, threshold, rep, i, lifesign)
      }
      utils::flush.console()
      result <- calculate.neuralnet(learningrate.limit = learningrate.limit, 
                                    learningrate.factor = learningrate.factor, covariate = covariate, 
                                    response = response, data = data, model.list = model.list, 
                                    threshold = threshold, lifesign.step = lifesign.step, 
                                    stepmax = stepmax, hidden = hidden, lifesign = lifesign, 
                                    startweights = startweights, algorithm = algorithm, 
                                    err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                                    act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                                    rep = i, linear.output = linear.output, exclude = exclude, 
                                    constant.weights = constant.weights, likelihood = likelihood, 
                                    learningrate.bp = learningrate.bp)
      if (!is.null(result$output.vector)) {
        list.result <- c(list.result, list(result))
        matrix <- cbind(matrix, result$output.vector)
      }
    }
    utils::flush.console()
    if (!is.null(matrix)) {
      weight.count <- length(unlist(list.result[[1]]$weights)) - 
        length(exclude) + length(constant.weights) - sum(constant.weights == 
                                                           0)
      if (!is.null(startweights) && length(startweights) < 
          (rep * weight.count)) {
        warning("some weights were randomly generated, because 'startweights' did not contain enough values", 
                call. = F)
      }
      ncol.matrix <- ncol(matrix)
    }
    else ncol.matrix <- 0
    if (ncol.matrix < rep) 
      warning(sprintf("algorithm did not converge in %s of %s repetition(s) within the stepmax", 
                      (rep - ncol.matrix), rep), call. = FALSE)
    nn <- generate.output(covariate, call, rep, threshold, matrix, 
                          startweights, model.list, response, err.fct, act.fct, 
                          data, list.result, linear.output, exclude)
    return(nn)
  }

varify.variables <-
  function (data, formula, startweights, learningrate.limit, learningrate.factor, 
            learningrate.bp, lifesign, algorithm, threshold, lifesign.step, 
            hidden, rep, stepmax, err.fct, act.fct) 
  {
    if (is.null(data)) 
      stop("'data' is missing", call. = FALSE)
    if (is.null(formula)) 
      stop("'formula' is missing", call. = FALSE)
    if (!is.null(startweights)) {
      startweights <- as.vector(unlist(startweights))
      if (any(is.na(startweights))) 
        startweights <- startweights[!is.na(startweights)]
    }
    data <- as.data.frame(data)
    formula <- stats::as.formula(formula)
    model.vars <- attr(stats::terms(formula), "term.labels")
    formula.reverse <- formula
    formula.reverse[[3]] <- formula[[2]]
    model.resp <- attr(stats::terms(formula.reverse), "term.labels")
    model.list <- list(response = model.resp, variables = model.vars)
    if (!is.null(learningrate.limit)) {
      if (length(learningrate.limit) != 2) 
        stop("'learningrate.factor' must consist of two components", 
             call. = FALSE)
      learningrate.limit <- as.list(learningrate.limit)
      names(learningrate.limit) <- c("min", "max")
      learningrate.limit$min <- as.vector(as.numeric(learningrate.limit$min))
      learningrate.limit$max <- as.vector(as.numeric(learningrate.limit$max))
      if (is.na(learningrate.limit$min) || is.na(learningrate.limit$max)) 
        stop("'learningrate.limit' must be a numeric vector", 
             call. = FALSE)
    }
    if (!is.null(learningrate.factor)) {
      if (length(learningrate.factor) != 2) 
        stop("'learningrate.factor' must consist of two components", 
             call. = FALSE)
      learningrate.factor <- as.list(learningrate.factor)
      names(learningrate.factor) <- c("minus", "plus")
      learningrate.factor$minus <- as.vector(as.numeric(learningrate.factor$minus))
      learningrate.factor$plus <- as.vector(as.numeric(learningrate.factor$plus))
      if (is.na(learningrate.factor$minus) || is.na(learningrate.factor$plus)) 
        stop("'learningrate.factor' must be a numeric vector", 
             call. = FALSE)
    }
    else learningrate.factor <- list(minus = c(0.5), plus = c(1.2))
    if (is.null(lifesign)) 
      lifesign <- "none"
    lifesign <- as.character(lifesign)
    if (!((lifesign == "none") || (lifesign == "minimal") || 
          (lifesign == "full"))) 
      lifesign <- "minimal"
    if (is.na(lifesign)) 
      stop("'lifesign' must be a character", call. = FALSE)
    if (is.null(algorithm)) 
      algorithm <- "rprop+"
    algorithm <- as.character(algorithm)
    if (!((algorithm == "rprop+") || (algorithm == "rprop-") || 
          (algorithm == "slr") || (algorithm == "sag") || (algorithm == 
                                                           "backprop"))) 
      stop("'algorithm' is not known", call. = FALSE)
    if (is.null(threshold)) 
      threshold <- 0.01
    threshold <- as.numeric(threshold)
    if (is.na(threshold)) 
      stop("'threshold' must be a numeric value", call. = FALSE)
    if (algorithm == "backprop") 
      if (is.null(learningrate.bp) || !is.numeric(learningrate.bp)) 
        stop("'learningrate' must be a numeric value, if the backpropagation algorithm is used", 
             call. = FALSE)
    if (is.null(lifesign.step)) 
      lifesign.step <- 1000
    lifesign.step <- as.integer(lifesign.step)
    if (is.na(lifesign.step)) 
      stop("'lifesign.step' must be an integer", call. = FALSE)
    if (lifesign.step < 1) 
      lifesign.step <- as.integer(100)
    if (is.null(hidden)) 
      hidden <- 0
    hidden <- as.vector(as.integer(hidden))
    if (prod(!is.na(hidden)) == 0) 
      stop("'hidden' must be an integer vector or a single integer", 
           call. = FALSE)
    if (length(hidden) > 1 && prod(hidden) == 0) 
      stop("'hidden' contains at least one 0", call. = FALSE)
    if (is.null(rep)) 
      rep <- 1
    rep <- as.integer(rep)
    if (is.na(rep)) 
      stop("'rep' must be an integer", call. = FALSE)
    if (is.null(stepmax)) 
      stepmax <- 10000
    stepmax <- as.integer(stepmax)
    if (is.na(stepmax)) 
      stop("'stepmax' must be an integer", call. = FALSE)
    if (stepmax < 1) 
      stepmax <- as.integer(1000)
    if (is.null(hidden)) {
      if (is.null(learningrate.limit)) 
        learningrate.limit <- list(min = c(1e-08), max = c(50))
    }
    else {
      if (is.null(learningrate.limit)) 
        learningrate.limit <- list(min = c(1e-10), max = c(0.1))
    }
    if (!is.function(act.fct) && act.fct != "logistic" && act.fct != 
        "tanh") 
      stop("''act.fct' is not known", call. = FALSE)
    if (!is.function(err.fct) && err.fct != "sse" && err.fct != 
        "ce") 
      stop("'err.fct' is not known", call. = FALSE)
    return(list(data = data, formula = formula, startweights = startweights, 
                learningrate.limit = learningrate.limit, learningrate.factor = learningrate.factor, 
                learningrate.bp = learningrate.bp, lifesign = lifesign, 
                algorithm = algorithm, threshold = threshold, lifesign.step = lifesign.step, 
                hidden = hidden, rep = rep, stepmax = stepmax, model.list = model.list))
  }

generate.initial.variables <-
  function (data, model.list, hidden, act.fct, err.fct, algorithm, 
            linear.output, formula) 
  {
    formula.reverse <- formula
    formula.reverse[[2]] <- stats::as.formula(paste(model.list$response[[1]], 
                                                    "~", model.list$variables[[1]], sep = ""))[[2]]
    formula.reverse[[3]] <- formula[[2]]
    response <- as.matrix(stats::model.frame(formula.reverse, data))
    formula.reverse[[3]] <- formula[[3]]
    covariate <- as.matrix(stats::model.frame(formula.reverse, data))
    covariate[, 1] <- 1
    colnames(covariate)[1] <- "intercept"
    if (is.function(act.fct)) {
      act.deriv.fct <- differentiate(act.fct)
      attr(act.fct, "type") <- "function"
    }
    else {
      if (act.fct == "tanh") {
        act.fct <- function(x) {
          tanh(x)
        }
        attr(act.fct, "type") <- "tanh"
        act.deriv.fct <- function(x) {
          1 - x^2
        }
      }
      else if (act.fct == "logistic") {
        act.fct <- function(x) {
          1/(1 + exp(-x))
        }
        attr(act.fct, "type") <- "logistic"
        act.deriv.fct <- function(x) {
          x * (1 - x)
        }
      }
    }
    if (is.function(err.fct)) {
      err.deriv.fct <- differentiate(err.fct)
      attr(err.fct, "type") <- "function"
    }
    else {
      if (err.fct == "ce") {
        if (all(response == 0 | response == 1)) {
          err.fct <- function(x, y) {
            -(y * log(x) + (1 - y) * log(1 - x))
          }
          attr(err.fct, "type") <- "ce"
          err.deriv.fct <- function(x, y) {
            (1 - y)/(1 - x) - y/x
          }
        }
        else {
          err.fct <- function(x, y) {
            1/2 * (y - x)^2
          }
          attr(err.fct, "type") <- "sse"
          err.deriv.fct <- function(x, y) {
            x - y
          }
          warning("'err.fct' was automatically set to sum of squared error (sse), because the response is not binary", 
                  call. = F)
        }
      }
      else if (err.fct == "sse") {
        err.fct <- function(x, y) {
          1/2 * (y - x)^2
        }
        attr(err.fct, "type") <- "sse"
        err.deriv.fct <- function(x, y) {
          x - y
        }
      }
    }
    return(list(covariate = covariate, response = response, err.fct = err.fct, 
                err.deriv.fct = err.deriv.fct, act.fct = act.fct, act.deriv.fct = act.deriv.fct))
  }
differentiate <-
  function (orig.fct, hessian = FALSE) 
  {
    body.fct <- deparse(body(orig.fct))
    if (body.fct[1] == "{") 
      body.fct <- body.fct[2]
    text <- paste("y~", body.fct, sep = "")
    text2 <- paste(deparse(orig.fct)[1], "{}")
    temp <- stats::deriv(eval(parse(text = text)), "x", func = eval(parse(text = text2)), 
                         hessian = hessian)
    temp <- deparse(temp)
    derivative <- NULL
    if (!hessian) 
      for (i in 1:length(temp)) {
        if (!any(grep("value", temp[i]))) 
          derivative <- c(derivative, temp[i])
      }
    else for (i in 1:length(temp)) {
      if (!any(grep("value", temp[i]), grep("grad", temp[i]), 
               grep(", c", temp[i]))) 
        derivative <- c(derivative, temp[i])
    }
    number <- NULL
    for (i in 1:length(derivative)) {
      if (any(grep("<-", derivative[i]))) 
        number <- i
    }
    if (is.null(number)) {
      return(function(x) {
        matrix(0, nrow(x), ncol(x))
      })
    }
    else {
      derivative[number] <- unlist(strsplit(derivative[number], 
                                            "<-"))[2]
      derivative <- eval(parse(text = derivative))
    }
    if (length(formals(derivative)) == 1 && length(derivative(c(1, 
                                                                1))) == 1) 
      derivative <- eval(parse(text = paste("function(x){matrix(", 
                                            derivative(1), ", nrow(x), ncol(x))}")))
    if (length(formals(derivative)) == 2 && length(derivative(c(1, 
                                                                1), c(1, 1))) == 1) 
      derivative <- eval(parse(text = paste("function(x, y){matrix(", 
                                            derivative(1, 1), ", nrow(x), ncol(x))}")))
    return(derivative)
  }
display <-
  function (hidden, threshold, rep, i.rep, lifesign) 
  {
    text <- paste("    rep: %", nchar(rep) - nchar(i.rep), "s", 
                  sep = "")
    cat("hidden: ", paste(hidden, collapse = ", "), "    thresh: ", 
        threshold, sprintf(eval(expression(text)), ""), i.rep, 
        "/", rep, "    steps: ", sep = "")
    if (lifesign == "full") 
      lifesign <- sum(nchar(hidden)) + 2 * length(hidden) - 
      2 + max(nchar(threshold)) + 2 * nchar(rep) + 41
    return(lifesign)
  }
calculate.neuralnet <-
  function (data, model.list, hidden, stepmax, rep, threshold, 
            learningrate.limit, learningrate.factor, lifesign, covariate, 
            response, lifesign.step, startweights, algorithm, act.fct, 
            act.deriv.fct, err.fct, err.deriv.fct, linear.output, likelihood, 
            exclude, constant.weights, learningrate.bp) 
  {
    time.start.local <- Sys.time()
    result <- generate.startweights(model.list, hidden, startweights, 
                                    rep, exclude, constant.weights)
    weights <- result$weights
    exclude <- result$exclude
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    result <- rprop(weights = weights, threshold = threshold, 
                    response = response, covariate = covariate, learningrate.limit = learningrate.limit, 
                    learningrate.factor = learningrate.factor, stepmax = stepmax, 
                    lifesign = lifesign, lifesign.step = lifesign.step, act.fct = act.fct, 
                    act.deriv.fct = act.deriv.fct, err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                    algorithm = algorithm, linear.output = linear.output, 
                    exclude = exclude, learningrate.bp = learningrate.bp)
    startweights <- weights
    weights <- result$weights
    step <- result$step
    reached.threshold <- result$reached.threshold
    net.result <- result$net.result
    error <- sum(err.fct(net.result, response))
    if (is.na(error) & type(err.fct) == "ce") 
      if (all(net.result <= 1, net.result >= 0)) 
        error <- sum(err.fct(net.result, response), na.rm = T)
    if (!is.null(constant.weights) && any(constant.weights != 
                                          0)) 
      exclude <- exclude[-which(constant.weights != 0)]
    if (length(exclude) == 0) 
      exclude <- NULL
    aic <- NULL
    bic <- NULL
    if (likelihood) {
      synapse.count <- length(unlist(weights)) - length(exclude)
      aic <- 2 * error + (2 * synapse.count)
      bic <- 2 * error + log(nrow(response)) * synapse.count
    }
    if (is.na(error)) 
      warning("'err.fct' does not fit 'data' or 'act.fct'", 
              call. = F)
    if (lifesign != "none") {
      if (reached.threshold <= threshold) {
        cat(rep(" ", (max(nchar(stepmax), nchar("stepmax")) - 
                        nchar(step))), step, sep = "")
        cat("\terror: ", round(error, 5), rep(" ", 6 - (nchar(round(error, 
                                                                    5)) - nchar(round(error, 0)))), sep = "")
        if (!is.null(aic)) {
          cat("\taic: ", round(aic, 5), rep(" ", 6 - (nchar(round(aic, 
                                                                  5)) - nchar(round(aic, 0)))), sep = "")
        }
        if (!is.null(bic)) {
          cat("\tbic: ", round(bic, 5), rep(" ", 6 - (nchar(round(bic, 
                                                                  5)) - nchar(round(bic, 0)))), sep = "")
        }
        time <- difftime(Sys.time(), time.start.local)
        cat("\ttime: ", round(time, 2), " ", attr(time, "units"), 
            sep = "")
        cat("\n")
      }
    }
    if (reached.threshold > threshold) 
      return(result = list(output.vector = NULL, weights = NULL))
    output.vector <- c(error = error, reached.threshold = reached.threshold, 
                       steps = step)
    if (!is.null(aic)) {
      output.vector <- c(output.vector, aic = aic)
    }
    if (!is.null(bic)) {
      output.vector <- c(output.vector, bic = bic)
    }
    for (w in 1:length(weights)) output.vector <- c(output.vector, 
                                                    as.vector(weights[[w]]))
    generalized.weights <- calculate.generalized.weights(weights, 
                                                         neuron.deriv = result$neuron.deriv, net.result = net.result)
    startweights <- unlist(startweights)
    weights <- unlist(weights)
    if (!is.null(exclude)) {
      startweights[exclude] <- NA
      weights[exclude] <- NA
    }
    startweights <- relist(startweights, nrow.weights, ncol.weights)
    weights <- relist(weights, nrow.weights, ncol.weights)
    return(list(generalized.weights = generalized.weights, weights = weights, 
                startweights = startweights, net.result = result$net.result, 
                output.vector = output.vector))
  }
generate.startweights <-
  function (model.list, hidden, startweights, rep, exclude, constant.weights) 
  {
    input.count <- length(model.list$variables)
    output.count <- length(model.list$response)
    if (!(length(hidden) == 1 && hidden == 0)) {
      length.weights <- length(hidden) + 1
      nrow.weights <- array(0, dim = c(length.weights))
      ncol.weights <- array(0, dim = c(length.weights))
      nrow.weights[1] <- (input.count + 1)
      ncol.weights[1] <- hidden[1]
      if (length(hidden) > 1) 
        for (i in 2:length(hidden)) {
          nrow.weights[i] <- hidden[i - 1] + 1
          ncol.weights[i] <- hidden[i]
        }
      nrow.weights[length.weights] <- hidden[length.weights - 
                                               1] + 1
      ncol.weights[length.weights] <- output.count
    }
    else {
      length.weights <- 1
      nrow.weights <- array((input.count + 1), dim = c(1))
      ncol.weights <- array(output.count, dim = c(1))
    }
    length <- sum(ncol.weights * nrow.weights)
    vector <- rep(0, length)
    if (!is.null(exclude)) {
      if (is.matrix(exclude)) {
        exclude <- matrix(as.integer(exclude), ncol = ncol(exclude), 
                          nrow = nrow(exclude))
        if (nrow(exclude) >= length || ncol(exclude) != 3) 
          stop("'exclude' has wrong dimensions", call. = FALSE)
        if (any(exclude < 1)) 
          stop("'exclude' contains at least one invalid weight", 
               call. = FALSE)
        temp <- relist(vector, nrow.weights, ncol.weights)
        for (i in 1:nrow(exclude)) {
          if (exclude[i, 1] > length.weights || exclude[i, 
                                                        2] > nrow.weights[exclude[i, 1]] || exclude[i, 
                                                                                                    3] > ncol.weights[exclude[i, 1]]) 
            stop("'exclude' contains at least one invalid weight", 
                 call. = FALSE)
          temp[[exclude[i, 1]]][exclude[i, 2], exclude[i, 
                                                       3]] <- 1
        }
        exclude <- which(unlist(temp) == 1)
      }
      else if (is.vector(exclude)) {
        exclude <- as.integer(exclude)
        if (max(exclude) > length || min(exclude) < 1) {
          stop("'exclude' contains at least one invalid weight", 
               call. = FALSE)
        }
      }
      else {
        stop("'exclude' must be a vector or matrix", call. = FALSE)
      }
      if (length(exclude) >= length) 
        stop("all weights are exluded", call. = FALSE)
    }
    length <- length - length(exclude)
    if (!is.null(exclude)) {
      if (is.null(startweights) || length(startweights) < (length * 
                                                           rep)) 
        vector[-exclude] <- stats::rnorm(length)
      else vector[-exclude] <- startweights[((rep - 1) * length + 
                                               1):(length * rep)]
    }
    else {
      if (is.null(startweights) || length(startweights) < (length * 
                                                           rep)) 
        vector <- stats::rnorm(length)
      else vector <- startweights[((rep - 1) * length + 1):(length * 
                                                              rep)]
    }
    if (!is.null(exclude) && !is.null(constant.weights)) {
      if (length(exclude) < length(constant.weights)) 
        stop("constant.weights contains more weights than exclude", 
             call. = FALSE)
      else vector[exclude[1:length(constant.weights)]] <- constant.weights
    }
    weights <- relist(vector, nrow.weights, ncol.weights)
    return(list(weights = weights, exclude = exclude))
  }
rprop <-
  function (weights, response, covariate, threshold, learningrate.limit, 
            learningrate.factor, stepmax, lifesign, lifesign.step, act.fct, 
            act.deriv.fct, err.fct, err.deriv.fct, algorithm, linear.output, 
            exclude, learningrate.bp) 
  {
    step <- 1
    nchar.stepmax <- max(nchar(stepmax), 7)
    length.weights <- length(weights)
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    length.unlist <- length(unlist(weights)) - length(exclude)
    learningrate <- as.vector(matrix(0.1, nrow = 1, ncol = length.unlist))
    gradients.old <- as.vector(matrix(0, nrow = 1, ncol = length.unlist))
    if (is.null(exclude)) 
      exclude <- length(unlist(weights)) + 1
    if (type(act.fct) == "tanh" || type(act.fct) == "logistic") 
      special <- TRUE
    else special <- FALSE
    if (linear.output) {
      output.act.fct <- function(x) {
        x
      }
      output.act.deriv.fct <- function(x) {
        matrix(1, nrow(x), ncol(x))
      }
    }
    else {
      if (type(err.fct) == "ce" && type(act.fct) == "logistic") {
        err.deriv.fct <- function(x, y) {
          x * (1 - y) - y * (1 - x)
        }
        linear.output <- TRUE
      }
      output.act.fct <- act.fct
      output.act.deriv.fct <- act.deriv.fct
    }
    result <- compute.net(weights, length.weights, covariate = covariate, 
                          act.fct = act.fct, act.deriv.fct = act.deriv.fct, output.act.fct = output.act.fct, 
                          output.act.deriv.fct = output.act.deriv.fct, special)
    err.deriv <- err.deriv.fct(result$net.result, response)
    gradients <- calculate.gradients(weights = weights, length.weights = length.weights, 
                                     neurons = result$neurons, neuron.deriv = result$neuron.deriv, 
                                     err.deriv = err.deriv, exclude = exclude, linear.output = linear.output)
    reached.threshold <- max(abs(gradients))
    min.reached.threshold <- reached.threshold
    while (step < stepmax && reached.threshold > threshold) {
      if (!is.character(lifesign) && step%%lifesign.step == 
          0) {
        text <- paste("%", nchar.stepmax, "s", sep = "")
        cat(sprintf(eval(expression(text)), step), "\tmin thresh: ", 
            min.reached.threshold, "\n", rep(" ", lifesign), 
            sep = "")
        utils::flush.console()
      }
      if (algorithm == "rprop+") 
        result <- plus(gradients, gradients.old, weights, 
                       nrow.weights, ncol.weights, learningrate, learningrate.factor, 
                       learningrate.limit, exclude)
      else if (algorithm == "backprop") 
        result <- backprop(gradients, weights, length.weights, 
                           nrow.weights, ncol.weights, learningrate.bp, 
                           exclude)
      else result <- minus(gradients, gradients.old, weights, 
                           length.weights, nrow.weights, ncol.weights, learningrate, 
                           learningrate.factor, learningrate.limit, algorithm, 
                           exclude)
      gradients.old <- result$gradients.old
      weights <- result$weights
      learningrate <- result$learningrate
      result <- compute.net(weights, length.weights, covariate = covariate, 
                            act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                            output.act.fct = output.act.fct, output.act.deriv.fct = output.act.deriv.fct, 
                            special)
      err.deriv <- err.deriv.fct(result$net.result, response)
      gradients <- calculate.gradients(weights = weights, length.weights = length.weights, 
                                       neurons = result$neurons, neuron.deriv = result$neuron.deriv, 
                                       err.deriv = err.deriv, exclude = exclude, linear.output = linear.output)
      reached.threshold <- max(abs(gradients))
      if (reached.threshold < min.reached.threshold) {
        min.reached.threshold <- reached.threshold
      }
      step <- step + 1
    }
    if (lifesign != "none" && reached.threshold > threshold) {
      cat("stepmax\tmin thresh: ", min.reached.threshold, "\n", 
          sep = "")
    }
    return(list(weights = weights, step = as.integer(step), reached.threshold = reached.threshold, 
                net.result = result$net.result, neuron.deriv = result$neuron.deriv))
  }
compute.net <-
  function (weights, length.weights, covariate, act.fct, act.deriv.fct, 
            output.act.fct, output.act.deriv.fct, special) 
  {
    neuron.deriv <- NULL
    neurons <- list(covariate)
    if (length.weights > 1) 
      for (i in 1:(length.weights - 1)) {
        temp <- neurons[[i]] %*% weights[[i]]
        act.temp <- act.fct(temp)
        if (special) 
          neuron.deriv[[i]] <- act.deriv.fct(act.temp)
        else neuron.deriv[[i]] <- act.deriv.fct(temp)
        neurons[[i + 1]] <- cbind(1, act.temp)
      }
    if (!is.list(neuron.deriv)) 
      neuron.deriv <- list(neuron.deriv)
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    net.result <- output.act.fct(temp)
    if (special) 
      neuron.deriv[[length.weights]] <- output.act.deriv.fct(net.result)
    else neuron.deriv[[length.weights]] <- output.act.deriv.fct(temp)
    if (any(is.na(neuron.deriv))) 
      stop("neuron derivatives contain a NA; varify that the derivative function does not divide by 0", 
           call. = FALSE)
    list(neurons = neurons, neuron.deriv = neuron.deriv, net.result = net.result)
  }
calculate.gradients <-
  function (weights, length.weights, neurons, neuron.deriv, err.deriv, 
            exclude, linear.output) 
  {
    if (any(is.na(err.deriv))) 
      stop("the error derivative contains a NA; varify that the derivative function does not divide by 0 (e.g. cross entropy)", 
           call. = FALSE)
    if (!linear.output) 
      delta <- neuron.deriv[[length.weights]] * err.deriv
    else delta <- err.deriv
    gradients <- crossprod(neurons[[length.weights]], delta)
    if (length.weights > 1) 
      for (w in (length.weights - 1):1) {
        delta <- neuron.deriv[[w]] * tcrossprod(delta, remove.intercept(weights[[w + 
                                                                                   1]]))
        gradients <- c(crossprod(neurons[[w]], delta), gradients)
      }
    gradients[-exclude]
  }
plus <-
  function (gradients, gradients.old, weights, nrow.weights, ncol.weights, 
            learningrate, learningrate.factor, learningrate.limit, exclude) 
  {
    weights <- unlist(weights)
    sign.gradient <- sign(gradients)
    temp <- gradients.old * sign.gradient
    positive <- temp > 0
    negative <- temp < 0
    not.negative <- !negative
    if (any(positive)) {
      learningrate[positive] <- pmin.int(learningrate[positive] * 
                                           learningrate.factor$plus, learningrate.limit$max)
    }
    if (any(negative)) {
      weights[-exclude][negative] <- weights[-exclude][negative] + 
        gradients.old[negative] * learningrate[negative]
      learningrate[negative] <- pmax.int(learningrate[negative] * 
                                           learningrate.factor$minus, learningrate.limit$min)
      gradients.old[negative] <- 0
      if (any(not.negative)) {
        weights[-exclude][not.negative] <- weights[-exclude][not.negative] - 
          sign.gradient[not.negative] * learningrate[not.negative]
        gradients.old[not.negative] <- sign.gradient[not.negative]
      }
    }
    else {
      weights[-exclude] <- weights[-exclude] - sign.gradient * 
        learningrate
      gradients.old <- sign.gradient
    }
    list(gradients.old = gradients.old, weights = relist(weights, 
                                                         nrow.weights, ncol.weights), learningrate = learningrate)
  }
backprop <-
  function (gradients, weights, length.weights, nrow.weights, ncol.weights, 
            learningrate.bp, exclude) 
  {
    weights <- unlist(weights)
    if (!is.null(exclude)) 
      weights[-exclude] <- weights[-exclude] - gradients * 
        learningrate.bp
    else weights <- weights - gradients * learningrate.bp
    list(gradients.old = gradients, weights = relist(weights, 
                                                     nrow.weights, ncol.weights), learningrate = learningrate.bp)
  }
minus <-
  function (gradients, gradients.old, weights, length.weights, 
            nrow.weights, ncol.weights, learningrate, learningrate.factor, 
            learningrate.limit, algorithm, exclude) 
  {
    weights <- unlist(weights)
    temp <- gradients.old * gradients
    positive <- temp > 0
    negative <- temp < 0
    if (any(positive)) 
      learningrate[positive] <- pmin.int(learningrate[positive] * 
                                           learningrate.factor$plus, learningrate.limit$max)
    if (any(negative)) 
      learningrate[negative] <- pmax.int(learningrate[negative] * 
                                           learningrate.factor$minus, learningrate.limit$min)
    if (algorithm != "rprop-") {
      delta <- 10^-6
      notzero <- gradients != 0
      gradients.notzero <- gradients[notzero]
      if (algorithm == "slr") {
        min <- which.min(learningrate[notzero])
      }
      else if (algorithm == "sag") {
        min <- which.min(abs(gradients.notzero))
      }
      if (length(min) != 0) {
        temp <- learningrate[notzero] * gradients.notzero
        sum <- sum(temp[-min]) + delta
        learningrate[notzero][min] <- min(max(-sum/gradients.notzero[min], 
                                              learningrate.limit$min), learningrate.limit$max)
      }
    }
    weights[-exclude] <- weights[-exclude] - sign(gradients) * 
      learningrate
    list(gradients.old = gradients, weights = relist(weights, 
                                                     nrow.weights, ncol.weights), learningrate = learningrate)
  }
calculate.generalized.weights <-
  function (weights, neuron.deriv, net.result) 
  {
    for (w in 1:length(weights)) {
      weights[[w]] <- remove.intercept(weights[[w]])
    }
    generalized.weights <- NULL
    for (k in 1:ncol(net.result)) {
      for (w in length(weights):1) {
        if (w == length(weights)) {
          temp <- neuron.deriv[[length(weights)]][, k] * 
            1/(net.result[, k] * (1 - (net.result[, k])))
          delta <- tcrossprod(temp, weights[[w]][, k])
        }
        else {
          delta <- tcrossprod(delta * neuron.deriv[[w]], 
                              weights[[w]])
        }
      }
      generalized.weights <- cbind(generalized.weights, delta)
    }
    return(generalized.weights)
  }
generate.output <-
  function (covariate, call, rep, threshold, matrix, startweights, 
            model.list, response, err.fct, act.fct, data, list.result, 
            linear.output, exclude) 
  {
    covariate <- t(remove.intercept(t(covariate)))
    nn <- list(call = call)
    class(nn) <- c("nn")
    nn$response <- response
    nn$covariate <- covariate
    nn$model.list <- model.list
    nn$err.fct <- err.fct
    nn$act.fct <- act.fct
    nn$linear.output <- linear.output
    nn$data <- data
    nn$exclude <- exclude
    if (!is.null(matrix)) {
      nn$net.result <- NULL
      nn$weights <- NULL
      nn$generalized.weights <- NULL
      nn$startweights <- NULL
      for (i in 1:length(list.result)) {
        nn$net.result <- c(nn$net.result, list(list.result[[i]]$net.result))
        nn$weights <- c(nn$weights, list(list.result[[i]]$weights))
        nn$startweights <- c(nn$startweights, list(list.result[[i]]$startweights))
        nn$generalized.weights <- c(nn$generalized.weights, 
                                    list(list.result[[i]]$generalized.weights))
      }
      nn$result.matrix <- generate.rownames(matrix, nn$weights[[1]], 
                                            model.list)
    }
    return(nn)
  }
generate.rownames <-
  function (matrix, weights, model.list) 
  {
    rownames <- rownames(matrix)[rownames(matrix) != ""]
    for (w in 1:length(weights)) {
      for (j in 1:ncol(weights[[w]])) {
        for (i in 1:nrow(weights[[w]])) {
          if (i == 1) {
            if (w == length(weights)) {
              rownames <- c(rownames, paste("Intercept.to.", 
                                            model.list$response[j], sep = ""))
            }
            else {
              rownames <- c(rownames, paste("Intercept.to.", 
                                            w, "layhid", j, sep = ""))
            }
          }
          else {
            if (w == 1) {
              if (w == length(weights)) {
                rownames <- c(rownames, paste(model.list$variables[i - 
                                                                     1], ".to.", model.list$response[j], sep = ""))
              }
              else {
                rownames <- c(rownames, paste(model.list$variables[i - 
                                                                     1], ".to.1layhid", j, sep = ""))
              }
            }
            else {
              if (w == length(weights)) {
                rownames <- c(rownames, paste(w - 1, "layhid.", 
                                              i - 1, ".to.", model.list$response[j], 
                                              sep = ""))
              }
              else {
                rownames <- c(rownames, paste(w - 1, "layhid.", 
                                              i - 1, ".to.", w, "layhid", j, sep = ""))
              }
            }
          }
        }
      }
    }
    rownames(matrix) <- rownames
    colnames(matrix) <- 1:(ncol(matrix))
    return(matrix)
  }
relist <-
  function (x, nrow, ncol) 
  {
    list.x <- NULL
    for (w in 1:length(nrow)) {
      length <- nrow[w] * ncol[w]
      list.x[[w]] <- matrix(x[1:length], nrow = nrow[w], ncol = ncol[w])
      x <- x[-(1:length)]
    }
    list.x
  }
remove.intercept <-
  function (matrix) 
  {
    matrix(matrix[-1, ], ncol = ncol(matrix))
  }
type <-
  function (fct) 
  {
    attr(fct, "type")
  }
print.nn <-
  function (x, ...) 
  {
    matrix <- x$result.matrix
    cat("Call: ", deparse(x$call), "\n\n", sep = "")
    if (!is.null(matrix)) {
      if (ncol(matrix) > 1) {
        cat(ncol(matrix), " repetitions were calculated.\n\n", 
            sep = "")
        sorted.matrix <- matrix[, order(matrix["error", ])]
        if (any(rownames(sorted.matrix) == "aic")) {
          print(t(rbind(Error = sorted.matrix["error", 
                                              ], AIC = sorted.matrix["aic", ], BIC = sorted.matrix["bic", 
                                                                                                   ], `Reached Threshold` = sorted.matrix["reached.threshold", 
                                                                                                                                          ], Steps = sorted.matrix["steps", ])))
        }
        else {
          print(t(rbind(Error = sorted.matrix["error", 
                                              ], `Reached Threshold` = sorted.matrix["reached.threshold", 
                                                                                     ], Steps = sorted.matrix["steps", ])))
        }
      }
      else {
        cat(ncol(matrix), " repetition was calculated.\n\n", 
            sep = "")
        if (any(rownames(matrix) == "aic")) {
          print(t(matrix(c(matrix["error", ], matrix["aic", 
                                                     ], matrix["bic", ], matrix["reached.threshold", 
                                                                                ], matrix["steps", ]), dimnames = list(c("Error", 
                                                                                                                         "AIC", "BIC", "Reached Threshold", "Steps"), 
                                                                                                                       c(1)))))
        }
        else {
          print(t(matrix(c(matrix["error", ], matrix["reached.threshold", 
                                                     ], matrix["steps", ]), dimnames = list(c("Error", 
                                                                                              "Reached Threshold", "Steps"), c(1)))))
        }
      }
    }
    cat("\n")
  }
#### COMPUTE ####
compute <- function (x, covariate, rep = 1) 
{
    nn <- x
    linear.output <- nn$linear.output
    weights <- nn$weights[[rep]]
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    weights <- unlist(weights)
    if (any(is.na(weights))) 
        weights[is.na(weights)] <- 0
    weights <- relist(weights, nrow.weights, ncol.weights)
    length.weights <- length(weights)
    covariate <- as.matrix(cbind(1, covariate))
    act.fct <- nn$act.fct
    neurons <- list(covariate)
    if (length.weights > 1) 
        for (i in 1:(length.weights - 1)) {
            temp <- neurons[[i]] %*% weights[[i]]
            act.temp <- act.fct(temp)
            neurons[[i + 1]] <- cbind(1, act.temp)
        }
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    if (linear.output) 
        net.result <- temp
    else net.result <- act.fct(temp)
    list(neurons = neurons, net.result = net.result)
}

plot.nn <-
function (x, rep = NULL, x.entry = NULL, x.out = NULL, radius = 0.15, 
    arrow.length = 0.2, intercept = TRUE, intercept.factor = 0.4, 
    information = TRUE, information.pos = 0.1, col.entry.synapse = "black", 
    col.entry = "black", col.hidden = "black", col.hidden.synapse = "black", 
    col.out = "black", col.out.synapse = "black", col.intercept = "blue", 
    fontsize = 12, dimension = 6, show.weights = TRUE, file = NULL, 
    ...) 
{
    net <- x
    if (is.null(net$weights)) 
        stop("weights were not calculated")
    if (!is.null(file) && !is.character(file)) 
        stop("'file' must be a string")
    if (is.null(rep)) {
        for (i in 1:length(net$weights)) {
            if (!is.null(file)) 
                file.rep <- paste(file, ".", i, sep = "")
            else file.rep <- NULL
            grDevices::dev.new()
            plot.nn(net, rep = i, x.entry, x.out, radius, arrow.length, 
                intercept, intercept.factor, information, information.pos, 
                col.entry.synapse, col.entry, col.hidden, col.hidden.synapse, 
                col.out, col.out.synapse, col.intercept, fontsize, 
                dimension, show.weights, file.rep, ...)
        }
    }
    else {
        if (is.character(file) && file.exists(file)) 
            stop(sprintf("%s already exists", sQuote(file)))
        result.matrix <- t(net$result.matrix)
        if (rep == "best") 
            rep <- as.integer(which.min(result.matrix[, "error"]))
        if (rep > length(net$weights)) 
            stop("'rep' does not exist")
        weights <- net$weights[[rep]]
        if (is.null(x.entry)) 
            x.entry <- 0.5 - (arrow.length/2) * length(weights)
        if (is.null(x.out)) 
            x.out <- 0.5 + (arrow.length/2) * length(weights)
        width <- max(x.out - x.entry + 0.2, 0.8) * 8
        radius <- radius/dimension
        entry.label <- net$model.list$variables
        out.label <- net$model.list$response
        neuron.count <- array(0, length(weights) + 1)
        neuron.count[1] <- nrow(weights[[1]]) - 1
        neuron.count[2] <- ncol(weights[[1]])
        x.position <- array(0, length(weights) + 1)
        x.position[1] <- x.entry
        x.position[length(weights) + 1] <- x.out
        if (length(weights) > 1) 
            for (i in 2:length(weights)) {
                neuron.count[i + 1] <- ncol(weights[[i]])
                x.position[i] <- x.entry + (i - 1) * (x.out - 
                  x.entry)/length(weights)
            }
        y.step <- 1/(neuron.count + 1)
        y.position <- array(0, length(weights) + 1)
        y.intercept <- 1 - 2 * radius
        information.pos <- min(min(y.step) - 0.1, 0.2)
        if (length(entry.label) != neuron.count[1]) {
            if (length(entry.label) < neuron.count[1]) {
                tmp <- NULL
                for (i in 1:(neuron.count[1] - length(entry.label))) {
                  tmp <- c(tmp, "no name")
                }
                entry.label <- c(entry.label, tmp)
            }
        }
        if (length(out.label) != neuron.count[length(neuron.count)]) {
            if (length(out.label) < neuron.count[length(neuron.count)]) {
                tmp <- NULL
                for (i in 1:(neuron.count[length(neuron.count)] - 
                  length(out.label))) {
                  tmp <- c(tmp, "no name")
                }
                out.label <- c(out.label, tmp)
            }
        }
        grid::grid.newpage()
        for (k in 1:length(weights)) {
            for (i in 1:neuron.count[k]) {
                y.position[k] <- y.position[k] + y.step[k]
                y.tmp <- 0
                for (j in 1:neuron.count[k + 1]) {
                  y.tmp <- y.tmp + y.step[k + 1]
                  result <- calculate.delta(c(x.position[k], 
                    x.position[k + 1]), c(y.position[k], y.tmp), 
                    radius)
                  x <- c(x.position[k], x.position[k + 1] - result[1])
                  y <- c(y.position[k], y.tmp + result[2])
                  grid::grid.lines(x = x, y = y, arrow = grid::arrow(length = grid::unit(0.15, 
                    "cm"), type = "closed"), gp = grid::gpar(fill = col.hidden.synapse, 
                    col = col.hidden.synapse, ...))
                  if (show.weights) 
                    draw.text(label = weights[[k]][neuron.count[k] - 
                      i + 2, neuron.count[k + 1] - j + 1], x = c(x.position[k], 
                      x.position[k + 1]), y = c(y.position[k], 
                      y.tmp), xy.null = 1.25 * result, color = col.hidden.synapse, 
                      fontsize = fontsize - 2, ...)
                }
                if (k == 1) {
                  grid::grid.lines(x = c((x.position[1] - arrow.length), 
                    x.position[1] - radius), y = y.position[k], 
                    arrow = grid::arrow(length = grid::unit(0.15, "cm"), 
                      type = "closed"), gp = grid::gpar(fill = col.entry.synapse, 
                      col = col.entry.synapse, ...))
                  draw.text(label = entry.label[(neuron.count[1] + 
                    1) - i], x = c((x.position - arrow.length), 
                    x.position[1] - radius), y = c(y.position[k], 
                    y.position[k]), xy.null = c(0, 0), color = col.entry.synapse, 
                    fontsize = fontsize, ...)
                  grid::grid.circle(x = x.position[k], y = y.position[k], 
                    r = radius, gp = grid::gpar(fill = "white", col = col.entry, 
                      ...))
                }
                else {
                  grid::grid.circle(x = x.position[k], y = y.position[k], 
                    r = radius, gp = grid::gpar(fill = "white", col = col.hidden, 
                      ...))
                }
            }
        }
        out <- length(neuron.count)
        for (i in 1:neuron.count[out]) {
            y.position[out] <- y.position[out] + y.step[out]
            grid::grid.lines(x = c(x.position[out] + radius, x.position[out] + 
                arrow.length), y = y.position[out], arrow = grid::arrow(length = grid::unit(0.15, 
                "cm"), type = "closed"), gp = grid::gpar(fill = col.out.synapse, 
                col = col.out.synapse, ...))
            draw.text(label = out.label[(neuron.count[out] + 
                1) - i], x = c((x.position[out] + radius), x.position[out] + 
                arrow.length), y = c(y.position[out], y.position[out]), 
                xy.null = c(0, 0), color = col.out.synapse, fontsize = fontsize, 
                ...)
            grid::grid.circle(x = x.position[out], y = y.position[out], 
                r = radius, gp = grid::gpar(fill = "white", col = col.out, 
                  ...))
        }
        if (intercept) {
            for (k in 1:length(weights)) {
                y.tmp <- 0
                x.intercept <- (x.position[k + 1] - x.position[k]) * 
                  intercept.factor + x.position[k]
                for (i in 1:neuron.count[k + 1]) {
                  y.tmp <- y.tmp + y.step[k + 1]
                  result <- calculate.delta(c(x.intercept, x.position[k + 
                    1]), c(y.intercept, y.tmp), radius)
                  x <- c(x.intercept, x.position[k + 1] - result[1])
                  y <- c(y.intercept, y.tmp + result[2])
                  grid::grid.lines(x = x, y = y, arrow = grid::arrow(length = grid::unit(0.15, 
                    "cm"), type = "closed"), gp = grid::gpar(fill = col.intercept, 
                    col = col.intercept, ...))
                  xy.null <- cbind(x.position[k + 1] - x.intercept - 
                    2 * result[1], -(y.tmp - y.intercept + 2 * 
                    result[2]))
                  if (show.weights) 
                    draw.text(label = weights[[k]][1, neuron.count[k + 
                      1] - i + 1], x = c(x.intercept, x.position[k + 
                      1]), y = c(y.intercept, y.tmp), xy.null = xy.null, 
                      color = col.intercept, alignment = c("right", 
                        "bottom"), fontsize = fontsize - 2, ...)
                }
                grid::grid.circle(x = x.intercept, y = y.intercept, 
                  r = radius, gp = grid::gpar(fill = "white", col = col.intercept, 
                    ...))
                grid::grid.text(1, x = x.intercept, y = y.intercept, 
                  gp = grid::gpar(col = col.intercept, ...))
            }
        }
        if (information) 
          grid::grid.text(paste("Error: ", round(result.matrix[rep, 
                "error"], 6), "   Steps: ", result.matrix[rep, 
                "steps"], sep = ""), x = 0.5, y = information.pos, 
                just = "bottom", gp = grid::gpar(fontsize = fontsize + 
                  2, ...))
        if (!is.null(file)) {
            weight.plot <- grDevices::recordPlot()
            save(weight.plot, file = file)
        }
    }
}
calculate.delta <-
function (x, y, r) 
{
    delta.x <- x[2] - x[1]
    delta.y <- y[2] - y[1]
    x.null <- r/sqrt(delta.x^2 + delta.y^2) * delta.x
    if (y[1] < y[2]) 
        y.null <- -sqrt(r^2 - x.null^2)
    else if (y[1] > y[2]) 
        y.null <- sqrt(r^2 - x.null^2)
    else y.null <- 0
    c(x.null, y.null)
}
draw.text <-
function (label, x, y, xy.null = c(0, 0), color, alignment = c("left", 
    "bottom"), ...) 
{
    x.label <- x[1] + xy.null[1]
    y.label <- y[1] - xy.null[2]
    x.delta <- x[2] - x[1]
    y.delta <- y[2] - y[1]
    angle = atan(y.delta/x.delta) * (180/pi)
    if (angle < 0) 
        angle <- angle + 0
    else if (angle > 0) 
        angle <- angle - 0
    if (is.numeric(label)) 
        label <- round(label, 5)
    vp <- grid::viewport(x = x.label, y = y.label, width = 0, height = , 
        angle = angle, name = "vp1", just = alignment)
    grid::grid.text(label, x = 0, y = grid::unit(0.75, "mm"), just = alignment, 
        gp = grid::gpar(col = color, ...), vp = vp)
}


confidence.interval <- function (x, alpha = 0.05) 
{
    net <- x
    covariate <- cbind(1, net$covariate)
    response <- net$response
    err.fct <- net$err.fct
    act.fct <- net$act.fct
    linear.output <- net$linear.output
    exclude <- net$exclude
    list.weights <- net$weights
    rep <- length(list.weights)
    if (is.null(list.weights)) 
        stop("weights were not calculated")
    nrow.weights <- sapply(list.weights[[1]], nrow)
    ncol.weights <- sapply(list.weights[[1]], ncol)
    lower.ci <- NULL
    upper.ci <- NULL
    nic <- NULL
    for (i in 1:rep) {
        weights <- list.weights[[i]]
        error <- net$result.matrix["error", i]
        if (length(weights) > 2) 
            stop("nic and confidence intervals will not be calculated for more than one hidden layer of neurons", 
                call. = FALSE)
        result.nic <- calculate.information.matrices(covariate, 
            response, weights, err.fct, act.fct, exclude, linear.output)
        nic <- c(nic, error + result.nic$trace)
        if (!is.null(result.nic$variance)) {
            if (all(diag(result.nic$variance) >= 0)) {
                weights.vector <- unlist(weights)
                if (!is.null(exclude)) {
                  d <- rep(NA, length(weights.vector))
                  d[-exclude] <- stats::qnorm(1 - alpha/2) * sqrt(diag(result.nic$variance))/sqrt(nrow(covariate))
                }
                else {
                  d <- stats::qnorm(1 - alpha/2) * sqrt(diag(result.nic$variance))/sqrt(nrow(covariate))
                }
                lower.ci <- c(lower.ci, list(relist(weights.vector - 
                  d, nrow.weights, ncol.weights)))
                upper.ci <- c(upper.ci, list(relist(weights.vector + 
                  d, nrow.weights, ncol.weights)))
            }
        }
    }
    if (length(lower.ci) < rep) 
        warning(sprintf("%s of %s repetition(s) could not calculate confidence intervals for the weights; varify that the neural network does not contain irrelevant neurons", 
            length(lower.ci), rep), call. = F)
    list(lower.ci = lower.ci, upper.ci = upper.ci, nic = nic)
}
calculate.information.matrices <-
function (covariate, response, weights, err.fct, act.fct, exclude, 
    linear.output) 
{
    temp <- act.fct
    if (type(act.fct) == "logistic") {
        act.deriv.fct <- function(x) {
            act.fct(x) * (1 - act.fct(x))
        }
        act.deriv2.fct <- function(x) {
            act.fct(x) * (1 - act.fct(x)) * (1 - (2 * act.fct(x)))
        }
    }
    else {
        attr(temp, "type") <- NULL
        act.deriv.fct <- differentiate(temp)
        act.deriv2.fct <- differentiate(temp, hessian = T)
    }
    temp <- err.fct
    attr(temp, "type") <- NULL
    err.deriv.fct <- differentiate(temp)
    err.deriv2.fct <- differentiate(temp, hessian = T)
    length.weights <- length(weights)
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    if (linear.output) {
        output.act.fct <- function(x) {
            x
        }
        output.act.deriv.fct <- function(x) {
            matrix(1, nrow(x), ncol(x))
        }
        output.act.deriv2.fct <- function(x) {
            matrix(0, nrow(x), ncol(x))
        }
    }
    else {
        output.act.fct <- act.fct
        output.act.deriv.fct <- act.deriv.fct
        output.act.deriv2.fct <- act.deriv2.fct
    }
    neuron.deriv <- NULL
    neuron.deriv2 <- NULL
    neurons <- list(covariate)
    if (length.weights > 1) 
        for (i in 1:(length.weights - 1)) {
            temp <- neurons[[i]] %*% weights[[i]]
            act.temp <- act.fct(temp)
            neuron.deriv[[i]] <- act.deriv.fct(temp)
            neuron.deriv2[[i]] <- act.deriv2.fct(temp)
            neurons[[i + 1]] <- cbind(1, act.temp)
        }
    if (!is.list(neuron.deriv)) 
        neuron.deriv <- list(neuron.deriv)
    if (!is.list(neuron.deriv2)) 
        neuron.deriv2 <- list(neuron.deriv2)
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    net.result <- output.act.fct(temp)
    neuron.deriv[[length.weights]] <- output.act.deriv.fct(temp)
    neuron.deriv2[[length.weights]] <- output.act.deriv2.fct(temp)
    err.deriv <- err.deriv.fct(net.result, response)
    err.deriv2 <- err.deriv2.fct(net.result, response)
    if (any(is.na(unlist(neuron.deriv2)))) {
        return(list(nic = NA, hessian = NULL))
        warning("neuron.deriv2 contains NA; this might be caused by a wrong choice of 'act.fct'")
    }
    if (any(is.na(err.deriv)) || any(is.na(err.deriv2))) {
        if (type(err.fct) == "ce") {
            one <- which(net.result == 1)
            if (length(one) > 0) {
                for (i in 1:length(one)) {
                  if (response[one[i]] == 1) {
                    err.deriv[one[i]] <- 1
                    err.deriv2[one[i]] <- 1
                  }
                }
            }
            zero <- which(net.result == 0)
            if (length(zero) > 0) {
                for (i in 1:length(zero)) {
                  if (response[zero[i]] == 0) {
                    err.deriv[zero[i]] <- 1
                    err.deriv2[zero[i]] <- -1
                  }
                }
            }
        }
    }
    if (any(is.na(err.deriv))) {
        return(list(nic = NA, hessian = NULL))
        warning("err.deriv contains NA; this might be caused by a wrong choice of 'act.fct'")
    }
    if (any(is.na(err.deriv2))) {
        return(list(nic = NA, hessian = NULL))
        warning("err.deriv2 contains NA; this might be caused by a wrong choice of 'act.fct'")
    }
    if (length.weights == 2) {
        length.betha <- (nrow.weights * ncol.weights)[1]
        length.alpha <- (nrow.weights * ncol.weights)[2]
        total.length.weights <- length.alpha + length.betha
        betha.ind <- matrix(1:length.betha, nrow = nrow.weights[1], 
            ncol = ncol.weights[1])
        alpha.ind <- matrix(1:length.alpha, nrow = nrow.weights[2], 
            ncol = ncol.weights[2])
        Hesse <- matrix(NA, nrow = total.length.weights, ncol = total.length.weights)
        Cross.Gradient <- matrix(NA, nrow = total.length.weights, 
            ncol = total.length.weights)
        Cross.Gradient2 <- matrix(NA, nrow = total.length.weights, 
            ncol = total.length.weights)
        for (i in 1:total.length.weights) {
            for (j in 1:total.length.weights) {
                if (is.null(exclude) || all(i != exclude & j != 
                  exclude)) {
                  if (i <= length.betha) {
                    temp <- which(betha.ind == i, arr.ind = T)
                    r <- temp[1]
                    s <- temp[2]
                  }
                  else {
                    temp <- which(alpha.ind == (i - length.betha), 
                      arr.ind = T)
                    r <- temp[1]
                    s <- temp[2]
                  }
                  if (j <= length.betha) {
                    temp <- which(betha.ind == j, arr.ind = T)
                    u <- temp[1]
                    v <- temp[2]
                  }
                  else {
                    temp <- which(alpha.ind == (j - length.betha), 
                      arr.ind = T)
                    u <- temp[1]
                    v <- temp[2]
                  }
                  if ((i <= length.betha) && (j <= length.betha)) {
                    Cross.Gradient[i, j] <- sum(((err.deriv^2 * 
                      neuron.deriv[[2]]^2) %*% (weights[[2]][(s + 
                      1), ] * weights[[2]][(v + 1), ])) * neuron.deriv[[1]][, 
                      s] * neurons[[1]][, r] * neuron.deriv[[1]][, 
                      v] * neurons[[1]][, u])
                    Cross.Gradient2[i, j] <- sum(((err.deriv2 * 
                      neuron.deriv[[2]]^2) %*% (weights[[2]][(s + 
                      1), ] * weights[[2]][(v + 1), ])) * neuron.deriv[[1]][, 
                      s] * neurons[[1]][, r] * neuron.deriv[[1]][, 
                      v] * neurons[[1]][, u])
                    if (s == v) 
                      Hesse[i, j] <- sum(neuron.deriv[[1]][, 
                        s] * neurons[[1]][, r] * neuron.deriv[[1]][, 
                        v] * neurons[[1]][, u] * ((neuron.deriv2[[2]] * 
                        err.deriv) %*% (weights[[2]][(s + 1), 
                        ] * weights[[2]][(v + 1), ]))) + sum(neuron.deriv2[[1]][, 
                        s] * neurons[[1]][, r] * neurons[[1]][, 
                        u] * ((neuron.deriv[[2]] * err.deriv) %*% 
                        weights[[2]][(s + 1), ]))
                    else Hesse[i, j] <- sum(neuron.deriv[[1]][, 
                      s] * neurons[[1]][, r] * neuron.deriv[[1]][, 
                      v] * neurons[[1]][, u] * ((neuron.deriv2[[2]] * 
                      err.deriv) %*% (weights[[2]][(s + 1), ] * 
                      weights[[2]][(v + 1), ])))
                  }
                  else if ((i > length.betha) && (j > length.betha)) {
                    if (v == s) {
                      Cross.Gradient[i, j] <- sum(err.deriv[, 
                        v]^2 * (neuron.deriv[[2]][, s] * neurons[[2]][, 
                        r] * neuron.deriv[[2]][, v] * neurons[[2]][, 
                        u]))
                      Cross.Gradient2[i, j] <- sum(err.deriv2[, 
                        v] * (neuron.deriv[[2]][, s] * neurons[[2]][, 
                        r] * neuron.deriv[[2]][, v] * neurons[[2]][, 
                        u]))
                    }
                    else {
                      Cross.Gradient[i, j] <- 0
                      Cross.Gradient2[i, j] <- 0
                    }
                    if (v == s) 
                      Hesse[i, j] <- sum(neuron.deriv2[[2]][, 
                        s] * err.deriv[, s] * neurons[[2]][, 
                        u] * neurons[[2]][, r])
                    else Hesse[i, j] <- 0
                  }
                  else if ((i > length.betha) && (j <= length.betha)) {
                    Cross.Gradient[i, j] <- sum(err.deriv[, s]^2 * 
                      (neuron.deriv[[2]][, s] * neurons[[2]][, 
                        r] * (neuron.deriv[[2]][, s] * weights[[2]][(v + 
                        1), s]) * neuron.deriv[[1]][, v] * neurons[[1]][, 
                        u]))
                    Cross.Gradient2[i, j] <- sum(err.deriv2[, 
                      s] * (neuron.deriv[[2]][, s] * neurons[[2]][, 
                      r] * (neuron.deriv[[2]][, s] * weights[[2]][(v + 
                      1), s]) * neuron.deriv[[1]][, v] * neurons[[1]][, 
                      u]))
                    if (v == r) 
                      Hesse[i, j] <- sum(neurons[[2]][, r] * 
                        neuron.deriv[[1]][, v] * neurons[[1]][, 
                        u] * neuron.deriv2[[2]][, s] * err.deriv[, 
                        s] * weights[[2]][(v + 1), s]) + sum(neuron.deriv[[2]][, 
                        s] * err.deriv[, s] * neurons[[1]][, 
                        u] * neuron.deriv[[1]][, v])
                    else Hesse[i, j] <- sum(neurons[[2]][, r] * 
                      neuron.deriv[[1]][, v] * neurons[[1]][, 
                      u] * neuron.deriv2[[2]][, s] * err.deriv[, 
                      s] * weights[[2]][(v + 1), s])
                  }
                  else {
                    Cross.Gradient[i, j] <- sum(err.deriv[, v]^2 * 
                      (neuron.deriv[[2]][, v] * neurons[[2]][, 
                        u] * (neuron.deriv[[2]][, v] * weights[[2]][(s + 
                        1), v]) * neuron.deriv[[1]][, s] * neurons[[1]][, 
                        r]))
                    Cross.Gradient2[i, j] <- sum(err.deriv2[, 
                      v] * (neuron.deriv[[2]][, v] * neurons[[2]][, 
                      u] * (neuron.deriv[[2]][, v] * weights[[2]][(s + 
                      1), v]) * neuron.deriv[[1]][, s] * neurons[[1]][, 
                      r]))
                    if (s == u) 
                      Hesse[i, j] <- sum(neurons[[2]][, u] * 
                        neuron.deriv[[1]][, s] * neurons[[1]][, 
                        r] * neuron.deriv2[[2]][, v] * err.deriv[, 
                        v] * weights[[2]][(s + 1), v]) + sum(neuron.deriv[[2]][, 
                        v] * err.deriv[, v] * neurons[[1]][, 
                        r] * neuron.deriv[[1]][, s])
                    else Hesse[i, j] <- sum(neurons[[2]][, u] * 
                      neuron.deriv[[1]][, s] * neurons[[1]][, 
                      r] * neuron.deriv2[[2]][, v] * err.deriv[, 
                      v] * weights[[2]][(s + 1), v])
                  }
                }
            }
        }
    }
    else if (length.weights == 1) {
        length.alpha <- sum(nrow.weights * ncol.weights)
        alpha.ind <- matrix(1:length.alpha, nrow = nrow.weights[1], 
            ncol = ncol.weights[1])
        Hesse <- matrix(NA, nrow = length.alpha, ncol = length.alpha)
        Cross.Gradient <- matrix(NA, nrow = length.alpha, ncol = length.alpha)
        Cross.Gradient2 <- matrix(NA, nrow = length.alpha, ncol = length.alpha)
        for (i in 1:length.alpha) {
            for (j in 1:length.alpha) {
                if (is.null(exclude) || all(i != exclude & j != 
                  exclude)) {
                  r <- which(alpha.ind == i, arr.ind = T)[1]
                  s <- which(alpha.ind == i, arr.ind = T)[2]
                  u <- which(alpha.ind == j, arr.ind = T)[1]
                  v <- which(alpha.ind == j, arr.ind = T)[2]
                  if (s == v) {
                    Hesse[i, j] <- sum(neuron.deriv2[[1]][, s] * 
                      err.deriv[, s] * neurons[[1]][, r] * neurons[[1]][, 
                      u])
                    Cross.Gradient[i, j] <- sum(neuron.deriv[[1]][, 
                      s]^2 * err.deriv[, s]^2 * neurons[[1]][, 
                      r] * neurons[[1]][, u])
                    Cross.Gradient2[i, j] <- sum(neuron.deriv[[1]][, 
                      s]^2 * err.deriv2[, s] * neurons[[1]][, 
                      r] * neurons[[1]][, u])
                  }
                  else {
                    Hesse[i, j] <- 0
                    Cross.Gradient[i, j] <- 0
                    Cross.Gradient2[i, j] <- 0
                  }
                }
            }
        }
    }
    B <- Cross.Gradient/nrow(neurons[[1]])
    A <- (Cross.Gradient2 + Hesse)/nrow(neurons[[1]])
    if (!is.null(exclude)) {
        B <- as.matrix(B[-exclude, -exclude])
        A <- as.matrix(A[-exclude, -exclude])
    }
    if (det(A) == 0) {
        trace <- NA
        variance <- NULL
    }
    else {
        A.inv <- MASS::ginv(A)
        variance <- A.inv %*% B %*% A.inv
        trace <- sum(diag(B %*% A.inv))
    }
    return(list(trace = trace, variance = variance))
}

#### PREDICTION ####
prediction <-
function (x, list.glm = NULL) 
{
    nn <- x
    data.result <- calculate.data.result(response = nn$response, 
        model.list = nn$model.list, covariate = nn$covariate)
    predictions <- calculate.predictions(covariate = nn$covariate, 
        data.result = data.result, list.glm = list.glm, matrix = nn$result.matrix, 
        list.net.result = nn$net.result, model.list = nn$model.list)
    if (type(nn$err.fct) == "ce" && all(data.result >= 0) && 
        all(data.result <= 1)) 
        data.error <- sum(nn$err.fct(data.result, nn$response), 
            na.rm = T)
    else data.error <- sum(nn$err.fct(data.result, nn$response))
    cat("Data Error:\t", data.error, ";\n", sep = "")
    predictions
}
calculate.predictions <-
function (covariate, data.result, list.glm, matrix, list.net.result, 
    model.list) 
{
    not.duplicated <- !duplicated(covariate)
    nrow.notdupl <- sum(not.duplicated)
    covariate.mod <- matrix(covariate[not.duplicated, ], nrow = nrow.notdupl)
    predictions <- list(data = cbind(covariate.mod, matrix(data.result[not.duplicated, 
        ], nrow = nrow.notdupl)))
    if (!is.null(matrix)) {
        for (i in length(list.net.result):1) {
            pred.temp <- cbind(covariate.mod, matrix(list.net.result[[i]][not.duplicated, 
                ], nrow = nrow.notdupl))
            predictions <- eval(parse(text = paste("c(list(rep", 
                i, "=pred.temp), predictions)", sep = "")))
        }
    }
    if (!is.null(list.glm)) {
        for (i in 1:length(list.glm)) {
            pred.temp <- cbind(covariate.mod, matrix(list.glm[[i]]$fitted.values[not.duplicated], 
                nrow = nrow.notdupl))
            text <- paste("c(predictions, list(glm.", names(list.glm[i]), 
                "=pred.temp))", sep = "")
            predictions <- eval(parse(text = text))
        }
    }
    for (i in 1:length(predictions)) {
        colnames(predictions[[i]]) <- c(model.list$variables, 
            model.list$response)
        if (nrow(covariate) > 1) 
            for (j in (1:ncol(covariate))) predictions[[i]] <- predictions[[i]][order(predictions[[i]][, 
                j]), ]
        rownames(predictions[[i]]) <- 1:nrow(predictions[[i]])
    }
    predictions
}
calculate.data.result <-
function (response, covariate, model.list) 
{
    duplicated <- duplicated(covariate)
    if (!any(duplicated)) {
        return(response)
    }
    which.duplicated <- seq_along(duplicated)[duplicated]
    which.not.duplicated <- seq_along(duplicated)[!duplicated]
    ncol.response <- ncol(response)
    if (ncol(covariate) == 1) {
        for (each in which.not.duplicated) {
            out <- NULL
            if (length(which.duplicated) > 0) {
                out <- covariate[which.duplicated, ] == covariate[each, 
                  ]
                if (any(out)) {
                  rows <- c(each, which.duplicated[out])
                  response[rows, ] = matrix(colMeans(matrix(response[rows, 
                    ], ncol = ncol.response)), ncol = ncol.response, 
                    nrow = length(rows), byrow = T)
                  which.duplicated <- which.duplicated[-out]
                }
            }
        }
    }
    else {
        tcovariate <- t(covariate)
        for (each in which.not.duplicated) {
            out <- NULL
            if (length(which.duplicated) > 0) {
                out <- apply(tcovariate[, which.duplicated] == 
                  covariate[each, ], 2, FUN = all)
                if (any(out)) {
                  rows <- c(each, which.duplicated[out])
                  response[rows, ] = matrix(colMeans(matrix(response[rows, 
                    ], ncol = ncol.response)), ncol = ncol.response, 
                    nrow = length(rows), byrow = T)
                  which.duplicated <- which.duplicated[-out]
                }
            }
        }
    }
    response
}


####Work in progress ####


  #
  #StartingNode
    # Starting Training number of hidden nodes,
    #if left blank defualt to:
  #increaeRate
    #The rate at which the number of hidden nodes increase per level of classification,
    #if left blank, defualt to: startingnode + 100 * classification level.
  #Dat
    #Your data set
  #final.id
    #ending id you want to classify data too.
    #should be in Dat[x,] fourm... input x
  #classColumn.range
    #all columns which are classes which incompase final.id
    #should be in form:  Dat[a:b,]... Input should be 

  #inputDat
    #should be data you want to test
    #should not include the column you want to classify, and should not contain any subclasesses or classes just the data to identify said classes
    #should be one row
muiltiNet <- function (#){
  # StartingNode = 0
  #LdecreaeRate =  0 #Linear rate in which nodes increase              #Removed for now and repalced with a function with a set of rules.
  #EdecreaseRate = 1 #exponetial increase rate
  hiddenMode = 3,
  
  final.id = 1,
  classColumn.range = 2:4,
  Dat = idenity,
  inputDat = 0,
  thres = 0.01) {
  if (sum(Dat) == 0) {
    return("ERROR No Dat: Dat == 0")
  }
  if (final.id == 0) {
    return("ERROR: Final.id == 0: No Column to train for selected")
  }
  if (sum(classColumn.range) == 0) {
    return("ERROR: no classcolumn.range")
  }
  if (sum(inputDat) == 0) {
    return("No input Data")
  }
  #Setting up classrange
  equationPLus <-
    Dat #what we will take columnnames() from to generate C~A+B
  if (sum(classColumn.range) == 0) {
    #Then all columns will be considered for subclasses
    print("All columns will be considered for SubClasses")
    
    classColumn.range <- Dat
    classColumn.range[, final.id] <- NULL
    return("This function is not supported yet.... Please provide a value for classColumn.range")
    
  } else{
    #classColumn.range <- Dat[,classColumn.range]
    #delete classColumn.Range from column names
    #equationPLus[,classColumn.range] <- NULL
    holder <- equationPLus
    for (i in classColumn.range) {
      holder[paste(colnames(equationPLus[i]))] <- NULL
      #equationPLus[,i] <- NULL
      print(i)
      
    }
    equationPLus <- holder
    
    classColumn.range <- Dat[, classColumn.range]
  }
  #Create final.id
  equationPLus[, final.id] <- NULL
  
  
  hiddenSelect <- function(hidd) {
    #w will be Dat
    if (hidd == 1) {
      hiddenMode <- round(length(colnames(equationPLus)))
      
    } else if (hidd == 2) {
      hiddenMode <-
        round((round(length(
          colnames(equationPLus)
        )) + 1) / (2 / 3))
    } else if (hidd == 3) {
      hiddenMode <-
        round(sqrt(length(colnames(equationPLus)) *  length(Dat[, 1])))
    }
    return(hiddenMode)
  }
  #if(StartingNode == 0){#Starting Node defualt config
  #Then have starting number of hiddens equal to num of columns
  #StartingNode = round(length(colnames(equationPLus)))
  
  #}
  #if(LdecreaeRate == 0){
  #Then determine a way of desiding % of data lost
  #}
  #col_names.final.id <- colnames(Dat)[,final.id]
  #final.id <- Dat[,final.id]
  #colnames(final.id) <- col_names.final.id
  
  #Find all lengths of sub colums
  
  #if final.id == sub columns with least # of categories, use normal neuralnet
  #and print something about it
  classColumn.range.names <- colnames(classColumn.range)
  classColumn.range.values <- numeric(0)
  for (i in 1:length(classColumn.range[1, ])) {
    classColumn.range.values[i] <-
      length(table(classColumn.range[, i]))
  }
  
  order.Set <-
    rbind.data.frame((1:length(classColumn.range.names)), classColumn.range.values) #hash map for names the valules in the 1:length() will corilate to classColumn.names
  order.Set <- order.Set[order(order.Set[2, ])]
  
  
  setValuefor.Order.Set <-
    length(order.Set[1, ]) #Needs to be counted here before it is cutdown
  for (i in 1:setValuefor.Order.Set) {
    #will loop for all the subclasses
    
    if (length(order.Set) == 0) {
      print("breaking order.Set == o")
      break
    }
    #Starting node increase area
    #StartingNode = StartingNode + LdecreaeRate * EdecreaseRate
    #StartingNode = StartingNode - LdecreaeRate * EdecreaseRate
    #hiddenSelect(hiddenMode) has now been added directly into node equation
    
    
    lowest <- order.Set[1]
    order.Set[1] <- NULL
    
    col_names <- colnames(equationPLus)
    #Train algorithum
    model_formula <-
      paste(paste(classColumn.range.names[lowest[1, 1]]),
            '~',
            paste(col_names, collapse = '+', sep = ''))
    nn 
    <-neuralnet(model_formula,
                Dat,
                threshold = thres,
                hidden = hiddenSelect(hiddenMode))
    plot(nn) #just to visulize and test remove this for final
    net.results <- compute(nn, inputDat)
    net.results <- round(net.results$net.result)
    #Dat <- Dat[-which(classColumn.range != net.results), ]
    #delete rows which do not have value of net.results in
    #Dat <- Dat[-which(classColumn.range[,1] != sample(net.results, length(classColumn.range[,1]), 1)), ] #need to change first classColumn.range[,1] to something more adaptive
    #Dat <- Dat[-which(classColumn.range[ classColumn.range.names[lowest[1,1]]] != sample(net.results, length(classColumn.range[,1]), 1)), ] #need to change first classColumn.range[,1] to something more adaptive
    #Dat2 <- Dat[-which(Dat[ classColumn.range.names[lowest[1,1]]] != sample(net.results, length(Dat[,1]), 1)), ] #need to change first classColumn.range[,1] to something more adaptive
    
    #Alternative to -which()
    
    holder <- numeric(0)
    for (j in 1:length(Dat[, 1])) {
      if (Dat[j, classColumn.range.names[lowest[1, 1]]] == net.results) {
        print(j) #print when equal
        holder <- rbind.data.frame(holder, Dat[j, ])
      }
    }
    Dat <- holder
    
    
    
    
    #print(lowest)
    #cat("Dat Row num", length(Dat[,1]))
    #print(Dat[ classColumn.range.names[lowest[1,1]]])
    #cat("netresults", net.results)
    
    
    #}
    #first time through check to make sure that there arnt less final.ids then class categories
    #if(i == 1){
    # if(length(table(final.id)) <= lowest[2,]){
    #    print("ERROR: final.id is  <= one of the subclasses")
    #    print("Jumping directly to training off of nn")
    ##
    #    #Train algorithum right here
    #
    #  } else{
    #    #Eliminate all Class/ subclass columns which are >= final.id
    #    for(j in 1:length(order.Set[2,])){
    #      if(length(table(final.id)) <= order.Set[2,length(order.Set[2,])]){                 #Testing how final.id compares to classColumns.range
    #        print("Useless sub column detected")                                             #having a bunch of errors this can wait
    #        print(".... >= id's to final.id")
    #        #remove longest data thing
    #        order.Set[length(order.Set[2,])] <- NULL
    #      }
    #
    #   }
    #  }
    #Train Algorithum
    #print(i)
    #nn <- neuralnet()
    
    # }
  }
  #train for final.id
  model_formula <-
    paste(paste(colnames(Dat[final.id])),
          '~',
          paste(col_names, collapse = '+', sep = ''))
  nn <-
    neuralnet(model_formula,
              Dat,
              threshold = thres,
              hidden = hiddenSelect(hiddenMode))
  net.results <- compute(nn, inputDat)
  net.results <- round(net.results$net.result)
  return(net.results)
}

  