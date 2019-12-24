
################################################################################
# Wtilde
################################################################################
Wtilde.sim <- function(W, Wperp, H){
  
  library(stats)
  steps <- dim(W)[1] #W1 and W1perp are steps x N matrices
  N <- dim(W)[2]
  alpha <- H - 1/2
  stopifnot(dim(Wperp) == c(steps,N))
  
  bstar <- function(k) {
    ((k^(alpha + 1) - (k - 1)^(alpha + 1))/(alpha + 1))^(1/alpha)
  }
  
  dt <- 1/steps
  dt1 <- dt^(H + 1/2)
  dt.2H <- dt^(2 * H)
  sqrt.dt <- sqrt(dt)
  sqrt.2H <- sqrt(2 * H)
  dt.H <- dt^H
  rhoH <- sqrt.2H/(H + 1/2)
  x <- 1:steps
  
  Gamma <- bstar(x)^alpha
  Gamma[1] <- 0
  
  f <- function(j){ # j is the path number so 1 <= j <= N
    Wr <- W[steps:1,j]
    Y <- convolve(Gamma, Wr, type = "open")[1:steps]
    What <- 1/sqrt.2H * (rhoH * W[,j] + sqrt(1 - rhoH^2) * Wperp[,j])
    return((What + Y) * dt.H)
  }
  
  Wtilde <- sqrt.2H * sapply(1:N, f) 
  return(Wtilde)
}

################################################################################
# hybridScheme
################################################################################
hybridScheme <- function(xi,params)function(N,steps,expiries) 
{
  eta <- params$eta
  H <- params$H
  rho <- params$rho
  
  W <- matrix(rnorm(N*steps),nrow = steps,ncol=N) # Volatility Brownian
  Wperp <- matrix(rnorm(N*steps),nrow = steps,ncol=N) # Second volatility Brownian
  Zperp <- matrix(rnorm(N*steps),nrow = steps,ncol=N)
  Z <- rho * W + sqrt(1 - rho * rho) * Zperp # Stock price Brownian
  
  Wtilde <- Wtilde.sim(W, Wperp, H) 
  
  S <- function(expiry) {
    
    dt <- expiry/steps
    ti <- (1:steps)*dt # t-grid
    
    Wtilde.H <- expiry^H * Wtilde # rescale for the time-interval
    xi.t <- xi(ti) # xi evaluated on the grid
    v1 <- xi.t * exp(eta * Wtilde.H - 1/2 * eta^2 * ti^(2*H))
    v0 <- rep(xi(0),N)
    v <- rbind(v0,v1[-steps,])  
    logs <- apply(sqrt(v*dt) * Z - v/2*dt,2,sum)
    s <- exp(logs)
    return(s)
  }
  
  st <- t(sapply(expiries,S))
  
  return(st)
}

###############################################################
# Amended hybrid BSS code to return quadratic variation w
###############################################################

hybridScheme.w <- function(xi,params)function(N,steps,expiries) 
{
  eta <- params$eta
  H <- params$H
  rho <- params$rho
  
  W <- matrix(rnorm(N*steps),nrow = steps,ncol=N) # Volatility Brownian
  Wperp <- matrix(rnorm(N*steps),nrow = steps,ncol=N) # Second volatility Brownian
  
  Wtilde <- Wtilde.sim(W, Wperp, H) 
  
  w <- function(expiry) {
    
    dt <- expiry/steps
    ti <- (1:steps)*dt # t-grid
    
    Wtilde.H <- expiry^H * Wtilde # rescale for the time-interval
    xi.t <- xi(ti) # xi evaluated on the grid
    v1 <- xi.t * exp(eta * Wtilde.H - 1/2 * eta^2 * ti^(2*H))
    v0 <- rep(xi(0),N)
    v <- rbind(v0,v1[-steps,])  
    w <- apply((v + v1)/2 * dt,2,sum)
    #logs <- apply(sqrt(v*dt) * Z - v/2*dt,2,sum)
    return(w)
  }
  
  wt <- t(sapply(expiries,w))
  #wt <- w(expiries[3])
  
  return(wt)
}

###############################################################
#
# Amended hybrid BSS code to return log S 
# and quadratic variation w.  
# Odd rows have log S; even rows have w
#
###############################################################
hybridScheme.sw <- function(xi,params)function(N,steps,expiries) 
{
  eta <- params$eta
  H <- params$H
  rho <- params$rho
  
  W <- matrix(rnorm(N*steps),nrow = steps,ncol=N) # Volatility Brownian
  Wperp <- matrix(rnorm(N*steps),nrow = steps,ncol=N) # Second volatility Brownian
  Zperp <- matrix(rnorm(N*steps),nrow = steps,ncol=N)
  Z <- rho * W + sqrt(1 - rho * rho) * Zperp # Stock price Brownian
  
  Wtilde <- Wtilde.sim(W, Wperp, H) 
  
  xw <- function(expiry) {
    
    dt <- expiry/steps
    ti <- (1:steps)*dt # t-grid
    
    Wtilde.H <- expiry^H * Wtilde # rescale for the time-interval
    xi.t <- xi(ti) # xi evaluated on the grid
    v1 <- xi.t * exp(eta * Wtilde.H - 1/2 * eta^2 * ti^(2*H))
    v0 <- rep(xi(0),N)
    v <- rbind(v0,v1[-steps,])  
    logs <- apply(sqrt(v*dt) * Z - v/2*dt,2,sum)
    w <- apply((v + v1)/2 * dt,2,sum)
    res <- array(, dim=c(2,N))
    res[1,] <- logs
    res[2,] <- w
    return(res)
  }
  
  #swt <-t(sapply(expiries,sw))
  m <- length(expiries)
  xwt <- array(, dim=c(2*m,N))
  for (i in 1:m){
    xwt[(2*i-1):(2*i),] <- xw(expiries[i])
  }
  
  return(xwt)
}