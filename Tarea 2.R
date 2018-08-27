##########################################################
#              TAREA 2 - Delia Del Aguila
##########################################################

##########################################################
# Para este ejemplo usaremos los datos de https://archive.ics.uci.edu/ml/machine-learning-databases/housing/. 
# El objetivo es predecir el valor mediano de las viviendas en áreas del censo de Estados Unidos, 
# utilizando variables relacionadas con criminalidad, ambiente, tipo de viviendas, etc.
#   1. Separa la muestra en dos partes: unos 400 para entrenamiento y el resto para prueba.
#   2. Describe las variables en la muestra de prueba (rango, media, mediana, por ejemplo).
#   3. Construye un modelo lineal para predecir MEDV en términos de las otras variables. 
#       Utiliza descenso en gradiente para estimar los coeficientes con los predictores estandarizados. 
#       Verifica tus resultados con la función lm.
#   4. Evalúa el error de entrenamiento \(\overline{err}\) de tu modelo, 
#       y evalúa después la estimación del error de predicción \(\hat{Err}\) con la muestra de prueba. 
#       Utiliza la raíz del la media de los errores al cuadrado.
##########################################################

# Carga de datos y Separación de muestra
library(data.table)
housing <- fread('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
head(housing)
colnames(housing) <- c("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV")

set.seed(213)
housing$muestra_unif <- runif(nrow(housing), 0, 1)
housing_entrena <- filter(housing, muestra_unif <= 400/nrow(housing))
housing_prueba <- filter(housing, muestra_unif > 400/nrow(housing))
nrow(housing_entrena)
nrow(housing_prueba)


# Descripción de Variables muestra de prueba
library(Hmisc)
summary(housing_prueba)
describe(housing_prueba)

#Estandarización
x1_s = (housing_entrena$CRIM - mean(housing_entrena$CRIM))/sd(housing_entrena$CRIM)
x2_s = (housing_entrena$ZN - mean(housing_entrena$ZN))/sd(housing_entrena$ZN)
x3_s = (housing_entrena$INDUS - mean(housing_entrena$INDUS))/sd(housing_entrena$INDUS)
x4_s = (housing_entrena$CHAS - mean(housing_entrena$CHAS))/sd(housing_entrena$CHAS)
x5_s = (housing_entrena$NOX - mean(housing_entrena$NOX))/sd(housing_entrena$NOX)
x6_s = (housing_entrena$RM - mean(housing_entrena$RM))/sd(housing_entrena$RM)
x7_s = (housing_entrena$AGE - mean(housing_entrena$AGE))/sd(housing_entrena$AGE)
x8_s = (housing_entrena$DIS - mean(housing_entrena$DIS))/sd(housing_entrena$DIS)
x9_s = (housing_entrena$RAD - mean(housing_entrena$RAD))/sd(housing_entrena$RAD)
x10_s = (housing_entrena$TAX - mean(housing_entrena$TAX))/sd(housing_entrena$TAX)
x11_s = (housing_entrena$PTRATIO - mean(housing_entrena$PTRATIO))/sd(housing_entrena$PTRATIO)
x12_s = (housing_entrena$B - mean(housing_entrena$B))/sd(housing_entrena$B)
x13_s = (housing_entrena$LSTAT - mean(housing_entrena$LSTAT))/sd(housing_entrena$LSTAT)
y_s = (housing_entrena$MEDV - mean(housing_entrena$MEDV))/sd(housing_entrena$MEDV)

dat <- data_frame(x1_s, x2_s, x3_s, x4_s, x5_s, x6_s, x7_s, x8_s, x9_s, x10_s, x11_s, x12_s, x13_s, y_s)


# Modelo Lineal
grad_calc <- function(x_ent, y_ent){
  salida_grad <- function(beta){
    f_beta <- as.matrix(cbind(1, x_ent)) %*% beta
    e <- y_ent - f_beta
    grad_out <- -as.numeric(t(cbind(1, x_ent)) %*% e)
    names(grad_out) <- c('Intercept', colnames(x_ent))
    grad_out
  }
  salida_grad
}
grad_housing <- grad_calc(housing_entrena[,1:13], housing_entrena$MEDV)
grad_housing(c(0,1,1,1,1,1,1,1,1,1,1,1,1,1))


descenso <- function(n, z_0, eta, h_deriv){
  z <- matrix(0,n, length(z_0))
  z[1, ] <- z_0
  for(i in 1:(n-1)){
    z[i+1, ] <- z[i, ] - eta * h_deriv(z[i, ])
  }
  z
}

rss_calc <- function(datos){
  # esta función recibe los datos (x,y) y devuelve
  # una función f(betas) que calcula rss
  y <- datos$lpsa
  x <- datos$lcavol
  fun_out <- function(beta){
    y_hat <- beta[1] + beta[2]*x
    e <- (y - y_hat)
    rss <- sum(e^2)
    0.5*rss
  }
  fun_out
}

rss_housing <- rss_calc(housing_entrena)

iteraciones <- descenso(100, c(0,1,1,1,1,1,1,1,1,1,1,1,1,1), 0.0005, grad_housing)
iteraciones
apply(iteraciones, 1, rss_housing)

grad_housing(iteraciones[100, ])




grad_sin_norm <- grad_calc(dat[, 1:13, drop = FALSE], dat$y)
iteraciones <- descenso(10, c(0, 0.25, 0.25, 0.25, 0.25, 0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25), 0.0001, grad_sin_norm)

