##########################################################
#              TAREA 2 - Delia Del Aguila
##########################################################

##########################################################
# Para este ejemplo usaremos los datos de https://archive.ics.uci.edu/ml/machine-learning-databases/housing/. 
# El objetivo es predecir el valor mediano de las viviendas en ?reas del censo de Estados Unidos, 
# utilizando variables relacionadas con criminalidad, ambiente, tipo de viviendas, etc.
#   1. Separa la muestra en dos partes: unos 400 para entrenamiento y el resto para prueba.
#   2. Describe las variables en la muestra de prueba (rango, media, mediana, por ejemplo).
#   3. Construye un modelo lineal para predecir MEDV en t?rminos de las otras variables. 
#       Utiliza descenso en gradiente para estimar los coeficientes con los predictores estandarizados. 
#       Verifica tus resultados con la funci?n lm.
#   4. Eval?a el error de entrenamiento \(\overline{err}\) de tu modelo, 
#       y eval?a despu?s la estimaci?n del error de predicci?n \(\hat{Err}\) con la muestra de prueba. 
#       Utiliza la ra?z del la media de los errores al cuadrado.
##########################################################

# Carga de datos y Separaci?n de muestra
install.packages("kknn")
library(data.table)
library(dplyr)
library(tidyverse)

housing <- fread('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
head(housing)
colnames(housing) <- c("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV")

#

set.seed(213)
housing$muestra_unif <- runif(nrow(housing), 0, 1)
housing_entrena <- filter(housing, muestra_unif <= 400/nrow(housing))
housing_prueba <- filter(housing, muestra_unif > 400/nrow(housing))
nrow(housing_entrena)
nrow(housing_prueba)


# Descripci?n de Variables muestra de prueba
library(Hmisc)
summary(housing_prueba)
describe(housing_prueba)

#Estandarizaci?n
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
rss_calc <- function(x, y){
  # x es un data.frame o matrix con entradas
  # y es la respuesta
  rss_fun <- function(beta){
    # esta funcion debe devolver rss
    y_hat <- as.matrix(cbind(1,x)) %*% beta
    e <- y - y_hat
    rss <- 0.5*sum(e^2)
    rss
  }
  rss_fun
}


grad_calc <- function(x, y){
  # devuelve una funci??n que calcula el gradiente para 
  # par??metros beta   
  # x es un data.frame o matrix con entradas
  # y es la respuesta
  grad_fun <- function(beta){
    f_beta <- as.matrix(cbind(1, x)) %*% beta
    e <- y - f_beta
    gradiente <- -apply(t(cbind(1,x)) %*% e, 1, sum)
    names(gradiente)[1] <- 'Intercept'
    gradiente
  }
  grad_fun
}


descenso <- function(n, z_0, eta, h_grad){
  # esta funci??n calcula n iteraciones de descenso en gradiente 
  z <- matrix(0,n, length(z_0))
  z[1, ] <- z_0
  for(i in 1:(n-1)){
    z[i+1,] <- z[i,] - eta*h_grad(z[i,])
  }
  z
}


set.seed(123)
housing$unif <- runif(nrow(housing), 0, 1)
housing <- arrange(housing, unif)
housing$id <- 1:nrow(housing)
dat_e <- housing[1:400,]
dat_p <- housing[400:nrow(housing),]
dim(dat_e)
dim(dat_p)

dat_norm <- housing %>% select(-id, -MEDV, -unif) %>%
  gather(variable, valor, CRIM:LSTAT) %>%
  group_by(variable) %>% summarise(m = mean(valor), s = sd(valor))
dat_norm

normalizar <- function(datos, dat_norm){
  datos_salida <- datos %>% select(-unif) %>%
    gather (variable, valor, CRIM:LSTAT) %>%
    left_join(dat_norm) %>%
    mutate(valor_s = (valor - m)/s) %>%
    dplyr::select(id, MEDV, variable, valor_s) %>%
    spread(variable, valor_s)
}
dat_e_norm <- normalizar(dat_e, dat_norm)
dat_p_norm <- normalizar(dat_p, dat_norm)

x_ent <- dat_e_norm %>% select(-id, -MEDV)
y_ent <- dat_e_norm$MEDV
rss <- rss_calc(x_ent, y_ent)
grad <- grad_calc(x_ent, y_ent) 

iteraciones <- descenso(1000, rep(0, ncol(x_ent)+1), 0.0001, grad)
rss_iteraciones <- apply(iteraciones, 1, rss)
plot(rss_iteraciones[500:1000])

beta <- iteraciones[1000,]
dat_coef <- data_frame(variable = c('Intercept',colnames(x_ent)), beta = beta)
quantile(y_ent)
dat_coef %>% mutate(beta = round(beta, 2)) %>% arrange(desc(abs(beta)))

lm(MEDV ~ ., data= dat_e_norm %>% select(-id))

calcular_preds <- function(x, beta){
  cbind(1, as.matrix(x))%*%beta
}
x_pr <- dat_p_norm %>% select(-id, -MEDV)
y_pr <- dat_p_norm$MEDV
preds <- calcular_preds(x_pr, beta)
qplot(x = preds, y = y_pr) + geom_abline(intercept = 0, slope = 1)
error_prueba <- mean((y_pr-preds)^2)
sqrt(error_prueba)

mean(abs(y_pr-preds))

# K VECINOS

library(kknn)
error_pred_vmc <- function(dat_ent, dat_prueba){
  salida <- function(k){
    vmc <- kknn(MEDV ~ ., train = dat_ent,  k = k,
                test = dat_prueba, kernel = 'rectangular')
    sqrt(mean((predict(vmc) - dat_prueba$MEDV)^2))
  }
  salida
}
calc_error <- error_pred_vmc(dat_e_norm, dat_p_norm)
dat_vmc <- data_frame(k = c(1,5,seq(10,200,10)))
dat_vmc <- dat_vmc %>% rowwise %>% mutate(error_prueba = calc_error(k))
dat_vmc
ggplot(dat_vmc, aes(x = k, y = error_prueba)) + geom_line() + geom_point() +
  geom_abline(intercept = sqrt(error_prueba), slope=0, colour='red') +
  annotate('text', x = 150, y = 4.6, label = 'modelo lineal', colour = 'red')
