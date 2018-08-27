##########################################################
#              TAREA 1 - Delia Del Aguila
##########################################################


##########################################################
# - Corre el ejemplo con otra muestra 
# - Reporta tus resultados de error de entrenamiento 
#   y error de prueba para los tres métodos.
##########################################################

models_comparison <- function(train_seed, test_seed, x, f)
{
  
  # Training Process  
  set.seed(train_seed)
  error <- rnorm(length(x), 0, 500)
  y <- f(x) + error
  train_data <- data.frame(x=x, y=y)
  
  curva_1 <- geom_smooth(data=train_data,
                         method = "loess", se=FALSE, color="gray", span=1, size=1.1)
  curva_2 <- geom_smooth(data=train_data,
                         method = "loess", se=FALSE, color="red", span=0.5, size=1.1)
  curva_3 <- geom_smooth(data=train_data,
                         method = "lm", se=FALSE, color="blue", size=1.1)
  models_graph <<- ggplot(train_data, aes(x=x, y=y)) + geom_point() +
    ggtitle("Models comparison over Train Data") +
    curva_1 + curva_2 + curva_3 
  
  mod_rojo <- loess(y ~ x, data = train_data, span=0.3)
  mod_gris <- loess(y ~ x, data = train_data, span=1)
  mod_recta <- lm(y ~ x, data = train_data)
  df_mods <- data_frame(nombre = c('recta', 'rojo','gris'))
  df_mods$modelo <- list(mod_recta, mod_rojo, mod_gris)
  
  error_f <- function(df, mod){
    function(mod){
      preds <- predict(mod, newdata = df)
      round(sqrt(mean((preds - df$y) ^ 2)))
    }
  }
  err_train <- error_f(train_data)
  
  df_mods <- df_mods %>% 
    mutate(error_train = map_dbl(modelo, err_train))
  df_mods
  
  #Testing Process
  set.seed(test_seed)
  x_0 <- sample(0:13, 100, replace = T)
  error <- rnorm(length(x_0), 0, 500)
  y_0 <- f(x_0) + error
  test_data <- data_frame(x = x_0, y = y_0)
  
  err_test <- error_f(test_data)
  df_mods <- df_mods %>% 
    mutate(error_test = map_dbl(modelo, err_test))
  df_mods
}

# Variables to train and to test models
train_seed <- 280572
test_seed <- 218052272

#Setting x and y
x <- c(1,7,10,0,0,5,9,13,2,4,17,18,1,2)
f <- function(x){
  ifelse(x < 10, 1000*sqrt(x), 1000*sqrt(10))
}

models_comparison(train_seed, test_seed, x, f)
models_graph


##########################################################
#  Evalúa los tres métodos comparando estos valores 
#  para un número grande de distintas simulaciones de los datos de entrenamiento
##########################################################

models_multi_comparison <- function(experiments) {

  err_train <- 0
  err_test <- 0

  for(i in 1:experiments){
    comparison_results <- models_comparison(train_seed+i, test_seed, x, f)
    err_train <- err_train + comparison_results[,3]
    err_test <- err_test + comparison_results[,4]
  }

  comparison_results[,3] <- err_train/experiments
 
  comparison_results

}

models_multi_comparison(100)

