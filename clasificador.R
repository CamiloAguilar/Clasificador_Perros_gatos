plotcd <- function(v){
  x <- matrix(v,64,64)
  image(1:65,1:65,t(apply(x,2,rev)),asp=1,xaxt="n",yaxt="n",
        col=grey((0:255)/255),ann=FALSE,bty="n")
}


### Ejemplo de uso.
load("gatosperros.RData")
plotcd(dm[sample(1:198,1),])

### Partición Conjuntos Entrenamiento y Prueba.
set.seed(1234)
ind.gatostest <- sample(1:99, 18, replace = FALSE)
gatos.test <- dm[ind.gatostest, ]
gatos.train <- dm[1:99, ][-ind.gatostest, ]

ind.perrostest <- sample(1:99, 18, replace = FALSE)
perros.test <- dm[100:198, ][ind.perrostest, ]
perros.train <- dm[100:198, ][-ind.perrostest, ]

## Conjuntos completos de entrenamiento y prueba
train <- rbind(gatos.train, perros.train)
y_train <- c(rep(1, 81), rep(0, 81))

test <- rbind(gatos.test, perros.test)
y_test <- c(rep(1, 18), rep(0, 18))

### Aplicación de componentes principales
pc <- prcomp(train, scale = T)

## varianza ganada
var_ganada <- cumsum(pc$sdev^2)/sum(pc$sdev^2)
plot(var_ganada)
abline(h=0.5, col="brown2")
abline(h=0.8, col="brown3")
abline(h=0.9, col="brown4")
abline(h=0.95, col="green4")
abline(h=0.99, col="green2")

which(var_ganada>0.5)[1] # 4 componentes
which(var_ganada>0.8)[1] # 24 componentes
which(var_ganada>0.9)[1] # 51 componentes
which(var_ganada>0.95)[1] # 80 componentes
which(var_ganada>0.99)[1] # 131 componentes


#*****************************************************************
## REDUCIR PERROS Y GATOS A 2 DIMENSIONES Y APLIQUE K-VECINOS
#*****************************************************************
dos_dim <- pc$x[, 1:2]
dos_dim <- as.data.frame(dos_dim)
y_lab <- ifelse(y_train==1, "gato", "perro")

library(ggplot2)
ggplot(as.data.frame(dos_dim), aes(PC1, PC2)) + geom_point(aes(colour=y_lab))

## Límites de simulación
lims <- c(-100, 100,#Estatura, eje X
          -60, 60) #Peso, eje Y

nuevopunto <- function(lims=lims){
     a <- runif(1,lims[1],lims[2])
     b <- runif(1,lims[3],lims[4])
     return(c(a, b))
}

#*****************
## 3-Vecinos
#*****************
## Gráfico
plot(dos_dim$PC1, dos_dim$PC2, type = "n", xlab="PC1", ylab="PC2",asp=1,
     xlim=lims[1:2],ylim=lims[3:4])

## K-vecinos a considerar
k <- 3

## Algoritmo de clasificación
for(iter in 1:20000){
     x <- nuevopunto(lims)
     diss <- dist(rbind(x,dos_dim[,1:2]))[1:length(y_lab)]
     knn <- head(order(diss),k)
     etiqueta <- names(sort(table(y_lab[knn]),decreasing = TRUE))[1]
     clase <- as.numeric(etiqueta=="gato")
     for(inn in 1:k){
          if(y_lab[knn[inn]]==etiqueta){
               points(x[1],x[2],col=c("gray","black")[clase+1],pch=20)
          }
     }
}

## Datos de entrenamiento
cols2 <- rep("red", length(y_lab))
cols2[which(y_lab=="gato")] <- "green"
points(dos_dim$PC1, dos_dim$PC2, pch=16, col=cols2)




#*****************
## 5-Vecinos
#*****************
## Gráfico
plot(dos_dim$PC1, dos_dim$PC2, type = "n", xlab="PC1", ylab="PC2",asp=1,
     xlim=lims[1:2],ylim=lims[3:4])

## K-vecinos a considerar
k <- 5

## Algoritmo de clasificación
for(iter in 1:20000){
     x <- nuevopunto(lims)
     diss <- dist(rbind(x,dos_dim[,1:2]))[1:length(y_lab)]
     knn <- head(order(diss),k)
     etiqueta <- names(sort(table(y_lab[knn]),decreasing = TRUE))[1]
     clase <- as.numeric(etiqueta=="gato")
     for(inn in 1:k){
          if(y_lab[knn[inn]]==etiqueta){
               points(x[1],x[2],col=c("gray","black")[clase+1],pch=20)
          }
     }
}

## Datos de entrenamiento
cols2 <- rep("red", length(y_lab))
cols2[which(y_lab=="gato")] <- "green"
points(dos_dim$PC1, dos_dim$PC2, pch=16, col=cols2)


#*************************************************
## PERCEPTRÓN PARA CLASIFICAR PERROS Y GATOS ####
#*************************************************

## Función perceptrón
perceptron <- function(x, y, num_pc=24, iter) {
     # Toma las componentes dadas por num_pc
     x <- x[, 1:num_pc]
     
     # pesos iniciales
     w <- rep(0, dim(x)[2] + 1)
     # Vector de errores
     err <- rep(0, iter)
     
     # Bucle para completar todas las iteraciones
     for (i in 1:iter) {
          # Bucle que recorre todos los individuos
          for (j in 1:length(y)) {
               
               # Función de paso
               z <- sum(w[2:length(w)] * as.numeric(x[j, ])) + w[1]
               if(z > 0) {
                    ypred <- 1
               } else {
                    ypred <- 0
               }
               
               # Regla de aprendizaje del perceptrón simple
               w <- w + 1 * (y[j] - ypred) * c(1, as.numeric(x[j, ]))
               
               # Función que calcula el error de cada iteración
               if ((y[j] - ypred) != 0.0) {
                    err[i] <- err[i] + 1
               }
          }
          
          # W bolsillo
          if(i == 1){
               w.bolsillo <- w
          } else{
               if(err[i] < err[i-1]){
                    w.bolsillo <- w
               }
          }
     }
     resultados <- list(w.bolsillo=w.bolsillo, errores = err)
     return(resultados)
}


## Se calcula el PERCEPTRÓN para todas las componentes principales, desde 2 hasta 162.
require(progress)

## Total de componentes
max_pca <- dim(pc$x)[1]

## Matriz pca
pca_train <- pc$x

## iteraciones para encontrar mejor perceptrón
total_pca <- list()
Num_CP <- NULL
error_train <- NULL
pb <- progress_bar$new(total = length(1:max_pca))
for(i in 2:max_pca){
     per_res <- perceptron(x = pca_train, y = y_train, num_pc = i, iter=1000)
     error_train <- c(error_train, min(per_res$errores))
     Num_CP <- c(Num_CP, i) 
     total_pca[[i-1]] <- per_res
     names(total_pca)[i-1] <- paste0("PC_", i)
     
     pb$tick()
}


## Genera tabla de resultados
## La siguiente tabla muestra el error bolsillo para cada conjunto de componentes principales
tabla_res <- data.frame(Num_CP=Num_CP, error_train=error_train/length(y_train))
head(tabla_res)

## Gráfico de los resultados
## Se muestra el número de componentes usadas vs error bolsillo resultante 
require(ggplot2)
ggplot(data=tabla_res, aes(x=Num_CP, y=error_train, colour="train")) + geom_line() +
     ggtitle("Error en el perceptrón vs CP") + xlab("Componentes principales") +
     ylab("Error porcentual")

#


