basePic <- function() {
  data <- read.csv("data", header=F)
  
  km <- kmeans(data,6)
  plot(data[,1],data[,2],col=km$cluster)
}

plotPs <- function() {
  ps <- read.csv("data", heade=F)
  
  points(as.matrix(ps), col="red",pch=20)
  
}