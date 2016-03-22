rawData <- read.csv("data/train.csv")
train <- rawData[1:(nrow(rawData) * 0.7), ]
crossValid <- rawData[(nrow(train) + 1):(nrow(rawData)), ]
test <- read.csv("data/test.csv")

cleaningData <- function(original) {
    if ("Survived" %in% colnames(original)) {
        usefulCol <- c("Survived", "Pclass", "Sex", "Age", "SibSp","Parch","Fare","Embarked")
    } else {
        usefulCol <- c("Pclass", "Sex", "Age", "SibSp","Parch","Fare","Embarked")
    }

    data.matrix(fixNa(original)[, usefulCol])
    fixNa(original)[, usefulCol]
}

fixNa <- function(data) {
    temp <- data
    temp$Age[is.na(temp$Age)] <- mean(temp$Age, na.rm=T)
    temp$Fare[is.na(temp$Fare)] <- mean(temp$Fare, na.rm=T)
    temp$Pclass[is.na(temp$Pclass)] <- mean(temp$Pclass, na.rm=T)
    temp$SibSp[is.na(temp$SibSp)] <- mean(temp$SibSp, na.rm=T)
    temp$SibSp[is.na(temp$Parch)] <- mean(temp$Parch, na.rm=T)
    temp
}

sigmoid <- function(z) {
    1.0 / (1.0 + exp(-z));
}

costFunc <- function(theta, X, y) {
    sum(cbind(y, 1 - y) * cbind(log(sigmoid(X %*% theta)), log(1 - sigmoid(X %*% theta)))) / -nrow(X)
}

gradient <- function(theta, X, y) {
    (t(X) %*% (sigmoid(X %*% theta) - y)) / nrow(X);
}

learning <- function(train, alpha, iterNum) {
    cleanTrain <- cleaningData(train)
    X <- cleanTrain[, -1]
    y <- cleanTrain[, 1]
    theta <- cbind(rep(c(0), ncol(X)))
    J_Hist <- rep(c(0), iterNum)
    converge <- 0
    for (i in 1:iterNum) {
        theta <- theta - alpha * gradient(theta, X, y)
        J_Hist[i] <- costFunc(theta, X, y)
        # if (i > 2 && abs(J_Hist[i - 1] - J_Hist[i]) < 0.0001) {
        #     converge <- converge + 1
        # }
        # 
        # if (converge > 10) {
        #     break
        # }
    }
    theta
}

builtInLearning <- function(train) {
    glm(Survived~., family = binomial(link='logit'),data=train)
}

crossValidate <- function(validateSet, theta) {
    cleanedData <- cleaningData(validateSet)
    X <- cleanedData[, -1]
    y <- cleanedData[, 1]
    m <- nrow(X)
    p <- rep(c(0), m)
    correct <- 0
    t_neg <- 0
    t_pos <- 0
    f_neg <- 0
    f_pos <- 0
    for (i in 1:m) {
        rate <- sigmoid(X[i,] %*% theta)
        
        if (rate >= 0.5 && y[i] == 1) {
            t_pos <- t_pos + 1
        } else if (rate < 0.5 && y[i] == 1) {
            t_neg <- t_neg + 1
        } else if (rate >= 0.5 && y[i] == 0) {
            f_neg <- f_neg + 1
        } else if (rate < 0.5 && y[i] == 0) {
            f_pos <- f_pos + 1
        }  

    }
    
    precision <- (t_pos / (t_pos + f_pos))
    recall <- (t_pos / (t_pos + f_neg))
    f1 <- (precision * recall) / (precision + recall)
    acc <- (t_pos + t_neg) / (t_pos + t_neg + f_pos + f_neg)
    
    list(precision = precision, recall = recall, f1 = f1, acc = acc)
}

myPredict <- function(test, theta) {
    X <- cleaningData(test)
    m <- nrow(X)
    p <- rep(c(0), m)

    for (i in 1:m) {
        rate <- sigmoid(X[i,] %*% theta)

        if (rate >= 0.5) {
            p[i] <- 1
        } else {
            p[i] <- 0
        }

    }

    data.frame(PassengerId=test[,"PassengerId"], Survived=p)
}