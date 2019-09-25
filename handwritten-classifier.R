setwd("/Users/chaeyoonlee/Classifying-handwritten/dataset")
list.files()

#install.packages("jpeg")
library(jpeg)

### Pixelize images
pixel = matrix(0, ncol=256)
for(i in 1:length(list.files())){
  temp = readJPEG(list.files()[i])
  temp_bw = (temp[,,1]+temp[,,2]+temp[,,3])/3
  temp_nu = matrix(as.numeric(temp_bw < 0.7), ncol=256)
  pixel = rbind(pixel, temp_nu)
}
pixel = pixel[-1,]
dim(pixel)

### Check inaccurate data
par(mfrow=c(4, 4))
for(i in 1:16){
  check = matrix(drop(pixel[i,]), ncol=16)
  x = y = seq(0, 1, length = nrow(check))
  image(x, y, check, col = grey(seq(0, 1, length=256)))
}

### Adjust pixel values for inaccurate data
par(mfrow=c(1, 1))
temp = readJPEG(list.files()[15])
temp_bw = (temp[,,1]+temp[,,2]+temp[,,3])/3
temp_nu = matrix(as.numeric(temp_bw < 0.6), ncol=16)
image(x, y, temp_nu, col = grey(seq(0, 1, length=256)))
plot.new()

pixel[15,] =  matrix(as.numeric(temp_bw < 0.6),ncol=256)

### Deleting unadjustable data
pixel_result = cbind(rep(0:9, 7), pixel)
pixel_result = pixel_result[-c(7,30,40,43,49,54,56,57,59,60,64),]
write.csv(pixel_result, "pixel_result.csv")

### Analyze data 
setwd("/Users/chaeyoonlee/Classifying-handwritten")
list.files()
pixel1 = read.csv("pixel_result.csv")
pixel2 = read.csv("pixel_result2.csv")

### factor
for(i in 2:dim(pixel1)[2])pixel1[,i] = factor(pixel1[,i],levels = c(0, 1))
for(i in 2:dim(pixel2)[2])pixel2[,i] = factor(pixel2[,i],levels = c(0, 1))
pixel1[,1] = as.factor(pixel1[,1])
pixel2[,1] = as.factor(pixel2[,1])
pixel = rbind(pixel1, pixel2)

### 1. SVM
#install.packages("e1071")
library(e1071)
svm_acc = c()
for(i in 1:100){
  id = sample(1:nrow(pixel), nrow(pixel)*0.7)
  train = pixel[id,]
  test  = pixel[-id,]
  svm = svm(num~., data=train, kernal="linear")
  svm.pred = predict(svm, newdata=test[,-1], type="class")
  svm_acc[i] = sum(diag(table(svm.pred, test[,1])))/length(test[,1])
}
svm_acc
mean(svm_acc)

### 2. Random Forest
ranfo_acc = c()
for(i in 1:100){
  id = sample(1:nrow(pixel), nrow(pixel)*0.7)
  train = pixel[id,]
  test  = pixel[-id,]
  ranfo.model = randomForest::randomForest(num~., data=train, ntree=200, mtry=4)
  ranfo.pred = predict(ranfo.model, newdata=test[,-1])
  ranfo_acc[i] = sum(diag(table(ranfo.pred, test[,1])))/length(test[,1])
}
ranfo_acc
mean(ranfo_acc)

### 3. Boosting
#install.packages(???adabag???)
library(adabag)
boost_acc = c()
for(i in 1:10){
  id = sample(1:nrow(pixel), nrow(pixel)*0.7)
  train = pixel[id,]
  test  = pixel[-id,]
  boost.model = boosting(num~., data=train, mfinal=100)
  boost.pred  = predict(boost.model, newdata=test[,-1], type="class")
  boost_acc[i] = sum(diag(table(boost.pred$class, test[,1])))/length(test[,1])
}
boost_acc
mean(boost_acc)

### 4. Tree
rpart_acc = c()
for(i in 1:100){
  id = sample(1:nrow(pixel), nrow(pixel)*0.7)
  train = pixel[id,]
  test  = pixel[-id,]
  rpart.model = rpart(num~., data=train)
  rpart.pred = predict(rpart.model, newdata=test[,-1], type="class")
  rpart_acc[i] = sum(diag(table(rpart.pred, test[,1])))/length(test[,1])
}
rpart_acc
mean(rpart_acc)

### 5. LASSO
#install.packages(???glmnet???)
library(glmnet)
lasso_acc = c()
for(i in 1:10){
  id = sample(1:nrow(pixel), nrow(pixel)*0.7)
  train = pixel[id,]
  test  = pixel[-id,]
  lasso.model = cv.glmnet(x=data.matrix(train[,-1]), y=data.matrix(train[,1]),
                          family="multinomial", alpha=1, nfold=10)
  lasso.pred = predict(lasso.model, newx=data.matrix(test[,-1]), 
                       lambda=lasso.model$lambda.min, type="class") 
  lasso_acc[i] = sum(diag(table(lasso.pred, test[,1])))/length(test[,1])
}
lasso_acc
mean(lasso_acc)

### 6. Bagging
#install.packages(???ipred???)
library(ipred)
bagging_acc = c()
for(i in 1:100){
  id = sample(1:nrow(pixel), (pixel)*0.7)
  train = pixel[id,]
  test  = pixel[-id,]
  bagging.model = bagging(num~., data=train)
  bagging.pred  = predict(bagging.model, newdata=test[,-1])
  bagging_acc[i] = sum(diag(table(bagging.pred, test[,1])))/length(test[,1])
}
bagging_acc
mean(bagging_acc)
