library(caret)
library(mlbench)
library(neuralnet)
library(klaR)
library(C50)
library(rpart)
library(doParallel)
library(corrplot)
library(ggplot2)
library(tidyverse)
library(ggthemes)
library(GGally)
library(ggExtra)
library(leaflet)
library(kableExtra)
library(RColorBrewer)
library(plotly)
library(randomForest)
library(e1071)
library(caTools)
library(dplyr)
library(gridExtra)
library(scales)



#Loading dataset
airbnb_data <- read.csv(file.choose(), sep = ",", header = T, stringsAsFactors = F)
#viewing the complete dataset
View(airbnb_data)

#removing columns name, host_id and host_name as we feel this data will be irrelevant to our analysis
airbnb_data <- airbnb_data[,-1:-4]
airbnb_data <- airbnb_data[,-9]

#Since we have decided to keep price as the dependent variable, checking if the class of price is set to integer or not. If it is not, we will have to change its class to integer.
class(airbnb_data$price)

#using apply and is.na function to check if the dataset has any NA values.
apply(is.na(airbnb_data), 2, which)
airbnb_data$reviews_per_month[is.na(airbnb_data$reviews_per_month)] <- 0

#checking for outliers in the price column be checking the extreme values
max(airbnb_data$price,na.rm = TRUE)
min(airbnb_data$price,na.rm = TRUE)

#Exploring the attributes for price = 10000 to determine if the values are garbage or not
nhood1 <- subset(airbnb_data, price == 10000, select = c(neighbourhood_group, neighbourhood, room_type, minimum_nights, number_of_reviews, reviews_per_month))
View(nhood1)

#removing the values that have 0 as price of property as it cannot be a valid value for price
airbnb_data <- airbnb_data[!(airbnb_data$price == 0),]

#visualizing the numeric distribution of price column to check for outliers
boxplot(airbnb_data$price, ylim = c(0,2000))
summary(airbnb_data$price)

#creating categories for price column based on price range and assigning it to price_dummy column
airbnb_data$price_dummy <- airbnb_data$price
airbnb_data$price_dummy <- ifelse((airbnb_data$price>=0 & airbnb_data$price<=2000), 'range1', airbnb_data$price_dummy)
airbnb_data$price_dummy <- ifelse((airbnb_data$price>=2001 & airbnb_data$price<=4000), 'range2', airbnb_data$price_dummy)
airbnb_data$price_dummy <- ifelse((airbnb_data$price>=4001 & airbnb_data$price<=6000), 'range3', airbnb_data$price_dummy)
airbnb_data$price_dummy <- ifelse((airbnb_data$price>=6001 & airbnb_data$price<=8000), 'range4', airbnb_data$price_dummy)
airbnb_data$price_dummy <- ifelse((airbnb_data$price>=8001 & airbnb_data$price<=10000), 'range5', airbnb_data$price_dummy)

#checking the frequency of values in each price range
table(airbnb_data$price_dummy)
airbnb_data <- airbnb_data[,-12]

#deleting all values of price greater than 2000 as 99.8% of data is within price range of 0 and 2000
airbnb_data <- airbnb_data[!(airbnb_data$price>500),]

#creating categories for price based on price range and assigning it to price_category column
airbnb_data$price_category <- airbnb_data$price
airbnb_data$price_category <- ifelse((airbnb_data$price>=0 & airbnb_data$price<=100), 'Very Cheap', airbnb_data$price_category)
airbnb_data$price_category <- ifelse((airbnb_data$price>=101 & airbnb_data$price<=200), 'Cheap', airbnb_data$price_category)
airbnb_data$price_category <- ifelse((airbnb_data$price>=201 & airbnb_data$price<=300), 'Moderate', airbnb_data$price_category)
airbnb_data$price_category <- ifelse((airbnb_data$price>=301 & airbnb_data$price<=400), 'Expensive', airbnb_data$price_category)
airbnb_data$price_category <- ifelse((airbnb_data$price>=401 & airbnb_data$price<=500), 'Very Expensive', airbnb_data$price_category)

str(airbnb_data)
#Converting the format of the neighbourhood_group, neighborhoods, room_type and price_category columns to factor type
airbnb_data$neighbourhood_group = as.factor(airbnb_data$neighbourhood_group)
airbnb_data$neighbourhood = as.factor(airbnb_data$neighbourhood)
airbnb_data$room_type = as.factor(airbnb_data$room_type)
airbnb_data$price_category <- as.factor(airbnb_data$price_category)

#performing linear regression model to find relevant variables for our dependent variable price.
model1 <- lm(price ~ ., data = airbnb_data[,-12])
model1 <- lm(price ~ id + neighbourhood_group + latitude + longitude + room_type + minimum_nights + number_of_reviews + reviews_per_month + calculated_host_listings_count + availability_365, data = airbnb_data)
summary(model1)
#based on our linear regression model, all variables are relevant with respect to the dependent variable.

airbnb_data <- airbnb_data[,-6]

smp_size <- floor(0.75 * nrow(airbnb_data))

## set the seed to make your partition reproducible
set.seed(123)
airbnb_index <- sample(seq_len(nrow(airbnb_data)), size = smp_size)

airbnb_train <- airbnb_data[airbnb_index, ]
airbnb_test <- airbnb_data[-airbnb_index, ]

#Vizuations to explore the data

# Correlation Plot
airbnb_cor <- airbnb_data[, sapply(airbnb_data, is.numeric)]
airbnb_cor <- airbnb_cor[complete.cases(airbnb_cor), ]
correlation_matrix <- cor(airbnb_cor, method = "spearman")
corrplot(correlation_matrix, method = "color")



table(airbnb_data$neighbourhood_group, airbnb_data$room_type)

theme_set(theme_solarized())

summary_data <- function(data) {
    paste('Mean:', round(mean(data, na.rm = T), 2), 
          '\nMedian:', round(median(data, na.rm = T), 2), 
          '\nMax:', round(max(data, na.rm = T), 2), 
          '\nMin:', round(min(data, na.rm = T), 2))
}

summary_data_dollar <- function(data) {
    paste('Mean:', dollar(round(mean(data, na.rm = T), 2), prefix = '$'), 
          '\nMedian:', dollar(round(median(data, na.rm = T), 2), prefix = '$'), 
          '\nMax:', dollar(round(max(data, na.rm = T), 2), prefix = '$'), 
          '\nMin:', dollar(round(min(data, na.rm = T), 2), prefix = '$'))
}

col <- c("orange", "orange1", "orange2", "orange3", "orange4")
col1 <- c("orange2", "orange3", "orange4")

airbnb_data %>% 
    ggplot(aes(neighbourhood_group)) + 
    geom_bar(fill = col, colour = 'black') + 
    geom_text(aes(label = percent(..prop..), y = ..count.., group = 1), stat = 'count', vjust = -0.5) + 
    labs(x = 'price_category')


airbnb_data %>% 
    ggplot(aes(room_type)) + 
    geom_bar(fill = col1, colour = 'black') + 
    geom_text(aes(label = percent(..prop..), y = ..count.., group = 1), stat = 'count', vjust = -0.5) + 
    labs(x = 'Room_Type')

airbnb_data %>% 
    ggplot(aes(minimum_nights)) + 
    geom_histogram(binwidth = 0.1, fill = 'tomato3', colour = 'black') + 
    geom_vline(xintercept = mean(airbnb_data$minimum_nights), linetype = 'dashed', colour = 'turquoise3', size = 1) + 
    scale_x_log10() + labs(x = 'Minumum_Nights', subtitle = summary_data(airbnb_data$minimum_nights))

airbnb_data %>% 
    ggplot(aes(longitude, latitude)) + 
    geom_hex() + 
    scale_fill_gradient(low = 'royalblue', high = 'red', breaks = c(500, 1000)) + 
    labs(x = 'Longitude', y = 'Latitude') + 
    facet_wrap(~ room_type) + 
    theme(legend.position = 'bottom')

table(airbnb_data$neighbourhood_group, airbnb_data$price_category)

p1 <- airbnb_data %>% 
    ggplot(aes(neighbourhood_group, fill = price_category)) + 
    geom_bar(position = 'dodge', colour = 'black') + 
    labs(x = 'Neighbourhood_Group') + 
    guides(fill = F)

temp <- airbnb_data %>% 
    group_by(neighbourhood_group, price_category) %>% 
    count() %>% 
    group_by(neighbourhood_group) %>% 
    mutate(prop = round(n/sum(n), 3))

p2 <- airbnb_data %>% 
    ggplot(aes(neighbourhood_group)) + 
    geom_bar(aes(fill = price_category), colour = 'black', position = 'fill') + 
    geom_text(data = temp, aes(y = prop, label = percent(prop), group = price_category), stat = 'identity', position = position_stack(vjust = 0.5)) + 
    labs(x = 'Neighbourhood_Group', y = 'prop', fill = 'Price_Category')

grid.arrange(p1, p2, ncol = 2)


#We will now use ML models to analyse the data. 

#Decision tree Model
tree_model <- train( price_category~ ., data = airbnb_train, method = "rpart")
predict(tree_model,airbnb_test[,-11])->pred_tree
confusionMatrix(pred_tree,airbnb_test[,11])
saveRDS(tree_model,file="tree_model.rds")

#Decision tree with tuned parameters
tuned_tree<-train(price_category~ ., 
                  data = airbnb_train,
                  method = "rpart", 
                  metric = "Accuracy",
                  #trControl=tr_control,
                  tuneLength=10,
                  control = rpart.control(minsplit =9
                  ),
                  tuneGrid = expand.grid(cp = seq(0, 1, 0.001))
)

predict(tuned_tree,airbnb_test[,-11])->pred_tt 
confusionMatrix(pred_tt,airbnb_test[,11])
saveRDS(tuned_tree,file="tree_model_tuned.rds")
library(rattle)
fancyRpartPlot(tuned_tree$finalModel)



#Random Forest
mtry <- sqrt(ncol(airbnb_train))
metric <- "Accuracy"

set.seed(13)
start3 <- Sys.time()
model_rf1 <- train(price_category ~ ., data = airbnb_train, method = "rf")
predict_rf <- predict(model_rf1, newdata = airbnb_test[,-11])
confusionMatrix(predict_rf1, airbnb_test[,11])

#Random Forest with tuned hyperparameters
set.seed(13)
start1 <- Sys.time()
model_rf <- train(price_category ~ ., data = airbnb_train, method = "rf",
                  trControl = trainControl(method = "repeatedcv", number = 5, repeats = 3, search = "grid"),
                  tuneGrid = expand.grid(.mtry = mtry), metric = metric, tuneLength = 15)
Sys.time() - start1
model_rf
predict_rf <- predict(model_rf, newdata = airbnb_test[,-11])
confusionMatrix(predict_rf, airbnb_test[,11])

saveRDS(model_rf,file = "rf_model_tuned.rds")


#SVM 
model_svm <- train(price_class ~ ., data = airbnb_train, method = "svmLinear",tuneGrid = expand.grid(C = seq(1, 10, 3)),trControl = trainControl(method = "cv", number = 3))
svmpred<-predict(svm_model,test[,-11])
confusionMatrix(svmpred,test[,11])
saveRDS(model_svm,file = "svm_model_tuned.rds")


model_comparison <- resamples(list(SVM = svm,DecisionTree=tuned_tree,RandomForrest=rf))
summary(model_comparison)




