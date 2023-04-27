# House-Price-Prediction

### A. Problem Statement

Thousands of houses are sold everyday. There are some questions every buyer asks himself. What is the actual price of this house? Am I paying a fair price? Till when can I get returns on my investment? What will be the price of my house after 5 years?


### B. Best Possible Solutions

a.Housing Expert
b.Intuition About House
c.Using Machine Learning

### C. Introduction About Project

House Price prediction are very stressful work as we have to consider different things while buying a house like the structure and the rooms kitchen parking space and gardens. People don’t know about the factor which influence the house price. But by using the Machine learning we can easily find the house which is to be prefect for us and helps to predict the price accurately.

### D. Tools and Libraries

### Tools

a.Python
b.Jupyter Notebook
c. Google Colab
d. GitHub

### E. Libraries

a.Pandas
b.Scikit Learn
c.Numpy
d.Matpoltlib

### F. Data Collection

https://www.kaggle.com/datasets/ruiqurm/lianjia    (Dataset used to build model)

For this project we used the data that is available on kaggle.There are 26 columns and 318851 Rows. These are the major point about the data set.<br>
url: the url which fetches the data<br>
id: the id of transaction<br>
Lng: and Lat coordinates, using the BD09 protocol.<br>
Cid: community id<br>
tradeTime: the time of transaction<br>
DOM: active days on market<br>
followers: the number of people follow the transaction.<br>
totalPrice: the total price<br>
price: the average price by square<br>
square: the square of house<br>
livingRoom: the number of living room<br>
drawingRoom: the number of drawing room<br>
kitchen: the number of kitchen<br>
bathroom the number of bathroom<br>
floor: the height of the house. I will turn the Chinese characters to English in the next version.<br>
buildingType: including tower( 1 ) , bungalow( 2 )，combination of plate and tower( 3 ), plate( 4 ).<br>
constructionTime: the time of construction<br>
renovationCondition: including other( 1 ), rough( 2 ),Simplicity( 3 ), hardcover( 4 )<br>
buildingStructure: including unknow( 1 ), mixed( 2 ), brick and wood( 3 ), brick and concrete( 4 ),steel( 5 ) and steel-concrete composite ( 6 ).<br>
ladderRatio: the proportion between number of residents on the same floor and number of elevator of ladder. It describes how many ladders a resident have on average.<br>
elevator: have ( 1 ) or not have elevator( 0 )<br>
fiveYearsProperty: if the owner have the property for less than 5 years.<br>

### G. Generic Flow Of Project
![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/18.png)

### H. EDA
![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/1.png)


#### A.Data Cleaning
we have 26 columns ,from these we don't want some column(i.e. url,id,cid) then we will perform data cleaning wich involve following steps. our target variable is totalPrice<br>
a. Impute/Remove missing values or Null values (NaN)<br>
b. Remove unnecessary and corrupted data.<br>
c. Date/Text parsing if required.

![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/2.png)<br>
we handle NAN value using appropriate solutions.

![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/3.png)<br>
DOM Column have more than 50% value are missing it's better to delete that column


![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/4.png)<br>
some column have unique character. we solve these problem using split method and create seprate column for unique character.<br>

We also have a categorical data we handle such kind of data using dummies variable concept. following are the columns which have categorical data.<br>
a. renovationCondition<br>
b. buildingStructure<br>
c. buildingType<br>
d. district<br>
e. elevator<br>
f. floor_type

![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/5.png)<br>
Summary of the Heat-Map<br>
a. totalPrice is highly corellated with community average,square,bathroom,livingroom and Trde Time.<br>
b. totalprice is highly negative corellated with ladderRatio,lat and lng.

![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/6.png)<br>
Summary of the Density Plot<br>
a. most of the output features is lies between 0-2500

![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/7.png)<br>
Summary of Scatterplot<br>
a. Most of the House Followers 0-400.

![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/8.png)<br>
Summery of Scatterplot with respect to renovationCondition<br>
a. most of the expensive houses have HardCover as a renovation condition

![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/9.png)<br>
Summary of lineplot<br>
a. Most of the peoples average are lies in 12500-150000 ...

#### B. Feature Engineering
we found outlier in our data ..

![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/10.png)<br>
from the above figure we can notice that we have an outlier present in our dataset.<br>
for outlier we can use IQR method and after using IQR method.Now, our data looks fine.
![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/11.png)



using the feature engineering we got out top 30 features with respect to totalPrice .
![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/12.png)

So,these are the top 20 features for our model<br>
a. tradeTime<br>
b. CommunityAverage<br>
c. square<br>
d. livingRoom<br>
e. bathRoom<br>
f. drawingRoom<br>
g. renovationCondition<br>
h. buildingStructure<br>
i. elevator<br>
j. constructionTime<br>
k. Followers

#### C. Data Normalization
Normalization (min-max Normalization)<br>
In this approach we scale down the feature in between 0 to 1

we have numerical column where we can apply min-max Normalization.<br>
![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/13.png)

### 5. Choosing Best ML Model
List of the model that we can use for our problem<br>
a. LinearRegression model<br>
b. KNN Model<br>
c. Decesion Tree<br>
d. Random Forest


![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/linear.jpeg)<br>
Using the linearRegression we got only 87 % accuracy.



### KNN
![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/knn.jpeg)<br>


### Decision Tree
![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/decision.jpeg)<br>

### Random Forest
![](https://github.com/Yash-Parwani/House-Price-Prediction/blob/main/new/random.jpeg)<br>

Using the Random Forest we got 98 % accuracy on train data and 89 % on test data .so,we can consider RandomForest as a  Best Algorithm for this problem.


### 6. Model Creation
So,using a RandomForest we got good accuracy , we can Hyperparameter tuning  for best accuracy.

Algorithm that can be used for Hyperparameter tuning are :-

a. GridSearchCV<br>
b. RandomizedSearchCV<br>
c. Bayesian Optimization-Automate Hyperparameter Tuning (Hyperopt)<br>
d. Sequential model based optimization<br>
e. Optuna-Automate Hyperparameter Tuning<br>
f. Genetic Algorithm<br>

Main parameters used by RandomForest Algorithm are :-

a. n_estimators --->    The number of trees in the forest.<br>
b. criterion--->{"mse", "mae"}-->The function to measure the quality of a split<br>
c. max_features--->{"auto", "sqrt", "log2"}-->    The number of features to consider when looking for the best split:


So, After Hyperparameter Tuning we got 90 % accuracy on test data and 94 % accuracy on train data. 
![](https://github.com/bibek376/Housing_Price_Prediction/blob/master/Picture_For_README/16.png)<br>
Now,Accuracy of model seems to be very good .so we can save the model using pickle. 




M. Model Creation

So,using a RandomForest we got good accuracy , we can Hyperparameter tuning for best accuracy.

Algorithm that can be used for Hyperparameter tuning are :-

a. GridSearchCV
b. RandomizedSearchCV
c. Bayesian Optimization-Automate Hyperparameter Tuning (Hyperopt)
d. Sequential model based optimization
e. Optuna-Automate Hyperparameter Tuning
f. Genetic Algorithm

Main parameters used by RandomForest Algorithm are :-

a. n_estimators ---> The number of trees in the forest.
b. criterion--->{"mse", "mae"}-->The function to measure the quality of a split
c. max_features--->{"auto", "sqrt", "log2"}--> The number of features to consider when looking for the best split:

So, After Hyperparameter Tuning we got 90 % accuracy on test data and 94 % accuracy on train data. 
Now,Accuracy of model seems to be very good .so we can save the model using pickle.
