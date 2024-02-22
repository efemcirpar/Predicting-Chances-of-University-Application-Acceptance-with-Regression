import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

students = pd.read_csv('adm_data.csv')
input_students = students.iloc[:,1:8]
Acceptence_Rate = students.iloc[:,8:]

GRE_Score = students.iloc[:,1:2]
TOEFL_Score = students.iloc[:,2:3]
University_Ratings = students.iloc[:,3:4]
SOP = students.iloc[:,4:5]
LOR = students.iloc[:,5:6]
CGPA = students.iloc[:,6:7]
Research = students.iloc[:,7:8]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#####################################

plt.scatter(GRE_Score, Acceptence_Rate, color = 'red')
plt.xlabel("GRE Score")
plt.ylabel("Acceptence Rate")

x_train, x_test, y_train, y_test = train_test_split(GRE_Score, Acceptence_Rate, test_size=0.2, random_state=0)

gre_Linear_Regressor = LinearRegression()
gre_Linear_Regressor= LinearRegression().fit(x_train, y_train)
gre_ar_prediction = gre_Linear_Regressor.predict(x_test)
plt.plot(x_test, gre_ar_prediction, color = 'blue')

plt.show()

#####################################

plt.scatter(TOEFL_Score, Acceptence_Rate, color = 'red')
plt.xlabel("TOEFL Score")
plt.ylabel("Acceptence Rate")

x_train, x_test, y_train, y_test = train_test_split(TOEFL_Score, Acceptence_Rate, test_size=0.2, random_state=0)

toefl_Linear_Regressor = LinearRegression()
toefl_Linear_Regressor= LinearRegression().fit(x_train, y_train)
toefl_ar_prediction = toefl_Linear_Regressor.predict(x_test)
plt.plot(x_test, toefl_ar_prediction, color = 'blue')

plt.show()

#####################################

plt.scatter(University_Ratings, Acceptence_Rate, color = 'red')
plt.xlabel("University Ratings")
plt.ylabel("Acceptence Rate")

x_train, x_test, y_train, y_test = train_test_split(University_Ratings, Acceptence_Rate, test_size=0.2, random_state=0)

unirate_Linear_Regressor = LinearRegression()
unirate_Linear_Regressor= LinearRegression().fit(x_train, y_train)
unirate_ar_prediction = unirate_Linear_Regressor.predict(x_test)
plt.plot(x_test, unirate_ar_prediction, color = 'blue')

plt.show()

#####################################

plt.scatter(SOP, Acceptence_Rate, color = 'red')
plt.xlabel("SOP")
plt.ylabel("Acceptence Rate")

x_train, x_test, y_train, y_test = train_test_split(SOP, Acceptence_Rate, test_size=0.2, random_state=0)

SOP_Linear_Regressor = LinearRegression()
SOP_Linear_Regressor= LinearRegression().fit(x_train, y_train)
SOP_ar_prediction = SOP_Linear_Regressor.predict(x_test)
plt.plot(x_test, SOP_ar_prediction, color = 'blue')

plt.show()

#####################################

plt.scatter(LOR, Acceptence_Rate, color = 'red')
plt.xlabel("LOR")
plt.ylabel("Acceptence Rate")

x_train, x_test, y_train, y_test = train_test_split(LOR, Acceptence_Rate, test_size=0.2, random_state=0)

LOR_Linear_Regressor = LinearRegression()
LOR_Linear_Regressor= LinearRegression().fit(x_train, y_train)
LOR_ar_prediction = LOR_Linear_Regressor.predict(x_test)
plt.plot(x_test, LOR_ar_prediction, color = 'blue')

plt.show()

#####################################

plt.scatter(CGPA, Acceptence_Rate, color = 'red')
plt.xlabel("CGPA")
plt.ylabel("Acceptence Rate")

x_train, x_test, y_train, y_test = train_test_split(CGPA, Acceptence_Rate, test_size=0.2, random_state=0)

cpga_Linear_Regressor = LinearRegression()
cpga_Linear_Regressor= LinearRegression().fit(x_train, y_train)
cpga_ar_prediction = cpga_Linear_Regressor.predict(x_test)
plt.plot(x_test, cpga_ar_prediction, color = 'blue')

plt.show()

#####################################

x_train, x_test, y_train, y_test = train_test_split(input_students, Acceptence_Rate, test_size=0.2, random_state=0)

students_Linear_Regressor = LinearRegression()
students_Linear_Regressor= LinearRegression().fit(x_train, y_train)
students_ar_prediction = students_Linear_Regressor.predict(x_test)

gre_input = int(input("Please enter your GRE Score: "))
toefl_input = int(input("Please enter your TOEFL Score: "))
unirate_input = float(input("Please enter University's Rating: "))
SOP_input = float(input("Please enter your SOP Score: "))
LOR_input = float(input("Please enter your LOR Score: "))
cpga_input = float(input("Please enter your CPGA: "))
researchpaper_input = int(input("Please enter your Research Paper number: "))

input_data = [[gre_input, toefl_input, unirate_input, SOP_input, LOR_input, cpga_input, researchpaper_input]]
input_data_dataframe = pd.DataFrame(input_data, columns = ['GRE Score', 'TOEFL Score','University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'])
x_train, x_test, y_train, y_test = train_test_split(input_students, Acceptence_Rate, test_size=0.2, random_state=0)
predict_Linear_Regressor = LinearRegression()
predict_Linear_Regressor= LinearRegression().fit(x_train, y_train)
predict_ar_prediction = predict_Linear_Regressor.predict(input_data_dataframe)
if(predict_ar_prediction > 1):
    predict_ar_prediction = 1
    print("%100")
else:
    print(f'%{predict_ar_prediction.item()*100:.2f}')
