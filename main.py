import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics, model_selection

# List location of the data file being imported.
url = "HR Employee Attrition.csv"

# List all column names in order used by the CSV file.
names = ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
         'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber',
         'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel',
         'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
         'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike',
         'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
         'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
         'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
         'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Read data set into data frame and establish column names. Create a duplicate data frame called df_test to alter.
df = pd.read_csv(url, names=names)
df_test = df

# For ease of testing different data, a line has been written to drop all columns so that the columns to keep can be
# commented out. Note that the indices used to pull X & y values could need to be updated with each change.
df_test = df_test.drop('Age', axis=1)
# df_test = df_test.drop('Attrition', axis=1)
df_test = df_test.drop('BusinessTravel', axis=1)
df_test = df_test.drop('DailyRate', axis=1)
df_test = df_test.drop('Department', axis=1)
df_test = df_test.drop('DistanceFromHome', axis=1)
df_test = df_test.drop('Education', axis=1)
df_test = df_test.drop('EducationField', axis=1)
df_test = df_test.drop('EmployeeCount', axis=1)
df_test = df_test.drop('EmployeeNumber', axis=1)
# df_test = df_test.drop('EnvironmentSatisfaction', axis=1)
df_test = df_test.drop('Gender', axis=1)
df_test = df_test.drop('HourlyRate', axis=1)
# df_test = df_test.drop('JobInvolvement', axis=1)
df_test = df_test.drop('JobLevel', axis=1)
df_test = df_test.drop('JobRole', axis=1)
# df_test = df_test.drop('JobSatisfaction', axis=1)
df_test = df_test.drop('MaritalStatus', axis=1)
df_test = df_test.drop('MonthlyIncome', axis=1)
df_test = df_test.drop('MonthlyRate', axis=1)
df_test = df_test.drop('NumCompaniesWorked', axis=1)
df_test = df_test.drop('Over18', axis=1)
df_test = df_test.drop('OverTime', axis=1)
df_test = df_test.drop('PercentSalaryHike', axis=1)
# df_test = df_test.drop('PerformanceRating', axis=1)
df_test = df_test.drop('RelationshipSatisfaction', axis=1)
df_test = df_test.drop('StandardHours', axis=1)
df_test = df_test.drop('StockOptionLevel', axis=1)
df_test = df_test.drop('TotalWorkingYears', axis=1)
df_test = df_test.drop('TrainingTimesLastYear', axis=1)
# df_test = df_test.drop('WorkLifeBalance', axis=1)
df_test = df_test.drop('YearsAtCompany', axis=1)
df_test = df_test.drop('YearsInCurrentRole', axis=1)
df_test = df_test.drop('YearsSinceLastPromotion', axis=1)
df_test = df_test.drop('YearsWithCurrManager', axis=1)

# Establishes Logistic Regression Model and the x & y values. X is the kept indicators from data and y is attrition.
# Also establishes a test data set by splitting the full data into 70% train and 30% test
mylog_model = linear_model.LogisticRegression(max_iter=2000)

y = df_test.values[:, 0]
X = df_test.values[:, 1:35]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.3)

mylog_model.fit(X_train, y_train)

y_prediction = mylog_model.predict(X_test)

# Test and print the accuracy of the model using the previously established test data.
accuracy = metrics.accuracy_score(y_test, y_prediction)
accuracy = accuracy * 100

print("The model has an accuracy of {}".format(round(accuracy, 2)))

# Print a confusion matrix using the previously established test data.
print(metrics.plot_confusion_matrix(mylog_model, X_test, y_test))

# Establish and print a horizontal bar chart of the reports of Work Life Balance in the data set.
import matplotlib.pyplot as plt


def bar_charth(numbers, labels, pos):
    plt.barh(pos, numbers, color='green')
    plt.yticks(ticks=pos, labels=labels)
    plt.show()


if __name__ == '__main__':
    hNumbers = [df['WorkLifeBalance'].value_counts()[1], df['WorkLifeBalance'].value_counts()[2],
                df['WorkLifeBalance'].value_counts()[3], df['WorkLifeBalance'].value_counts()[4]]
    hLabels = ['Lowest', 'Low', 'High', 'Highest']
    pos = list(range(4))
    bar_charth(hNumbers, hLabels, pos)

# Establish and print a bar graph of the reports of Job Satisfaction in the data set
import matplotlib.pyplot as plt


def bar_chart(numbers, labels, pos):
    plt.bar(pos, numbers, color='blue')
    plt.xticks(ticks=pos, labels=labels)
    plt.title('Reports of Job Satisfaction')
    plt.xlabel('Job Satisfaction')
    plt.ylabel('Number of Employees')
    plt.show()


if __name__ == '__main__':
    sNumbers = [df['JobSatisfaction'].value_counts()[1], df['JobSatisfaction'].value_counts()[2],
                df['JobSatisfaction'].value_counts()[3], df['JobSatisfaction'].value_counts()[4]]
    sLabels = ['Lowest', 'Low', 'High', 'Highest']
    sPos = list(range(4))
    bar_chart(sNumbers, sLabels, sPos)

# Establish and print a pie chart of the Attrition reported in the data set.
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


def pie_chart():
    numbers = [df['Attrition'].value_counts()['Yes'], df['Attrition'].value_counts()['No']]
    labels = ['Attrition', 'Not Attrition']

    explode = (0.1, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(numbers, explode=explode, labels=labels,
            shadow=True, startangle=90,
            autopct='%1.1f%%')
    ax1.axis('equal')
    plt.title('Attrition Percentage')
    plt.show()


if __name__ == '__main__':
    pie_chart()

# Print user commands for a prediction to be made and take in user input.
value = input(
    "Enter the user Environment Satisfaction, Job Involvement, Job Satisfaction, Performance Rating, and Work Life Balance \n as reported by the latest employee survey. Remember that all values are rated 1 through 4. \n Enter them in order seperated by a space. \n")

# Print prediction based on user input.
valueList = list(value.split(" "))
print(
    'If Yes is displayed then the employee is at risk of attrition. If No is displayed then there is no risk of attrition.')
print(mylog_model.predict([valueList]))



