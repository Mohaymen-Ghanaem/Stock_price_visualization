# Stock fitting project
#Stack: CSV, Numpy, scikit-learn, matplotlib
#Import all requirements
import csv
import numpy as np
from numpy.ma.core import get_data
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from datetime import datetime
dates = []
prices = []
#Function to grab CSV file data
def grab_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            date_obj = datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S')
            dates.append(date_obj.day)
            prices.append(float(row[1]))
#Function to create the graph
def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates,prices)
    plt.scatter(dates, prices, color= 'red', label= 'Data')
    plt.plot(dates, svr_rbf.predict(dates), color= 'blue', label= 'RBF Model')
    plt.plot(dates, svr_lin.predict(dates), color= 'green', label= 'Linear Model')
    plt.plot(dates, svr_poly.predict(dates), color= 'purple', label= 'Polynomial Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    x_predict = [[x]]
    return svr_rbf.predict(x_predict)[0], svr_lin.predict(x_predict)[0], svr_poly.predict(x_predict)[0]
grab_data('Appl_May_Stock.csv')
predicted_prices = predict_prices(dates, prices, 30)

