import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error



# Load and preprocess data (unchanged)
cars = pd.read_csv("D:\AI\Car_Price_Predictor\Car_Price_Predictor\dataset\Dataset.csv")
cars = cars.drop(['full_model_name', 'brand_rank', 'distance below 30k km', 'new and less used',
                  'inv_car_price', 'inv_car_dist', 'inv_car_age', 'inv_brand', 'std_invprice',
                  'std_invdistance_travelled', 'std_invrank', 'best_buy1', 'best_buy2'], axis=1)

percentile25 = cars['price'].quantile(0.25)
percentile75 = cars['price'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
car_index = cars[(cars['price'] > upper_limit) | (cars['price'] < lower_limit)].index
cars.drop(car_index, inplace=True)
cars = cars.reset_index()
cars.drop('index', axis=1, inplace=True)

y = cars.iloc[:, 3]
x = cars.drop(['price'], axis=1)

for a in ['brand', 'model_name', 'fuel_type', 'city']:
    labelEncoder = LabelEncoder()
    x[a+'_enc'] = labelEncoder.fit_transform(x[a])

    names = labelEncoder.classes_
    nam = [b + '_model' if b.isnumeric() else b for b in names]

    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(x[[a]]).toarray())
    enc_df = enc_df.set_axis(nam, axis=1)
    x = x.join(enc_df, rsuffix='_other')

encoded = x[['brand', 'brand_enc', 'model_name', 'model_name_enc', 'fuel_type', 'fuel_type_enc', 'city', 'city_enc']]
x = x.drop(['brand', 'brand_enc', 'model_name', 'model_name_enc', 'fuel_type', 'fuel_type_enc', 'city', 'city_enc'], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)
# rf =ExtraTreesRegressor(n_estimators=900)
# rf.fit(xTrain, yTrain)
# print(rf.score(xTest,yTest))

xgb = XGBRegressor(n_estimators=900)  # You can tune n_estimators and other hyperparameters
xgb.fit(xTrain, yTrain)
print(xgb.score(xTest,yTest))
# Create a colorful and creative Tkinter interface
root = tk.Tk()
root.title("Car Price Predictor")

# Create a custom style for ttk widgets
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12), foreground="blue")
style.configure("TButton", font=("Helvetica", 12), background="green", foreground="white")

# Function to handle the submission
def submit():

       year_value = input1.get()
       brand_value = input2.get()
       model_value = input3.get()
       distance_value = input4.get()
       fuel_value = input5.get()
       city_value = input6.get()
       car_age_value = input7.get()
       thisdict = {
              "year": int(year_value),
              brand_value: 1,
               model_value: 1,
              "distance_travelled(kms)": int(distance_value),
              fuel_value: 1,
              city_value: 1,
              "car_age": int(car_age_value)
       }
       smthn = x.copy(deep=True)
       smthn = smthn.drop(smthn.index)
       input_df = pd.DataFrame(thisdict, index=[0])
       smthn.loc[len(smthn)] = 0
       smthn.update(input_df ,overwrite=True)
       output = rf.predict(smthn)
       print(smthn)
       
       output_label.config(text=f"The Price is {output}")




root = tk.Tk()
root.title("Input Window")

label1 = ttk.Label(root, text="year :")
label2 = ttk.Label(root, text="brand:")
label3 = ttk.Label(root, text="model_name:")
label4 = ttk.Label(root, text="distance_travelled(kms):")
label5 = ttk.Label(root, text="fuel_type:")
label6 = ttk.Label(root, text="city:")
label7 = ttk.Label(root, text="car_age:")

input1 = ttk.Entry(root)
input4 = ttk.Entry(root)
input7 = ttk.Entry(root)

input2_options = list(cars.brand.unique())
input3_options = list(cars.model_name.unique())
input5_options = list(cars.fuel_type.unique())
input6_options = list(cars.city.unique())
input2 = ttk.Combobox(root, values = input2_options)
input3 = ttk.Combobox(root, values = input3_options)
input5 = ttk.Combobox(root, values = input5_options)
input6 = ttk.Combobox(root, values = input6_options)

submit_button = ttk.Button(root, text="Submit", command=submit)

output_label = ttk.Label(root, text="")

label1.grid(row=0, column=0)
input1.grid(row=0, column=1)
label2.grid(row=0, column=2)
input2.grid(row=0, column=3)
label3.grid(row=0, column=4)
input3.grid(row=0, column=5)
label4.grid(row=0, column=6)
input4.grid(row=0, column=7)
label5.grid(row=0, column=8)
input5.grid(row=0, column=9)
label6.grid(row=0, column=10)
input6.grid(row=0, column=11)
label7.grid(row=0, column=12)
input7.grid(row=0, column=13)
submit_button.grid(row=7, column=1)
output_label.grid(row=8, column=1)
root.mainloop()
