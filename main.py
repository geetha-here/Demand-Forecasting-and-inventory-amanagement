from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/forecasts'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utility function for forecasting
def forecast_sales(data, steps=6):
    future_dates = pd.period_range(start=data.index[-1] + 1, periods=steps, freq='M')
    forecast_df = pd.DataFrame(index=future_dates)

    for cloth_type in data.columns:
        X = np.arange(len(data)).reshape(-1, 1)
        y = data[cloth_type].values

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(data), len(data) + steps).reshape(-1, 1)
        forecast = model.predict(future_X)
        forecast_df[cloth_type] = forecast

    return forecast_df

# Function to generate and save bar chart for categorical data
def plot_bar_chart(data, column_name):
    counts = data[column_name].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values, color=plt.cm.Paired.colors[:len(counts)])
    plt.title(f'Sales Distribution by {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    bar_path = os.path.join(STATIC_FOLDER, f'{column_name}_bar.png')
    plt.savefig(bar_path)
    plt.close()
    return bar_path

# Function to generate and save demand forecasting line chart
def plot_demand_forecast(forecasted_sales, column_name):
    plt.figure(figsize=(10, 6))
    plt.plot(forecasted_sales.index.strftime('%Y-%m'), forecasted_sales[column_name], marker='o', label=f'Forecasted Demand for {column_name}')
    plt.title(f'Demand Forecast for {column_name}')
    plt.xlabel('Month')
    plt.ylabel('Sales Forecast')
    plt.xticks(rotation=45)
    plt.legend()
    forecast_plot_path = os.path.join(STATIC_FOLDER, f'{column_name}_demand_forecast.png')
    plt.savefig(forecast_plot_path)
    plt.close()
    return forecast_plot_path

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'yash' and password == '12345':
            return redirect(url_for('upload'))
        else:
            error = "Invalid username or password. Try again."
            return render_template('login.html', error=error)
    return render_template('login.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file part in the request."
            return render_template('upload.html', error=error)

        file = request.files['file']
        if file.filename == '':
            error = "No file selected."
            return render_template('upload.html', error=error)

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process dataset
            df = pd.read_csv(file_path)
            df['Date of Purchasing'] = pd.to_datetime(df['Date of Purchasing'])
            df['Month'] = df['Date of Purchasing'].dt.to_period('M')
            sales_data = df.groupby(['Month', 'Type of Cloth Purchased']).size().unstack().fillna(0)

            # Forecast sales
            forecasted_sales = forecast_sales(sales_data, steps=6)

            # Save forecast to CSV for debugging
            forecast_csv_path = os.path.join(UPLOAD_FOLDER, 'forecasted_sales.csv')
            forecasted_sales.to_csv(forecast_csv_path)

            # Generate visualizations (Demand Forecast)
            for cloth_type in sales_data.columns:
                plot_demand_forecast(forecasted_sales, cloth_type)

            # Create optimized production plan
            safety_margin = 1.1  # Increase production by 10%
            optimized_plan = forecasted_sales.applymap(lambda x: max(0, int(x * safety_margin)))
            optimized_csv_path = os.path.join(UPLOAD_FOLDER, 'optimized_plan.csv')
            optimized_plan.to_csv(optimized_csv_path)

            # Create bar charts for categorical data
            plot_bar_chart(df, 'Type of Cloth Purchased')
            plot_bar_chart(df, 'Material of the Cloth')

            return redirect(url_for('results'))
    return render_template('upload.html')


@app.route('/results')
def results():
    forecast_csv_path = os.path.join(UPLOAD_FOLDER, 'forecasted_sales.csv')
    optimized_csv_path = os.path.join(UPLOAD_FOLDER, 'optimized_plan.csv')

    if os.path.exists(forecast_csv_path) and os.path.exists(optimized_csv_path):
        forecasted_sales = pd.read_csv(forecast_csv_path, index_col=0)
        optimized_plan = pd.read_csv(optimized_csv_path, index_col=0)

        # Demand forecasting (for the next 6 months)
        forecasted_demand = forecasted_sales.copy()  # Forecast demand is the same as the forecasted sales in this case
        
        # Render the result page
        return render_template(
            'result.html',
            forecasts=forecasted_sales.columns.tolist(),
            forecasted_sales=forecasted_sales.to_dict(orient='records'),
            optimized_plan=optimized_plan.to_dict(orient='records'),
            forecasted_demand=forecasted_demand.to_dict(orient='records')
        )
    else:
        return "Error: No forecasted or optimized data available. Please upload a dataset."


if __name__ == '__main__':
    app.run(debug=True)
