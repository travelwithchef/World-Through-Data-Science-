import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

def main():
    # Load JSON data
    with open('data/weather-data.json', 'r') as f:
        data = json.load(f)

    # Convert to DataFrame (assuming data is a list of dicts)
    df = pd.DataFrame(data)

    # Preview data
    print("Data preview:")
    print(df.head())

    # Basic stats
    stats = df.describe(include='all')

    # Correlation matrix (numeric cols only)
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()

    # Plot 1: Correlation heatmap (Plotly)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                         title="Correlation Heatmap")

    # Plot 2: Histogram for each numeric column
    figs_hist = []
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f'Histogram of {col}', marginal="box")
        figs_hist.append(fig)

    # Machine Learning example
    # You need to set the target column name accordingly
    target_column = 'temperature'  # example, replace as per your JSON data

    if target_column in df.columns:
        data_ml = df.dropna(subset=[target_column])
        X = data_ml.drop(columns=[target_column])
        y = data_ml[target_column]

        # Use only numeric features
        X_num = X.select_dtypes(include=np.number)
        if not X_num.empty:
            X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42)

            # Simple regression if target numeric and many unique values
            if y.dtype.kind in 'biufc' and len(np.unique(y)) > 10:
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                ml_result = f"Regression Model Mean Squared Error: {mse:.4f}"
            else:
                # Otherwise classification
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                ml_result = f"Classification Model Accuracy: {acc:.4f}"
        else:
            ml_result = "No numeric features available for ML."
    else:
        ml_result = f"Target column '{target_column}' not found in data. Skipping ML."

    # Generate HTML content
    html_parts = []

    # Add title
    html_parts.append("<h1>Weather Data Analysis</h1>")

    # Add basic stats table
    html_parts.append("<h2>Summary Statistics</h2>")
    html_parts.append(stats.to_html())

    # Add ML result
    html_parts.append("<h2>Machine Learning Result</h2>")
    html_parts.append(f"<p>{ml_result}</p>")

    # Add correlation heatmap
    html_parts.append("<h2>Correlation Heatmap</h2>")
    html_parts.append(pio.to_html(fig_corr, full_html=False, include_plotlyjs='cdn'))

    # Add histograms
    html_parts.append("<h2>Histograms</h2>")
    for fig in figs_hist:
        html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Combine all parts into one HTML file
    html_output = """
    <html>
    <head>
        <title>Weather Data Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
    """ + "\n".join(html_parts) + """
    </body>
    </html>
    """

    # Save to lab.html
    with open("lab.html", "w") as f:
        f.write(html_output)

    print("lab.html generated successfully with visualizations.")

if __name__ == "__main__":
    main()
