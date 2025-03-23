# ⚡️ Peer-to-Peer Energy Trading Project: Machine Learning ⚡️

Welcome to the Machine Learning section of the Peer-to-Peer Energy Trading Project! This repository contains the machine learning components that predict electricity consumption and production for efficient energy trading.

## 📁 Project Structure

The repository is organized into two main folders: `Consumption` and `Production`.

### 🔋 Consumption

The `Consumption` folder contains two models:

1. **Short Term Consumption Model**: Predicts electricity consumption for the next three hours.
2. **Long Term Consumption Model**: Predicts the average daily consumption for a week.

#### Files in `Consumption`:

- `household_power_consumption.csv`: 📊 Data file for household power consumption.
- `Long Term Consumption Model Pre Processing.ipynb`: 📝 Pre-processing notebook for the long-term consumption model.
- `Long Term SGD Regression Model.ipynb`: 🧠 Python notebook for creating the long-term consumption model.
- `long_term_consumption.csv`: 📊 Data file for long-term consumption.
- `long_term_scaler.pkl`: 📏 Scaler file for long-term consumption data.
- `Short Term Consumption Model Pre Processing.ipynb`: 📝 Pre-processing notebook for the short-term consumption model.
- `Short Term SGD Regression Model.ipynb`: 🧠 Python notebook for creating the short-term consumption model.
- `short_term_consumption.csv`: 📊 Data file for short-term consumption.
- `short_term_scaler.pkl`: 📏 Scaler file for short-term consumption data.

### ⚙️ Production

The `Production` folder contains one model:

1. **Short Term Production Model**: Predicts electricity production for the next three hours.

#### Files in `Production`:

- `production_dataset.csv`: 📊 Data file for production dataset.
- `production_dataset.xlsx`: 📊 Excel file for production dataset.
- `Production Model Pre Processing.ipynb`: 📝 Pre-processing notebook for the production model.
- `Production SGD Regression Model.ipynb`: 🧠 Python notebook for creating the production model.
- `production_scaler.pkl`: 📏 Scaler file for production data.

## 🚀 Getting Started

To get started with the project, you need to install the required dependencies. You can install them using the following command:

```sh
pip install -r requirements.txt
