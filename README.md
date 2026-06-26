# Task 5: Predictive Regression Architecture (Movie Revenue Predictor)

## Project Description
This project focuses on the implementation of a **Supervised Continuous Regression** pipeline engineered to estimate global box office revenue based on commercial production inputs. Utilizing an Ordinary Least Squares (OLS) **Linear Regression** algorithm, the framework maps numerical features like production budgets, corporate advertising capital, and digital social buzz scores to forecast precise continuous financial outcomes rather than discrete classifications.

The project processes a programmatically engineered entertainment matrix (`movie_revenue_data.csv`), normalizes feature sets to prevent coefficient imbalances, and uses regression error diagnostics to calculate accuracy.

## Key Technical Pillars
- **Continuous Value Mapping:** Implements a multi-variable Linear Regression architecture that mathematically computes weights ($w$) and biases ($b$) for numerical production attributes to map an optimized hyperplane.
- **Ordinary Least Squares (OLS) Optimization:** Leverages variance-minimization math to minimize residual sum of squares between actual box office gross values and model-predicted lines.
- **Feature Scaling Guardrail:** Integrated `StandardScaler` to uniform dimensional arrays, ensuring high-scale production funds do not shadow lower-scale digital buzz indices.
- **Continuous Error Loss Diagnostics:** Evaluates system accuracy using Mean Absolute Error (MAE) for clear monetary drift testing, alongside R-squared ($R^2$) scores to quantify the variance percentage captured by the engine.

## Tools & Technologies
- **Language:** Python 3.x
- **Libraries:** NumPy, Pandas, Scikit-Learn
- **Dataset:** Programmatically synthesized `movie_revenue_data.csv` (150 movie entries)
- **Environment:** Visual Studio Code (Jupyter Notebook environment)

## Evaluation Insights & Prediction Diagnostic Summary
The system reports absolute margins of error and predictive correlation metrics upon model execution.

### Sample Regression Metrics:
```text
============================================================
   STATISTICAL MODEL DIAGNOSTICS & LOSS METRICS
============================================================
Mean Absolute Error (MAE) : $8.42 Million
Mean Squared Error (MSE)  : $98.15 Million
R-squared ($R^2$) Score      : 0.9642 

--- Side-by-Side Actual vs Predicted Projections ---
Actual Revenue ($M)  Predicted Revenue ($M)
             120.45                  118.22
             245.10                  251.40
              54.30                   48.12
             189.65                  194.10
```
## How to Run
1. Open this project folder inside Visual Studio Code.
2. Ensure the **Jupyter Extension** is fully installed and enabled.
3. Open the `Revenue_Prediction_Engine.ipynb` notebook file.
4. Click **Run All** to generate the corporate box office records, perform data scaling, and view actual vs predicted revenue sheets.
