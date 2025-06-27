# Task1

## Folder Structure:

  - code/: Contains the Python script for preprocessing, task2, task3 and the packaged Tableau workbook having vizualisation.
  - images/: Contains the images used in the report.
  - Dataset/: Contains the preprocessed data, and the Dataset created at intermediate iterations (2 and 3).

## Requirements:

Before running the code, make sure you have the required dependencies installed. Install the required packages from `requirements.txt` using either of the following commands:

   - Using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

   - Using `conda`:
     ```bash
     conda install --file requirements.txt
     ```

## Running the Code:
| File Name            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `preprocessing.py`   | Preprocesses the 2001 and 2011 datasets by calculating literacy rates, higher education proportions, urbanization rate, growth rate, and density rate. Ensures datatype consistency between datasets and prepares cleaned data for analysis. |
| `iter2_linear_modeling.py` | Performs linear regression analysis on the relationship between female-to-male literacy ratio and overall literacy rate for 2001 and 2011. It evaluates statistical significance, computes metrics like mean squared error, and visualizes the results with regression lines. |
| `iter2.py` | Computes cluster transition matrices between 2001 and 2011 for three criteria: higher education & urbanization rate, literacy rate & population density, and male-female literacy ratio. Visualizes transitions as heatmaps and calculates correlations between demographic changes. |
| `iter3.py` | Assigns weights to literacy and demographic features based on cluster transitions, calculates final weights using PCA, and computes weighted scores for each state in 2001 and 2011 for further visualizations and analysis. |

To run file `a.py` do:
```bash
  python3 code/a.py
``` 

## Visualizations
To view the visualizations, follow these steps:

    1. Download and install [Tableau Desktop](https://www.tableau.com/products/desktop).
    2. Install the "Sankey" extension in Tableau.
    3. Open the provided Tableau Packaged Workbook to explore the visualizations.

## Images

All the images used in the report are present in the `images/` directory, named according to the figure number of the report pdf.
