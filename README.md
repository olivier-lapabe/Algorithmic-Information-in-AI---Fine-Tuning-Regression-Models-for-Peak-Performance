# Regression Model Tuning Project

This project aims to fine-tune regression models to identify the one that provides the optimal balance between model complexity and predictive performance, primarily evaluated using the Model Description Length (MDL).

## Project Structure

The project consists of the following files:

- `model_tuning.py`: This is the main script that runs the model tuning process.
- `model_complexity.py`: Contains functions to calculate the complexity of models.
- `requirements.txt`: Lists all Python libraries required to run the project.

## Getting Started

To run this project, you need to have Python installed on your system. If you do not have Python installed, please visit [Python's website](https://www.python.org/) for installation instructions.

### Installation

1. Unzip the project files.

2. It's recommended to create a virtual environment to keep dependencies required by the project separate from your global Python environment.
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies.
   ```
   pip install -r requirements.txt
   ```

### Running the Model Tuning

To run the model tuning process, execute the `model_tuning.py` script:

```
python model_tuning.py
```

Datasets can be changes in the main method by specifying the variable "d".
This script will scale the feature variables, construct various regression models, experiment with different polynomial degrees, fit the models, calculate the MDL, and perform a comparative analysis.

### Output

The output will be displayed in the terminal. It will include the complexities and MSEs of the different models tested, along with a plot illustrating the synthetic data used for model training.
