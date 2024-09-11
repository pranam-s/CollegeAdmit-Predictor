# CollegeAdmit Predictor

CollegeAdmit Predictor is an AI-powered application that predicts a student's likelihood of admission to various colleges based on their academic performance and socioeconomic background.

## Features

- Analyze historical college admission data
- Predict admission chances for different college tiers
- Visualize admission trends across income brackets
- Interactive web interface for user input and results display

## Dataset

The project uses the [Elite College Admissions dataset](https://www.kaggle.com/datasets/mexwell/elite-college-admissions) from Kaggle.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/CollegeAdmit_Predictor.git
   cd CollegeAdmit_Predictor
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Please download the dataset before following the steps.

1. Run the Flask application:
   ```
   python run.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter the required information and click "Predict" to see the admission prediction results.

## Project Structure

- `data/`: Contains the dataset used for analysis and model training, please download
- `src/`: Source code for data preprocessing, model training, and visualization
- `app/`: Flask application files for the web interface
- `tests/`: Unit tests for data preprocessing and model functions

## Visualizations

The project includes several visualizations to help understand the data and model performance:

- Admission rates by income bracket and college tier
- Feature importance
- ROC curve
- Income distribution

To view these visualizations, run the application and navigate to the `/visualizations` route.

## Error Handling

The application includes error handling for common issues:

- 404 Page Not Found errors
- 500 Internal Server errors
- Input validation errors

Errors are logged for debugging purposes.

## Logging

The application uses Python's built-in logging module to log important events and errors. Logs are printed to the console and can be easily extended to write to a file if needed.