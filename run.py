from app import app
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model import train_model, evaluate_model
from src.visualization import plot_admission_rates, plot_feature_importance, plot_roc_curve, plot_income_distribution
from src.utils import logger
import os

if __name__ == '__main__':
    try:
        # Create static folder if it doesn't exist
        if not os.path.exists('static'):
            os.makedirs('static')

        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = load_data('data/CollegeAdmissions_Data.csv')
        X, y, feature_names, preprocessor = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Train the model
        logger.info("Training the model...")
        app.model = train_model(X_train, y_train)
        app.preprocessor = preprocessor

        # Evaluate the model
        logger.info("Evaluating the model...")
        accuracy, roc_auc, report = evaluate_model(app.model, X_test, y_test)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info("Classification Report:")
        logger.info(report)

        # Generate visualizations
        logger.info("Generating visualizations...")
        plot_admission_rates(df, range(1, 7))
        plot_feature_importance(app.model, feature_names)
        plot_roc_curve(app.model, X_test, y_test)
        plot_income_distribution(df)

        # Run the Flask app
        logger.info("Starting the Flask app...")
        app.run(debug=False)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")