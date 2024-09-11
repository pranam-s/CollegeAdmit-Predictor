from flask import render_template, request, jsonify, flash
from app import app
from app.forms import PredictionForm
from src.model import predict_admission
from src.utils import logger
import pandas as pd
import numpy as np

@app.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    if form.validate_on_submit():
        try:
            input_data = preprocess_input(form.data)
            prediction = predict_admission(app.model, input_data)
            return render_template('result.html', prediction=prediction[0][1])
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash("An error occurred while processing your request. Please try again.", "error")
    return render_template('index.html', form=form)

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

def preprocess_input(form_data):
    input_df = pd.DataFrame({
        'tier': [form_data['tier']],
        'flagship': [form_data['flagship']],
        'public': [form_data['public']],
        'par_income_bin': [form_data['income_percentile']],
        'rel_apply': [form_data['rel_apply']],
        'rel_attend': [form_data['rel_attend']],
        'rel_att_cond_app': [form_data['rel_att_cond_app']]
    })
    return app.preprocessor.transform(input_df)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500