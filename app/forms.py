from flask_wtf import FlaskForm
from wtforms import SelectField, FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class PredictionForm(FlaskForm):
    tier = SelectField('College Tier', choices=[(1, 'Ivy-Plus'), (2, 'Other Elite'), (3, 'Highly Selective Public'), (4, 'Highly Selective Private'), (5, 'Selective Public'), (6, 'Selective Private')], validators=[DataRequired()])
    flagship = SelectField('Flagship', choices=[(0, 'No'), (1, 'Yes')], validators=[DataRequired()])
    public = SelectField('Public', choices=[(0, 'No'), (1, 'Yes')], validators=[DataRequired()])
    income_percentile = FloatField('Parent Income Percentile', validators=[DataRequired(), NumberRange(min=0, max=100)])
    rel_apply = FloatField('Relative Application Rate', validators=[DataRequired(), NumberRange(min=0)])
    rel_attend = FloatField('Relative Attendance Rate', validators=[DataRequired(), NumberRange(min=0)])
    rel_att_cond_app = FloatField('Relative Attendance Rate Conditional on Application', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Predict')