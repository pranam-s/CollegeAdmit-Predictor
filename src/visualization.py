import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_admission_rates(df, college_tiers):
    plt.figure(figsize=(12, 6))
    for tier in college_tiers:
        tier_data = df[df['tier'] == tier]
        sns.lineplot(x='par_income_bin', y='rel_attend', data=tier_data, label=f'Tier {tier}')
    
    plt.title('Relative Attendance Rates by Income Bracket and College Tier')
    plt.xlabel('Parent Income Percentile')
    plt.ylabel('Relative Attendance Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/admission_rates.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importances or coefficients")

    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()

def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('static/roc_curve.png')
    plt.close()

def plot_income_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='par_income_bin', data=df)
    plt.title('Distribution of Parent Income Bins')
    plt.xlabel('Parent Income Percentile')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/income_distribution.png')
    plt.close()