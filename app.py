# ==========
# LIBRARIES
# ==========
# Standard Library
import time
import random
import os

# Data Processing
import pandas as pd
import numpy as np

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit
import streamlit as st

# Machine Learning - Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Machine Learning - Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Machine Learning - Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Machine Learning - Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Model Persistence
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# =============
# CONFIGURATION
# =============

# Page Configuration
st.set_page_config(
    page_title="Bank Marketing Analytics System",
    page_icon=":bank:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color Theme
available_colors = ["blue", "green", "red", "orange", "violet", "gray"]
theme_color = random.choice(available_colors)

# File Path
FILE_PATH = "data/bank-additional-full.csv"

# Models Directory
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ===================
# STREAM DATA FUNCTION
# ====================

def stream_data(text):
    """Stream text word by word for better user experience."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

# ============
# DATA LOADING
# ============

@st.cache_data
def load_dataset():
    """
    Load and perform initial cleaning on the Bank Marketing dataset.
    Returns the cleaned DataFrame ready for analysis.
    """
    # Load data
    df = pd.read_csv(FILE_PATH, sep=";")
    
    # Convert binary columns to numeric
    binary_cols = ["default", "housing", "loan", "y"]
    for col in binary_cols:
        # Handle 'unknown' properly
        df[col] = df[col].replace({"yes": 1, "no": 0, "unknown": 0})
        # Ensure no NaN values
        df[col] = df[col].fillna(0)
        # Convert to int
        df[col] = df[col].astype(int)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

@st.cache_data
def get_dataset_metadata(df):
    """Extract and return dataset metadata."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_features": len(df.select_dtypes(include=['int64', 'float64']).columns),
        "categorical_features": len(df.select_dtypes(include=['object']).columns),
        "binary_features": len([col for col in df.columns if df[col].nunique() == 2]),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        "duplicates": df.duplicated().sum(),
        "target_distribution": df['y'].value_counts().to_dict(),
        "subscription_rate": (df['y'].sum() / len(df)) * 100
    }

# ================
# DATA PREPARATION
# ================

def prepare_data_for_modeling(df, target='y'):
    """
    Prepare data for machine learning modeling.
    Includes feature engineering, encoding, and scaling.
    """
    # Create a copy
    data = df.copy()
    
    # Feature Engineering
    # Age groups
    data['age_group'] = pd.cut(
        data['age'],
        bins=[0, 25, 35, 50, 65, 100],
        labels=['Under 25', '25-35', '36-50', '51-65', 'Over 65']
    )
    
    # Campaign intensity
    data['campaign_intensity'] = pd.cut(
        data['campaign'],
        bins=[0, 1, 3, 5, 10, 100],
        labels=['Low (1)', 'Medium (2-3)', 'High (4-5)', 'Very High (6-10)', 'Extreme (>10)']
    )
    
    # Previous contact indicator
    data['previously_contacted'] = (data['pdays'] != 999).astype(int)
    
    # Define feature categories
    numerical_features = [
        'age', 'duration', 'campaign', 'pdays', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
        'euribor3m', 'nr.employed'
    ]
    
    categorical_features = [
        'job', 'marital', 'education', 'contact', 
        'month', 'day_of_week', 'poutcome',
        'age_group', 'campaign_intensity'
    ]
    
    binary_features = ['default', 'housing', 'loan', 'previously_contacted']
    
    # Prepare features and target
    X = data.drop(columns=[target])
    y = data[target]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('bin', 'passthrough', binary_features)
    ])
    
    return X, y, preprocessor

def create_model_pipeline(preprocessor, model):
    """Create a complete pipeline with preprocessing and model."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# ========================
# CLASS IMBALANCE HANDLING
# ========================

def get_class_weights(y):
    """
    Calculate class weights for imbalanced datasets.
    Returns a dictionary of class weights.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def manual_oversample(X, y):
    """
    Manual oversampling of minority class using numpy.
    Alternative to SMOTE when imbalanced-learn is not available.
    """
    # Separate majority and minority classes
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    
    # Get indices for each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y != minority_class)[0]  # Get all majority indices
    
    # Calculate how many samples to add
    n_majority = len(majority_indices)
    n_minority = len(minority_indices)
    n_to_add = n_majority - n_minority
    
    if n_to_add <= 0:
        return X, y
    
    # Randomly sample from minority class with replacement
    sampled_indices = np.random.choice(minority_indices, size=n_to_add, replace=True)
    
    # Combine original data with oversampled data
    X_resampled = np.vstack([X, X[sampled_indices]])
    y_resampled = np.hstack([y, y[sampled_indices]])
    
    return X_resampled, y_resampled

# =================
# MODEL DEFINITIONS
# =================

def get_models():
    """Return dictionary of classification models with proper handling."""
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=10,
            min_samples_split=10,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights='distance'
        ),
        "Support Vector Machine": SVC(
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        "Naive Bayes": GaussianNB()
    }
    
    return models

def get_model_performance_summary():
    """Return a summary of expected model performance based on literature."""
    summary = {
        "Random Forest": {"AUC-ROC": "0.89+", "F1": "0.72+", "Speed": "Medium"},
        "Logistic Regression": {"AUC-ROC": "0.82+", "F1": "0.68+", "Speed": "Fast"},
        "Gradient Boosting": {"AUC-ROC": "0.90+", "F1": "0.74+", "Speed": "Medium"},
        "XGBoost": {"AUC-ROC": "0.92+", "F1": "0.75+", "Speed": "Fast"},
    }
    return summary

# ===========================
# MODEL TRAINING & EVALUATION
# ===========================

def train_model(X, y, preprocessor, model_name, use_oversampling=True):
    """
    Train a single model with preprocessing.
    Uses class weights or manual oversampling for imbalance handling.
    """
    model = get_models()[model_name]
    pipeline = create_model_pipeline(preprocessor, model)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    models_needing_oversampling = ["K-Nearest Neighbors", "Naive Bayes", "Gradient Boosting"]
    
    if use_oversampling and model_name in models_needing_oversampling:
        # These models don't have class_weight parameter, so use manual oversampling
        X_train_resampled, y_train_resampled = manual_oversample(
            X_train.values, y_train.values
        )
        X_train_resampled = pd.DataFrame(
            X_train_resampled,columns=X_train.columns
        )
        y_train_resampled = pd.Series(
            y_train_resampled
        )
        # Fit the entire pipeline on resampled data
        # Note: We need to recreate pipeline with preprocessor already fitted
        pipeline.fit(X_train_resampled, y_train_resampled)
    else:
        # Models with class_weight='balanced' handle imbalance internally
        pipeline.fit(X_train, y_train)
    
    return pipeline, X_train, X_test, y_train, y_test, None, None

def evaluate_model(pipeline, X_test, y_test):
    """Evaluate a trained model and return metrics."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
        "Predictions": y_pred,
        "Probabilities": y_proba
    }
    
    return metrics

def cross_validate_model(pipeline, X, y, cv=5):
    """Perform cross-validation on the model."""
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    return {
        "mean_score": cv_scores.mean(),
        "std_score": cv_scores.std(),
        "scores": cv_scores,
        "cv_used": cv
    }


def train_and_evaluate_all_models(X, y, preprocessor):
    """Train and evaluate all models, return results dictionary."""
    results = {}
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    models = get_models()
    total_models = len(models)
    
    for idx, (model_name, model) in enumerate(models.items()):
        status_text.text(f"Training {model_name}... ({idx+1}/{total_models})")
        
        # Train model
        pipeline, X_train, X_test, y_train, y_test, _, _ = train_model(
            X, y, preprocessor, model_name
        )
        
        # Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        # Store results
        results[model_name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test
        }
        
        # Update progress
        progress_bar.progress((idx + 1) / total_models)
    
    status_text.success("All models selected trained successfully!")
    progress_bar.empty()
    
    return results

def save_model(pipeline, model_name):
    """Save trained model to file."""
    filename = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
    joblib.dump(pipeline, filename)
    return f"Model saved as {filename}"

def load_saved_model(model_name):
    """Load saved model from file."""
    filename = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
    if os.path.exists(filename):
        return joblib.load(filename)

# =======================
# VISUALIZATION FUNCTIONS
# =======================

def plot_age_distribution(df):
    """Plot age distribution histogram."""
    fig = px.histogram(
        df, x='age', nbins=30,
        title='Distribution of Client Ages',
        labels={'age': 'Age (years)', 'count': 'Number of Clients'},
        color_discrete_sequence=['#1a237e']
    )
    fig.add_vline(x=df['age'].mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {df['age'].mean():.1f}")
    fig.update_layout(showlegend=False, height=400, bargap=0.05)
    return fig

def plot_job_distribution(df):
    """Plot job distribution bar chart."""
    job_counts = df['job'].value_counts().reset_index()
    job_counts.columns = ['Job', 'Count']
    
    fig = px.bar(
        job_counts, x='Job', y='Count',
        title='Client Distribution by Job Type',
        labels={'Count': 'Number of Clients'},
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    return fig

def plot_subscription_distribution(df):
    """Plot target variable distribution."""
    sub_counts = df['y'].value_counts().reset_index()
    sub_counts.columns = ['Subscribed', 'Count']
    sub_counts['Subscribed'] = sub_counts['Subscribed'].map({1: 'Yes', 0: 'No'})
    
    fig = px.pie(
        sub_counts, values='Count', names='Subscribed',
        title=f'Term Deposit Subscription Distribution<br><sub>Rate: {df["y"].mean()*100:.1f}%</sub>',
        color='Subscribed',
        color_discrete_map={'Yes': '#2e7d32', 'No': '#c62828'}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig

def plot_duration_vs_subscription(df):
    """Plot duration boxplot by subscription status."""
    plot_df = df.copy()
    plot_df['Subscribed'] = plot_df['y'].map({1: 'Yes', 0: 'No'})
    
    fig = px.box(
        plot_df, x='Subscribed', y='duration',
        title='Call Duration by Subscription Status',
        labels={'duration': 'Duration (seconds)'},
        color='Subscribed',
        color_discrete_map={'Yes': '#2e7d32', 'No': '#c62828'}
    )
    fig.update_layout(height=400)
    return fig

def plot_age_vs_subscription(df):
    """Plot age vs subscription violin plot."""
    plot_df = df.copy()
    plot_df['Subscribed'] = plot_df['y'].map({1: 'Yes', 0: 'No'})
    
    fig = px.violin(
        plot_df, x='Subscribed', y='age',
        title='Age Distribution by Subscription Status',
        labels={'age': 'Age (years)'},
        color='Subscribed',
        color_discrete_map={'Yes': '#2e7d32', 'No': '#c62828'},
        box=True
    )
    fig.update_layout(height=400)
    return fig

def plot_correlation_heatmap(df):
    """Plot correlation heatmap of numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numerical_cols].corr()
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        aspect='auto',
        title='Feature Correlation Heatmap',
        color_continuous_scale='balance',
        zmin=-1, zmax=1
    )
    fig.update_layout(height=500)
    return fig

def plot_economic_trends(df):
    """Plot economic indicators over time."""
    monthly = df.groupby('month')[['emp.var.rate', 'euribor3m', 'cons.conf.idx']].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['emp.var.rate'],
        name='Employment Variation Rate',
        mode='lines+markers',
        line=dict(color='#1a237e')
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['euribor3m'],
        name='Euribor 3M Rate',
        mode='lines+markers',
        line=dict(color='#ff6f00')
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['cons.conf.idx'],
        name='Consumer Confidence Index',
        mode='lines+markers',
        line=dict(color='#2e7d32')
    ))
    
    fig.update_layout(
        title='Economic Indicators Trend by Month',
        xaxis_title='Month',
        yaxis_title='Value',
        height=400,
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect='auto',
        title='Confusion Matrix',
        labels=dict(x='Predicted', y='Actual', color='Count'),
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    return fig

def plot_roc_curve(y_test, y_proba):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc:.3f})',
        line=dict(color='#1a237e', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400,
        legend=dict(x=0.7, y=0.05)
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based models."""
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        
        # Ensure we have enough feature names
        n_features = min(len(feature_names), len(importances))
        
        imp_df = pd.DataFrame({
            'Feature': feature_names[:n_features],
            'Importance': importances[:n_features]
        }).sort_values('Importance', ascending=False).head(20)
        
        fig = px.bar(
            imp_df, x='Importance', y='Feature',
            orientation='h',
            title='Top 20 Feature Importances',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        return fig
    return None

def plot_model_comparison(results):
    """Create a comparison chart for all models."""
    if not results:
        return None
    
    metrics_df = pd.DataFrame({
        model_name: [
            results[model_name]['metrics']['Accuracy'],
            results[model_name]['metrics']['Precision'],
            results[model_name]['metrics']['Recall'],
            results[model_name]['metrics']['F1-Score'],
            results[model_name]['metrics']['AUC-ROC']
        ]
        for model_name in results.keys()
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']).T
    
    # Create bar chart
    fig = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df.index,
            y=metrics_df[metric],
            text=[f"{v:.3f}" for v in metrics_df[metric]],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        height=450,
        yaxis=dict(range=[0, 1], title='Score'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

# ========================
# BUSINESS RECOMMENDATIONS
# ========================

def generate_business_recommendations(df, model_results=None):
    """Generate business recommendations based on analysis."""
    recommendations = []
    
    # 1. Demographic targeting
    age_sub_rate = df.groupby(pd.cut(df['age'], bins=[0, 30, 45, 60, 100]))['y'].mean()
    best_age_group = age_sub_rate.idxmax()
    recommendations.append({
        "category": "Customer Targeting",
        "recommendation": f"Focus marketing efforts on clients aged {best_age_group} as they show the highest conversion rates.",
        "impact": "High",
        "effort": "Low"
    })
    
    # 2. Job-based targeting
    job_sub_rate = df.groupby('job')['y'].mean()
    top_jobs = job_sub_rate.nlargest(3).index.tolist()
    recommendations.append({
        "category": "Customer Segmentation",
        "recommendation": f"Prioritize clients in {', '.join(top_jobs)} job categories for targeted campaigns.",
        "impact": "High",
        "effort": "Low"
    })
    
    # 3. Call strategy
    campaign_effect = df.groupby('campaign')['y'].mean()
    optimal_campaigns = campaign_effect.idxmax()
    recommendations.append({
        "category": "Campaign Strategy",
        "recommendation": f"Limit campaign contacts to {optimal_campaigns} per client for optimal conversion rates.",
        "impact": "Medium",
        "effort": "Low"
    })
    
    # 4. Economic timing
    if 'euribor3m' in df.columns:
        euribor_effect = df.groupby(pd.cut(df['euribor3m'], bins=5))['y'].mean()
        best_euribor = euribor_effect.idxmax()
        recommendations.append({
            "category": "Timing Strategy",
            "recommendation": f"Schedule campaigns during periods with lower euribor rates for better conversion.",
            "impact": "Medium",
            "effort": "Low"
        })
    
    # 5. Model-based recommendation
    if model_results:
        best_model = max(model_results.items(), key=lambda x: x[1]['metrics']['AUC-ROC'])
        recommendations.append({
            "category": "Model Implementation",
            "recommendation": f"Deploy {best_model[0]} for production as it achieves the highest AUC-ROC ({best_model[1]['metrics']['AUC-ROC']:.3f}).",
            "impact": "High",
            "effort": "Medium"
        })
    
    return recommendations

# ========================
# STREAMLIT PAGE FUNCTIONS
# ========================

def business_understanding():
    """Render the Business Understanding section."""
    st.header(f":{theme_color}[_Business Understanding_]", divider=theme_color)
    
    options = ["Project Overview", "Research Framework", "Business Problem", "Objectives", "Risks and Assumptions", "Success Criteria"]
    selection = st.pills(None, options, default=options[0])
    
    with st.container(border=True):
        st.header(f":{theme_color}[_{selection}_]", divider=theme_color)
        
        if selection == options[0]:
            st.write_stream(stream_data(
                """
                ### About the Bank Marketing Dataset
                
                This dataset is related to **direct marketing campaigns (phone calls)** of a **Portuguese banking institution**. The marketing campaigns were based on phone calls, where often more than one contact to the same client was required to assess if the product (bank term deposit) would be subscribed.
                
                **[Data Source](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)**
                
                **[Introductory Paper](http://dx.doi.org/10.1016/j.dss.2014.03.001)**
                
                #### Business Context:
                The Portuguese banking sector faces intense competition for customer deposits. 
                Understanding which clients are likely to subscribe to term deposits allows the bank to:
                
                1. **Optimize marketing spend** by targeting high-potential clients
                2. **Improve conversion rates** through better client understanding
                3. **Enhance customer experience** by reducing unnecessary calls
                4. **Increase profitability** through higher deposit base
                """
            ))
        
        elif selection == options[1]:
            st.write_stream(stream_data(
                """
                ### Research Problem
                
                Banking institutions spend significant resources on telemarketing campaigns with relatively low conversion rates (typically 5-15%). 
                The challenge is to identify which clients are most likely to subscribe to a term deposit, enabling more efficient 
                resource allocation.
                
                ### Research Question
                
                *"What combination of client demographic characteristics, campaign interaction patterns, 
                and macroeconomic indicators best predicts term deposit subscription?"*
                
                ### Research Hypotheses
                
                1. **Demographic Impact**: Older clients (>50) and those with higher education show higher subscription rates
                2. **Campaign Strategy**: Cellular contacts and specific months yield higher conversion
                3. **Economic Context**: Lower euribor rates and higher employment rates correlate with increased subscriptions
                4. **Previous Engagement**: Clients with previous successful interactions are more likely to convert
                5. **Model Performance**: Ensemble methods will outperform simpler models
                
                ### Expected Outcomes
                
                - A predictive model with AUC-ROC > 0.85
                - Identification of top 10 most important predictors
                - Actionable business recommendations for campaign optimization
                - An interactive dashboard for marketing team use
                """
            ))
        
        elif selection == options[2]:
            st.write_stream(stream_data(
                """
                ### The Current Challenge
                
                The banking institution currently conducts broad, untargeted telemarketing campaigns 
                to promote term deposit products. This approach has several inefficiencies:
                
                - **Low Conversion Rates**: Only ~11% of contacted clients subscribe
                - **High Operational Costs**: Significant resources spent on unproductive calls
                - **Customer Frustration**: Excessive calls may lead to customer dissatisfaction
                - **Missed Opportunities**: High-potential clients may not be prioritized
                
                ### The Solution
                
                An analytics system that:
                - **Predicts** which clients are most likely to subscribe
                - **Identifies** key drivers of subscription behavior
                - **Provides** actionable insights for campaign optimization
                - **Enables** data-driven decision making
                """
            ))
        
        elif selection == options[3]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write_stream(stream_data(
                    """
                    #### General Objective
                    
                    To develop an end-to-end data analytics platform that predicts term deposit 
                    subscription likelihood and provides actionable business insights for optimizing 
                    bank telemarketing campaigns.
                    
                    #### Specific Objectives
                    
                    1. **Descriptive Analytics**: Provide a comprehensive overview of client demographics, campaign patterns, and subscription trends

                    2. **Exploratory Analysis**: Identify key patterns, relationships, and drivers of subscription behavior
                    
                    3. **Predictive Modeling**: Build and evaluate multiple classification models to predict subscription likelihood
                    
                    4. **Business Intelligence**: Generate actionable recommendations for marketing strategy optimization
                    """
                ))
            
            with col2:
                st.write_stream(stream_data(
                    """
                    #### Success Metrics
                    
                    | Metric | Target |
                    |--------|--------|
                    | Model AUC-ROC | > 0.85 |
                    | Model F1-Score | > 0.70 |
                    | Subscription Rate Increase | > 15% |
                    | Call Volume Reduction | > 30% |
                    | Marketing ROI Improvement | > 20% |
                    """
                ))
        
        elif selection == options[4]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write_stream(stream_data(
                    """
                    #### Business Risks
                    
                    | Risk | Impact | Mitigation |
                    |------|--------|------------|
                    | Model accuracy degrades over time | High | Regular model retraining |
                    | Customer behavior changes | Medium | Monitor model drift |
                    | Privacy concerns with data | High | Anonymize data |
                    | Marketing team adoption resistance | Medium | Clear value demonstration |
                    """
                ))
            
            with col2:
                st.write_stream(stream_data(
                    """
                    #### Technical Assumptions
                    
                    1. **Data Quality**: The dataset accurately represents client behavior
                    2. **Representation**: The data is representative of the current client base
                    3. **Consistency**: Client behavior patterns remain relatively stable
                    4. **Completeness**: Missing values are appropriately handled
                    5. **Independence**: Records are independent
                    """
                ))
        
        elif selection == options[5]:
            st.write_stream(stream_data(
                """
                #### Business Success Criteria
                
                1. **Conversion Improvement**: Achieve ≥15% increase in term deposit subscription rate
                2. **Efficiency Gain**: Reduce campaign call volume by ≥30%
                3. **ROI Improvement**: Increase marketing ROI by ≥20%
                4. **Customer Experience**: Reduce customer complaints about excessive calls
                
                #### Technical Success Criteria
                
                1. **Model Performance**: AUC-ROC ≥ 0.85, F1-Score ≥ 0.70
                2. **Interpretability**: Top 5 most important features identified and explained
                3. **Usability**: Dashboard is intuitive and provides clear actionable insights
                4. **Scalability**: Platform can handle data updates and model retraining
                """
            ))

def business_summary(df):
    """Render the Business Summary / Executive Dashboard."""
    st.header(f":{theme_color}[_Executive Dashboard_]", divider=theme_color)
    
    # Calculate metrics
    total_clients = len(df)
    subscribers = df['y'].sum()
    subscription_rate = (subscribers / total_clients) * 100
    
    avg_age = df['age'].mean()
    avg_duration = df['duration'].mean()
    avg_campaign = df['campaign'].mean()
    
    previously_contacted = (df['pdays'] != 999).sum()
    first_time_contacted = total_clients - previously_contacted
    
    duplicate_count = df.duplicated().sum()
    total_contacts = df['campaign'].sum()
    avg_contacts_per_sub = total_contacts / subscribers if subscribers > 0 else 0
    
    with st.container(border=True):
        st.write_stream(stream_data(
        """
        This summarises key business indicators derived from the customer marketing dataset. These metrics provide an overview of campaign performance, customer characteristics and subscription outcomes before deeper analytical exploration.
        """
        ))
        
        # Row 1: Key Metrics
        st.subheader(f":{theme_color}[_Campaign Overview_]", divider=theme_color)
        c1, c2 = st.columns(2)
        
        with c1:
            st.metric(
                "_Total Customers_",
                f":{theme_color}[_{total_clients:,}_]",
                border=True
            )
            st.metric(
                "_Subscription Rate_",
                f":{theme_color}[_{subscription_rate:.2f}%_]",
                border=True
            )
        
        with c2:
            st.metric(
                "_Subscribed Customers_",
                f":{theme_color}[_{subscribers}_]",
                border=True
            )
            st.metric(
                "_Average Age_",
                f":{theme_color}[_{avg_age:.0f}_]",
                border=True
            )
        
        # Row 2: Campaign Performance
        st.subheader(f":{theme_color}[_Campaign Performance_]", divider=theme_color)
        c3, c4 = st.columns(2)
        
        with c3:
            st.metric(
                "_Average Call Duration_",
                f":{theme_color}[_{round(avg_duration)} seconds_]",
                border=True
            )
            
            success_df = df[df["poutcome"] == "success"]
            st.metric(
                "_Success Rate_",
                f":{theme_color}[_{round((len(success_df)/len(df))*100,2)}%_]",
                border=True
            )
        
        with c4:
            st.metric(
                "_Average Campaign Contacts_",
                f":{theme_color}[_{round(avg_campaign)}_]",
                border=True
            )
            st.metric(
                "_Avg Contacts per Subscription_",
                f":{theme_color}[_{round(avg_contacts_per_sub)}_]",
                border=True
            )
        
        # Quick Insights
        st.subheader(f":{theme_color}[_Quick Business Insights_]", divider=theme_color)
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            best_age = df.groupby(pd.cut(df['age'], bins=[0,30,45,60,100]))['y'].mean().idxmax()
            st.info(f"**High Potential Segment**: Clients aged {best_age} show the highest conversion rate.")
        
        with insight_col2:
            top_job = df.groupby('job')['y'].mean().idxmax()
            st.info(f"**Best Performing Job**: {top_job.title()} clients have the highest subscription rate.")
        
        with insight_col3:
            if 'euribor3m' in df.columns:
                euribor_corr = df['euribor3m'].corr(df['y'])
                st.info(f"**Economic Indicator**: Euribor rate shows {euribor_corr:.2f} correlation with subscription rate.")

def dataset_summary(df):
    """Render the Dataset Summary section."""
    st.header(f":{theme_color}[_Dataset Summary_]", divider=theme_color)
    
    metadata = get_dataset_metadata(df)
    
    st.write_stream(stream_data(
        """
        This summary provides an overview of the data available for analysis and modeling.
        """
    ))
    
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f":{theme_color}[_{metadata['rows']:,}_]", border=True)
        
        with col2:
            st.metric("Total Columns", f":{theme_color}[_{metadata['columns']}_]", border=True)
        
        with col3:
            st.metric("Numeric Features", f":{theme_color}[_{metadata['numeric_features']}_]", border=True)
        
        with col4:
            st.metric("Categorical Features", f":{theme_color}[_{metadata['categorical_features']}_]", border=True)
        
        # Dataset Information
        with st.expander("Dataset Information", expanded=True):
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.write_stream(stream_data(
                    f"""
                    **Dataset Overview**
                    - **Source**: UCI Machine Learning Repository
                    - **Domain**: Banking / Financial Services
                    - **Learning Task**: Binary Classification
                    - **Target Variable**: Term Deposit Subscription (y)
                    - **Data Period**: May 2008 - November 2010
                    - **Total Records**: {metadata['rows']:,}
                    - **Features**: {metadata['columns']}
                    """
                ))
            
            with info_col2:
                st.write_stream(stream_data(
                    f"""
                    **Feature Composition**
                    - **Binary Features**: {metadata['binary_features']}
                    - **Categorical Features**: {metadata['categorical_features']}
                    - **Numerical Features**: {metadata['numeric_features']}
                    - **Target Distribution**: 
                        - Yes: {metadata['target_distribution'].get(1, 0):,} ({metadata['target_distribution'].get(1, 0)/metadata['rows']*100:.1f}%)
                        - No: {metadata['target_distribution'].get(0, 0):,} ({metadata['target_distribution'].get(0, 0)/metadata['rows']*100:.1f}%)
                    """
                ))

def dataset_preview(df):
    """Render the Dataset Preview section."""
    st.header(f":{theme_color}[_Dataset Preview_]", divider=theme_color)
    st.write_stream(stream_data(
        """
        Explore a sample of the dataset to understand its structure and the types of information available. You can view the first few records, a random sample, or the last few records.
        """
    ))
    
    # Controls
    col1, col2, col3 = st.columns([1, 1, 1], border=True)
    
    with col1:
        preview_type = st.selectbox(
            "Preview Type",
            ["First Records", "Random Sample", "Last Records"]
        )
    
    with col2:
        n_rows = st.slider("Number of Rows", 3, 20, 5)
    
    with col3:
        show_columns = st.segmented_control(
            "Select Columns",
            options=df.columns.tolist(),
            default=df.columns[:8].tolist(),
            selection_mode="multi"
        )
    
    # Display preview
    if show_columns:
        display_df = df[show_columns]
    else:
        display_df = df
    
    if preview_type == "First Records":
        preview_df = display_df.head(n_rows)
    elif preview_type == "Random Sample":
        preview_df = display_df.sample(min(n_rows, len(display_df)))
    else:
        preview_df = display_df.tail(n_rows)
    
    with st.container(border=True):
        st.dataframe(preview_df, use_container_width=True)
    
    st.caption(f"Showing {len(preview_df)} records of {len(df)} total records")

def data_quality(df):
    """Render the Data Quality section."""
    st.header(f":{theme_color}[_Data Quality Assessment_]", divider=theme_color)
    
    st.write_stream(stream_data(
        """
        Assessing data quality is crucial for building reliable models. This examines missing values (including 'unknown' encoding), duplicate records, and data type issues to identify potential data quality concerns.
        """
    ))
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Completeness", f":{theme_color}[_{(1 - df.isnull().sum().sum()/df.size)*100:.1f}%_]", border=True)
    
    with col2:
        dup_count = df.duplicated().sum()
        st.metric("Duplicate Rows", f":{theme_color}[_{dup_count}_]", border=True)
    
    with col3:
        unknown_count = sum(df[col].astype(str).str.contains('unknown').sum() for col in df.select_dtypes(include=['object']).columns)
        st.metric("Unknown Values", f":{theme_color}[_{unknown_count:,}_]", border=True)
    
    with col4:
        st.metric("Memory Usage", f":{theme_color}[_{df.memory_usage(deep=True).sum()/1024/1024:.2f} MB_]", border=True)
    
    # Detailed Quality Report
    with st.container(border=True):
        tab1, tab2, tab3 = st.tabs(["Missing & Unknown", "Data Types", "Feature Completeness"])
        
        with tab1:
            nulls = df.isnull().sum()
            null_cols = nulls[nulls > 0]
            
            if len(null_cols) > 0:
                st.write("**Actual Missing Values (Nulls):**")
                st.dataframe(null_cols)
            else:
                st.success("No actual null values found in the dataset.")
            
            st.write("**'Unknown' Values in Categorical Features:**")
            unknown_data = []
            for col in df.select_dtypes(include=['object']).columns:
                unknown_count = df[col].astype(str).str.contains('unknown').sum()
                if unknown_count > 0:
                    unknown_data.append({
                        'Feature': col,
                        'Unknown Count': unknown_count,
                        'Percentage': (unknown_count / len(df)) * 100
                    })
            
            if unknown_data:
                unknown_df = pd.DataFrame(unknown_data).sort_values('Unknown Count', ascending=False)
                st.dataframe(unknown_df)
                st.info(f"**Insight**: {len(unknown_data)} categorical features contain 'unknown' values, which likely represent missing data. These are encoded as a separate category for modeling.")
            else:
                st.success("No 'unknown' values found in categorical features.")
        
        with tab2:
            st.write("**Data Types Summary:**")
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            st.dataframe(dtype_counts)
            
            st.write("**Detailed Feature Data Types:**")
            dtype_df = pd.DataFrame({
                'Feature': df.columns,
                'Data Type': df.dtypes.values,
                'Unique Values': [df[col].nunique() for col in df.columns],
                'Sample Values': [str(df[col].iloc[:3].tolist()) for col in df.columns]
            })
            st.dataframe(dtype_df)
        
        with tab3:
            st.write("**Feature Completeness Report:**")
            completeness = []
            for col in df.columns:
                non_null = df[col].count()
                unique = df[col].nunique()
                completeness.append({
                    'Feature': col,
                    'Completeness': (non_null / len(df)) * 100,
                    'Unique Values': unique
                })
            
            completeness_df = pd.DataFrame(completeness).sort_values('Completeness')
            st.dataframe(completeness_df)
            
            # Overall Assessment
            st.subheader("Overall Assessment")
            
            quality_score = 100
            issues = []
            
            if len(null_cols) > 0:
                quality_score -= 10
                issues.append("Missing values detected")
            
            if len(unknown_data) > 0:
                quality_score -= 5
                issues.append(f"{len(unknown_data)} features have 'unknown' values")
            
            if dup_count > 0:
                quality_score -= 5
                issues.append(f"{dup_count} duplicate rows found")
            
            if quality_score >= 90:
                st.success(f"**Data Quality Score: {quality_score}%** - Good quality data with minor issues.")
            elif quality_score >= 70:
                st.warning(f"**Data Quality Score: {quality_score}%** - Some data quality issues to address.")
            else:
                st.error(f"**Data Quality Score: {quality_score}%** - Significant data quality issues detected.")
            
            if issues:
                st.write("**Issues Detected:**")
                for issue in issues:
                    st.write(f"- {issue}")

def features(df):
    """Render the Feature Descriptions section."""
    st.header(f":{theme_color}[_Feature Descriptions_]", divider=theme_color)
    st.write_stream(stream_data(
        """
        Understanding each feature is essential for meaningful analysis and modeling. 
        This section provides detailed descriptions of all features in the Bank Marketing dataset.
        """
    ))
    
    # Feature details
    feature_details = {
        "Feature": [
            "age", "job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "duration", "campaign",
            "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx",
            "cons.conf.idx", "euribor3m", "nr.employed", "y"
        ],
        "Category": [
            "Demographic", "Demographic", "Demographic", "Demographic",
            "Financial", "Financial", "Financial",
            "Contact", "Contact", "Contact", "Contact",
            "Campaign", "Campaign", "Campaign", "Campaign",
            "Economic", "Economic", "Economic", "Economic", "Economic",
            "Target"
        ],
        "Description": [
            "Age of client", "Type of job", "Marital status", "Highest education level",
            "Has credit in default?", "Has housing loan?", "Has personal loan?",
            "Contact communication type", "Last contact month", "Last contact day of week",
            "Last contact duration (seconds)", "Number of contacts during this campaign",
            "Days since last contact from previous campaign", "Number of contacts before this campaign",
            "Outcome of previous marketing campaign", "Employment variation rate (quarterly)",
            "Consumer price index (monthly)", "Consumer confidence index (monthly)",
            "Euribor 3 month rate (daily)", "Number of employees (quarterly)",
            "Has client subscribed a term deposit? (target)"
        ],
        "Data Type": [
            "Numeric", "Categorical", "Categorical", "Categorical",
            "Binary", "Binary", "Binary",
            "Categorical", "Categorical", "Categorical", "Numeric",
            "Numeric", "Numeric", "Numeric", "Categorical",
            "Numeric", "Numeric", "Numeric", "Numeric", "Numeric",
            "Binary"
        ],
        "Business Importance": [
            "High", "High", "Medium", "High",
            "Medium", "Medium", "Low",
            "Medium", "High", "Low", "High (excluded for prediction)",
            "High", "High", "Medium", "High",
            "High", "Medium", "Medium", "High", "Medium",
            "N/A (Target)"
        ],
    }
    
    feature_df = pd.DataFrame(feature_details)
    
    with st.container(border=True):
        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="small"),
                "Category": st.column_config.TextColumn("Category", width="small"),
                "Description": st.column_config.TextColumn("Description", width="medium"),
                "Data Type": st.column_config.TextColumn("Data Type", width="small"),
                "Business Importance": st.column_config.TextColumn("Business Importance", width="small")
            }
        )
        
        # Feature categories summary
        st.subheader(f":{theme_color}[_Feature Categories_]", divider=theme_color)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Demographic", "4", border=True)
        with col2:
            st.metric("Financial", "3", border=True)
        with col3:
            st.metric("Contact", "4", border=True)
        with col4:
            st.metric("Campaign", "4", border=True)
        with col5:
            st.metric("Economic", "5", border=True)

def descriptive_stats(df):
    """Render the Descriptive Statistics section."""
    st.subheader(f":{theme_color}[_Descriptive Statistics_]", divider=theme_color)
    st.write_stream(stream_data(
        """
        Descriptive statistics provide a summary of the central tendency, dispersion, and shape of the dataset's distribution. Understanding these statistics is fundamental for identifying patterns, outliers, and data quality issues.
        """
    ))
    
    options = ["Numerical Features", "Categorical Features", "Key Insights"]
    selection = st.pills(None, options, default="Numerical Features")
    
    with st.container(border=True):
        if selection == "Numerical Features":
            st.write_stream(stream_data(
                """
                For numerical features, we examine measures of central tendency (mean, median) and dispersion (standard deviation, quartiles) to understand the distribution of values.
                """
            ))
            st.dataframe(df.describe(), use_container_width=True)
        
        elif selection == "Categorical Features":
            st.write_stream(stream_data(
                """
                For categorical features, value counts reveal the distribution of categories within each column, helping identify dominant categories and potential imbalances.
                """
            ))
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                with st.expander(f"{col.title()}"):
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, 'Count']
                    value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(1)
                    st.dataframe(value_counts, use_container_width=True, hide_index=True)
        
        else:
            st.subheader(f":{theme_color}[_Key Statistical Insights_]", divider=theme_color)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write_stream(stream_data(
                    f"""
                    **Age Distribution**
                    - Average age: {df['age'].mean():.1f} years
                    - Median age: {df['age'].median():.1f} years
                    - Age range: {df['age'].min()} to {df['age'].max()} years
                    - Most clients are between {df['age'].quantile(0.25):.0f} and {df['age'].quantile(0.75):.0f} years old
                    """
                ))
                
                st.write_stream(stream_data(
                    f"""
                    **Call Duration**
                    - Average duration: {df['duration'].mean():.0f} seconds ({df['duration'].mean()/60:.1f} minutes)
                    - Median duration: {df['duration'].median():.0f} seconds
                    - Wide variation: {df['duration'].min()} to {df['duration'].max()} seconds
                    - 75% of calls are under {df['duration'].quantile(0.75):.0f} seconds
                    """
                ))
            
            with col2:
                st.write_stream(stream_data(
                    f"""
                    **Campaign Contacts**
                    - Average contacts: {df['campaign'].mean():.2f}
                    - Median contacts: {df['campaign'].median():.0f}
                    - Most clients: 1-3 contacts
                    - Some clients have up to {df['campaign'].max()} contacts
                    """
                ))
                
                st.write_stream(stream_data(
                    f"""
                    **Previous Contacts**
                    - {((df['pdays'] == 999).sum() / len(df) * 100):.1f}% of clients were never contacted before
                    - {((df['pdays'] != 999).sum() / len(df) * 100):.1f}% have previous contact history
                    - pdays = 999 indicates no previous contact
                    """
                ))
            
            # Subscriber vs Non-Subscriber Comparison
            st.subheader(f":{theme_color}[_Subscriber vs Non-Subscriber Comparison_]", divider=theme_color)
            
            sub_df = df[df['y'] == 1]
            non_sub_df = df[df['y'] == 0]
            
            comp_data = []
            for col in ['age', 'duration', 'campaign', 'previous']:
                comp_data.append({
                    'Feature': col,
                    'Subscribers (Mean)': sub_df[col].mean(),
                    'Non-Subscribers (Mean)': non_sub_df[col].mean(),
                    'Difference': sub_df[col].mean() - non_sub_df[col].mean(),
                    'Subscribers (Median)': sub_df[col].median(),
                    'Non-Subscribers (Median)': non_sub_df[col].median()
                })
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df.round(2), use_container_width=True)

def visualizations(df):
    """Render the Exploratory Data Analysis visualizations."""
    st.header(f":{theme_color}[_Exploratory Data Analysis_]", divider=theme_color)
    st.write_stream(stream_data("""
    Visualizations help understand data patterns, relationships, and distributions. 
    We explore the Bank Marketing dataset through three levels of analysis: 
    single variables (Univariate), relationships between two variables (Bivariate), 
    and patterns across multiple variables (Multivariate).
    """))
    
    analysis_type = st.pills(
        "Select Analysis Type",
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"],
        default="Univariate Analysis"
    )
    
    with st.container(border=True):
        if analysis_type == "Univariate Analysis":
            univariate_analysis(df)
        elif analysis_type == "Bivariate Analysis":
            bivariate_analysis(df)
        else:
            multivariate_analysis(df)

def univariate_analysis(df):
    """Render univariate analysis visualizations."""
    st.subheader(f":{theme_color}[_Univariate Analysis_]", divider=theme_color)
    st.write_stream(stream_data("""
    Univariate analysis examines individual variables to understand their distribution, central tendency, and dispersion. These visualizations provide insights into the composition of the client base and campaign characteristics.
    """))
    
    options = [
        "Age Distribution",
        "Job Distribution",
        "Education Distribution",
        "Marital Status",
        "Subscription Distribution",
        "Duration Distribution",
        "Campaign Contacts"
    ]
    
    selection = st.segmented_control("Select Variable", options, default=options[0])
    
    with st.container(border=True):
        if selection == "Age Distribution":
            st.caption("**Distribution of Client Ages**")
            fig = plot_age_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Observation**: The age distribution is roughly normal with a peak around 30-50 years.
            
            **Insight**: Most clients are middle-aged, which may influence campaign targeting strategies.
            """)
        
        elif selection == "Job Distribution":
            st.caption("**Distribution of Client Job Types**")
            fig = plot_job_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Observation**: Blue-collar and administrative workers are the most common job types.
            
            **Insight**: Campaign messages should be tailored to the dominant job categories.
            """)
        
        elif selection == "Education Distribution":
            st.caption("**Distribution of Education Levels**")
            edu_counts = df['education'].value_counts().reset_index()
            edu_counts.columns = ['Education', 'Count']
            fig = px.bar(edu_counts, x='Education', y='Count', title='Education Level Distribution', color='Count', color_continuous_scale='Greens')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Observation**: Most clients have high school or university-level education.
            
            **Insight**: Educational background may correlate with financial literacy and product understanding.
            """)
        
        elif selection == "Marital Status":
            st.caption("**Distribution of Marital Status**")
            marital_counts = df['marital'].value_counts().reset_index()
            marital_counts.columns = ['Status', 'Count']
            fig = px.pie(marital_counts, values='Count', names='Status', title='Marital Status Distribution', color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Observation**: Married clients represent the majority.
            
            **Insight**: Married clients may have different financial needs and decision-making processes.
            """)
        
        elif selection == "Subscription Distribution":
            st.caption("**Term Deposit Subscription Distribution**")
            fig = plot_subscription_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            st.warning("""
            **Observation**: Only ~11% of clients subscribed to the term deposit.
            
            **Insight**: The dataset is highly imbalanced, which requires special handling during modeling.
            """)
        
        elif selection == "Duration Distribution":
            st.caption("**Distribution of Call Duration**")
            fig = px.histogram(df, x='duration', nbins=50, title='Call Duration Distribution', labels={'duration': 'Duration (seconds)'}, color_discrete_sequence=['#ff6f00'])
            fig.add_vline(x=df['duration'].mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {df['duration'].mean():.0f}s")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Observation**: Call duration is heavily right-skewed with most calls under 300 seconds.
            
            **Insight**: Short calls may indicate low engagement; longer calls may show higher interest.
            """)
        
        else:  # Campaign Contacts
            st.caption("**Number of Campaign Contacts**")
            fig = px.histogram(df, x='campaign', nbins=20, title='Campaign Contact Distribution', labels={'campaign': 'Number of Contacts'}, color_discrete_sequence=['#00695c'])
            fig.add_vline(x=df['campaign'].mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {df['campaign'].mean():.1f}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Observation**: Most clients are contacted 1-2 times during the campaign.
            
            **Insight**: Excessive contacts (>5) may lead to diminishing returns or customer annoyance.
            """)

def bivariate_analysis(df):
    """Render bivariate analysis visualizations."""
    st.subheader(f":{theme_color}[_Bivariate Analysis_]", divider=theme_color)
    st.write_stream(stream_data("""
    Bivariate analysis examines relationships between two variables to identify patterns, correlations, and potential predictors of subscription behavior.
    """))
    
    options = [
        "Duration vs Subscription",
        "Age vs Subscription",
        "Campaign vs Subscription",
        "Job vs Subscription",
        "Education vs Subscription",
        "Month vs Subscription",
        "Previous Outcome vs Subscription"
    ]
    
    selection = st.segmented_control("Select Relationship", options, default=options[0])
    
    with st.container(border=True):
        if selection == "Duration vs Subscription":
            st.caption("**Call Duration by Subscription Status**")
            fig = plot_duration_vs_subscription(df)
            st.plotly_chart(fig, use_container_width=True)
            st.success("""
            **Key Finding**: Clients who subscribe have significantly longer call durations.
            
            **Business Implication**: Engaging conversations lead to conversions. Marketing teams should focus on call quality.
            
            **Note**: Duration is not available before the call, so it should be excluded from prediction models.
            """)
        
        elif selection == "Age vs Subscription":
            st.caption("**Age by Subscription Status**")
            fig = plot_age_vs_subscription(df)
            st.plotly_chart(fig, use_container_width=True)
            st.success("""
            **Key Finding**: Subscribers tend to be slightly older on average.
            
            **Business Implication**: Target marketing towards older clients who may have more savings.
            
            **Action**: Create age-based segmentation for campaign strategy.
            """)
        
        elif selection == "Campaign vs Subscription":
            st.caption("**Campaign Contacts by Subscription Status**")
            plot_df = df.copy()
            plot_df['Subscribed'] = plot_df['y'].map({1: 'Yes', 0: 'No'})
            fig = px.box(plot_df, x='Subscribed', y='campaign', title='Campaign Contacts by Subscription Status',
                        color='Subscribed', color_discrete_map={'Yes': '#2e7d32', 'No': '#c62828'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.success("""
            **Key Finding**: Optimal conversion occurs with 2-3 campaign contacts.
            
            **Business Implication**: Avoid excessive follow-ups to prevent customer fatigue.
            
            **Action**: Limit campaign contacts to 3 per client for optimal ROI.
            """)
        
        elif selection == "Job vs Subscription":
            st.caption("**Subscription Rate by Job Type**")
            job_sub = df.groupby('job')['y'].mean().reset_index()
            job_sub.columns = ['Job', 'Subscription Rate']
            job_sub['Subscription Rate'] = job_sub['Subscription Rate'] * 100
            job_sub = job_sub.sort_values('Subscription Rate', ascending=True)
            fig = px.bar(job_sub, x='Subscription Rate', y='Job', orientation='h',
                        title='Subscription Rate by Job Type', color='Subscription Rate',
                        color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.success("""
            **Key Finding**: Retired and student clients show highest subscription rates.
            
            **Business Implication**: Prioritize these job categories in marketing campaigns.
            
            **Action**: Develop targeted messaging for high-converting job segments.
            """)
        
        elif selection == "Education vs Subscription":
            st.caption("**Subscription Rate by Education Level**")
            edu_sub = df.groupby('education')['y'].mean().reset_index()
            edu_sub.columns = ['Education', 'Subscription Rate']
            edu_sub['Subscription Rate'] = edu_sub['Subscription Rate'] * 100
            edu_sub = edu_sub.sort_values('Subscription Rate', ascending=True)
            fig = px.bar(edu_sub, x='Subscription Rate', y='Education', orientation='h',
                        title='Subscription Rate by Education Level', color='Subscription Rate',
                        color_continuous_scale='Greens')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.success("""
            **Key Finding**: University graduates show higher subscription rates.
            
            **Business Implication**: Educational content may resonate better with higher-educated clients.
            
            **Action**: Tailor communication style based on education level.
            """)
        
        elif selection == "Month vs Subscription":
            st.caption("**Campaign Activity and Subscription by Month**")
            month_data = df.groupby('month').agg({
                'y': ['mean', 'count']
            }).reset_index()
            month_data.columns = ['Month', 'Subscription Rate', 'Contact Count']
            month_data['Subscription Rate'] = month_data['Subscription Rate'] * 100
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=month_data['Month'], y=month_data['Contact Count'], 
                                name='Contacts', marker_color='#1a237e'), secondary_y=False)
            fig.add_trace(go.Scatter(x=month_data['Month'], y=month_data['Subscription Rate'], 
                                    name='Subscription Rate %', mode='lines+markers', 
                                    line=dict(color='#ff6f00', width=3)), secondary_y=True)
            fig.update_layout(title='Campaign Activity and Subscription Rate by Month', 
                            height=400, xaxis_title='Month')
            fig.update_yaxes(title_text='Number of Contacts', secondary_y=False)
            fig.update_yaxes(title_text='Subscription Rate (%)', secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            st.success("""
            **Key Finding**: May and July show highest campaign activity; September shows highest conversion.
            
            **Business Implication**: Schedule more campaigns during high-conversion months.
            
            **Action**: Focus marketing resources on September campaigns.
            """)
        
        else:  # Previous Outcome vs Subscription
            st.caption("**Subscription Rate by Previous Campaign Outcome**")
            poutcome_sub = df.groupby('poutcome')['y'].mean().reset_index()
            poutcome_sub.columns = ['Previous Outcome', 'Subscription Rate']
            poutcome_sub['Subscription Rate'] = poutcome_sub['Subscription Rate'] * 100
            fig = px.bar(poutcome_sub, x='Previous Outcome', y='Subscription Rate',
                        title='Subscription Rate by Previous Campaign Outcome',
                        color='Subscription Rate', color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.success("""
            **Key Finding**: Clients with previous campaign success are much more likely to convert.
            
            **Business Implication**: Follow up with clients who have had positive past interactions.
            
            **Action**: Prioritize clients with 'success' poutcome for current campaigns.
            """)

def multivariate_analysis(df):
    """Render multivariate analysis visualizations."""
    st.subheader(f":{theme_color}[_Multivariate Analysis_]", divider=theme_color)
    st.write_stream(stream_data("""
    Multivariate analysis examines relationships across multiple variables simultaneously, revealing complex patterns and interactions that may not be visible in univariate or bivariate analysis.
    """))
    
    options = [
        "Correlation Heatmap",
        "Economic Trends",
        "Age-Duration-Subscription Analysis",
        "Feature Relationships"
    ]
    
    selection = st.segmented_control("Select Analysis", options, default=options[0])
    
    with st.container(border=True):
        if selection == "Correlation Heatmap":
            st.caption("**Feature Correlation Heatmap**")
            fig = plot_correlation_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Key Observations**:
            - Strong correlations among economic indicators (euribor3m, emp.var.rate)
            - Duration shows moderate correlation with subscription
            - Age and duration have weak positive correlation
            - Campaign contacts and previous contacts are somewhat related
            
            **Implications**: Economic indicators capture similar information - consider feature reduction.
            """)
        
        elif selection == "Economic Trends":
            st.caption("**Economic Indicators Trends by Month**")
            fig = plot_economic_trends(df)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Key Observations**:
            - Euribor rates declined significantly from 2008 to 2010
            - Employment rates fluctuated during the period
            - Consumer confidence showed variability
            
            **Implications**: Economic conditions varied significantly during the data period, 
            which may have influenced subscription behavior.
            """)
        
        elif selection == "Age-Duration-Subscription Analysis":
            st.caption("**Age, Duration, and Subscription Relationship**")
            plot_df = df.copy()
            plot_df['Subscribed'] = plot_df['y'].map({1: 'Yes', 0: 'No'})
            plot_df['Age Group'] = pd.cut(plot_df['age'], bins=[0, 30, 45, 60, 100], labels=['<30', '30-45', '45-60', '>60'])
            
            fig = px.scatter(
                plot_df, x='age', y='duration',
                color='Subscribed',
                title='Age vs Duration colored by Subscription Status',
                labels={'age': 'Age (years)', 'duration': 'Duration (seconds)'},
                color_discrete_map={'Yes': '#2e7d32', 'No': '#c62828'},
                opacity=0.6,
                facet_col='Age Group'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Key Observations**:
            - Across all age groups, longer duration correlates with subscription
            - Older age groups (45-60, >60) show higher subscription rates
            - The pattern holds consistently across age segments
            
            **Implications**: Engagement (duration) is a strong indicator of conversion across all demographics.
            """)
        
        else:  # Feature Relationships
            st.caption("**Key Feature Relationships**")
            features = ['age', 'duration', 'campaign', 'euribor3m', 'y']
            plot_df = df[features].copy()
            plot_df['y'] = plot_df['y'].map({1: 'Subscribed', 0: 'Not Subscribed'})
            
            fig = px.scatter_matrix(
                plot_df,
                dimensions=['age', 'duration', 'campaign', 'euribor3m'],
                color='y',
                title='Feature Relationships Matrix',
                color_discrete_map={'Subscribed': '#2e7d32', 'Not Subscribed': '#c62828'},
                opacity=0.5
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Key Observations**:
            - Duration shows the strongest visual separation between subscribers and non-subscribers
            - Age and euribor rate show moderate separation
            - Campaign contacts show some overlap
            
            **Implications**: Duration, age, and economic conditions are key discriminators for subscription.
            """)

def insights(df, model_results=None):
    """Render the Insights section."""
    st.header(f":{theme_color}[_Insights from Analysis_]", divider=theme_color)
    st.write_stream(stream_data("""
    Based on our comprehensive analysis, we've identified several key patterns and insights that inform our understanding of the bank marketing campaign and guide our modeling strategy.
    """))
    
    tabs = st.tabs(["Demographics", "Campaign Strategy", "Economic Impact", "Model Insights"])
    
    with tabs[0]:
        with st.container(border=True):
            st.subheader(f":{theme_color}[_Demographic Insights_]", divider=theme_color)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write_stream(stream_data(f"""
                **Age Impact**
                - Clients aged **45-60** show the highest subscription rate
                - Average age of subscribers: **{df[df['y']==1]['age'].mean():.1f}** vs non-subscribers: **{df[df['y']==0]['age'].mean():.1f}**
                - Age is a **moderate predictor** of subscription
                
                **Job Categories**
                - **{df.groupby('job')['y'].mean().idxmax().title()}** clients are most likely to subscribe
                - **{df.groupby('job')['y'].mean().idxmin().title()}** clients show lowest conversion
                - Target high-converting job categories for better ROI
                """))
            
            with col2:
                st.write_stream(stream_data(f"""
                **Education Impact**
                - Higher education correlates with higher subscription rates
                - University graduates show **{df[df['education']=='university.degree']['y'].mean()*100:.1f}%** conversion
                - Basic education levels show **{df[df['education'].str.contains('basic')]['y'].mean()*100:.1f}%** conversion
                
                **Marital Status**
                - **Married** clients have highest conversion rate: **{df[df['marital']=='married']['y'].mean()*100:.1f}%**
                - **Single** clients show lower conversion: **{df[df['marital']=='single']['y'].mean()*100:.1f}%**
                - Marital status is a **significant predictor**
                """))
            
            st.success("""
            **Actionable Recommendation**: 
            
            Focus marketing campaigns on clients aged 45-60 with university education, 
            particularly those in management, retired, or student roles. 
            Develop targeted messaging for these segments.
            """)
    
    with tabs[1]:
        with st.container(border=True):
            st.subheader(f":{theme_color}[_Campaign Strategy Insights_]", divider=theme_color)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write_stream(stream_data(f"""
                **Call Duration**
                - Subscribers have **{df[df['y']==1]['duration'].mean():.0f}s** average call duration
                - Non-subscribers have only **{df[df['y']==0]['duration'].mean():.0f}s**
                - Duration is the **strongest individual predictor** (but excluded from production models)
                
                **Contact Method**
                - **Cellular** contacts show higher conversion
                - Recommend prioritizing cellular for first contact
                """))
            
            with col2:
                st.write_stream(stream_data(f"""
                **Campaign Intensity**
                - Optimal contacts: **{df.groupby('campaign')['y'].mean().idxmax()}**
                - Excessive contacts show **diminishing returns**
                - Campaign intensity is a **valuable predictor**
                
                **Timing**
                - **September** shows highest conversion: **{df[df['month']=='sep']['y'].mean()*100:.1f}%**
                - **May** has highest campaign activity but lower conversion
                - **Friday** contacts show slightly better results
                """))
            
            st.success("""
            **Actionable Recommendation**: 
            
            Limit campaign contacts to 3 per client, prioritize cellular calls, 
            and schedule campaigns during high-conversion months (especially September). 
            Focus on call quality to increase engagement duration.
            """)
    
    with tabs[2]:
        with st.container(border=True):
            st.subheader(f":{theme_color}[_Economic Impact Insights_]", divider=theme_color)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write_stream(stream_data(f"""
                **Economic Indicators Correlation**
                - **Euribor 3M Rate**: **{df['euribor3m'].corr(df['y']):.2f}** correlation with subscription
                - **Employment Variation Rate**: **{df['emp.var.rate'].corr(df['y']):.2f}** correlation
                - **Consumer Confidence**: **{df['cons.conf.idx'].corr(df['y']):.2f}** correlation
                - Lower rates correlate with **higher subscription likelihood**
                """))
            
            with col2:
                st.write_stream(stream_data("""
                **Economic Trends**
                - Economic indicators have **moderate to strong correlation** with subscription
                - The period (2008-2010) saw significant economic changes
                - **Lower euribor rates** during 2009-2010 aligned with increased conversions
                - Economic context is **essential for realistic modeling**
                
                **Note**: Economic indicators are highly correlated with each other
                """))
            
            st.success("""
            **Actionable Recommendation**: 
            
            Monitor economic indicators when planning campaigns. 
            Lower euribor rates and higher employment rates suggest favorable conditions for term deposit marketing. Consider macro-economic factors in campaign timing.
            """)
    
    with tabs[3]:
        with st.container(border=True):
            st.subheader(f":{theme_color}[_Model & Data Insights_]", divider=theme_color)
            
            st.write_stream(stream_data(f"""
            **Data Characteristics**
            - **Class Imbalance**: Only **{df['y'].mean()*100:.1f}%** positive cases
            - Requires special handling in modeling
            - **Duration** is highly predictive but **not available before calls**
            
            **Model Performance Observations**
            - Ensemble methods expected to perform best
            - Economic indicators contribute significantly to predictive power
            - Previous campaign outcome is a **strong predictor**
            - Feature importance analysis helps identify key drivers
            """))
            
            st.info("""
            **Overall Conclusion**: 
            
            The Bank Marketing dataset provides rich information for predicting term deposit subscriptions. 
            
            Key predictors include demographic factors (age, job, education), campaign characteristics (duration, intensity, timing), and economic context. 
            
            A combination of these factors can effectively identify high-potential clients, enabling more efficient and effective marketing campaigns.
            """)

def modeling(df):
    """Render the Machine Learning section."""
    st.header(f":{theme_color}[_Machine Learning Modeling_]", divider=theme_color)
    st.write_stream(stream_data("""
    This section implements multiple classification models to predict term deposit subscription. 
    Models are evaluated using appropriate metrics for imbalanced classification, and the best performing model is identified for deployment consideration.
    """))
    
    # Prepare data
    with st.spinner("Preparing data for modeling..."):
        X, y, preprocessor = prepare_data_for_modeling(df)
    
    # Display prepared data info
    with st.expander("Data Preparation Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features", X.shape[1], border=True)
        with col2:
            st.metric("Target Classes", f"{y.nunique()}", border=True)
        with col3:
            st.metric("Positive Rate", f"{y.mean()*100:.1f}%", border=True)
        
        st.write("**Preprocessing Pipeline:**")
        st.code("""
        Numerical Features: StandardScaler
        Categorical Features: OneHotEncoder
        Binary Features: Passthrough
        Class Imbalance: Manual oversampling of minority class
        """)
    
    with st.container(border=True):
        st.subheader(f":{theme_color}[_Model Training_]", divider=theme_color)
        
        # Initialize session state
        if 'training_complete' not in st.session_state:
            st.session_state['training_complete'] = False
        if 'model_results' not in st.session_state:
            st.session_state['model_results'] = None
        
        selected_models = st.multiselect(
            "Select Models to Train",
            list(get_models().keys()),
            default=["Logistic Regression", "Random Forest", "Decision Tree"]
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            train_button = st.button("Train Models", use_container_width=True)
        with col2:
            if st.session_state['training_complete']:
                st.success("Selected models trained successfully!")
        
        if train_button:
            st.session_state['training_complete'] = False
            
            # Train models with progress
            results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}... ({idx+1}/{len(selected_models)})")
                
                pipeline, X_train, X_test, y_train, y_test, _, _ = train_model(
                    X, y, preprocessor, model_name
                )
                
                metrics = evaluate_model(pipeline, X_test, y_test)
                
                results[model_name] = {
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred": metrics["Predictions"],
                    "y_proba": metrics["Probabilities"]
                }
                
                progress_bar.progress((idx + 1) / len(selected_models))
            
            status_text.text("All models trained successfully!")
            progress_bar.empty()
            
            st.session_state['model_results'] = results
            st.session_state['training_complete'] = True
            st.rerun()
    
    # Display results if available
    if st.session_state['training_complete'] and st.session_state['model_results']:
        model_results_display(st.session_state['model_results'])
        st.session_state['model_results_for_recs'] = st.session_state['model_results']

def model_results_display(results):
    """Render model evaluation results."""
    st.subheader(f":{theme_color}[_Model Evaluation Results_]", divider=theme_color)
    
    # Model Comparison Chart
    fig_compare = plot_model_comparison(results)
    if fig_compare:
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # Create metrics comparison table
    metrics_df = pd.DataFrame({
        model_name: [
            results[model_name]['metrics']['Accuracy'],
            results[model_name]['metrics']['Precision'],
            results[model_name]['metrics']['Recall'],
            results[model_name]['metrics']['F1-Score'],
            results[model_name]['metrics']['AUC-ROC']
        ]
        for model_name in results.keys()
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']).T
    
    metrics_df['Rank (by AUC-ROC)'] = metrics_df['AUC-ROC'].rank(ascending=False).astype(int)
    metrics_df = metrics_df.sort_values('AUC-ROC', ascending=False)
    
    st.dataframe(metrics_df.round(3), use_container_width=True)
    
    # Highlight best model
    best_model = metrics_df.index[0]
    st.success(f"**Best Model: {best_model}** with AUC-ROC: {metrics_df.loc[best_model, 'AUC-ROC']:.3f}")
    
    # Save model button
    if st.button(f"Save Best Model ({best_model})"):
        save_msg = save_model(results[best_model]['pipeline'], best_model)
        st.info(save_msg)
    
    # Detailed evaluation for each model
    st.subheader(f":{theme_color}[_Detailed Model Evaluation_]", divider=theme_color)
    
    model_tabs = st.tabs(list(results.keys()))
    
    for idx, (model_name, result) in enumerate(results.items()):
        with model_tabs[idx]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f":{theme_color}[_{result['metrics']['Accuracy']:.3f}_]", border=True)
                st.metric("Precision", f":{theme_color}[_{result['metrics']['Precision']:.3f}_]", border=True)
                st.metric("Recall", f":{theme_color}[_{result['metrics']['Recall']:.3f}_]", border=True)
                st.metric("F1-Score", f":{theme_color}[_{result['metrics']['F1-Score']:.3f}_]", border=True)
                st.metric("AUC-ROC", f":{theme_color}[_{result['metrics']['AUC-ROC']:.3f}_]", border=True)
            
            with col2:
                fig_cm = plot_confusion_matrix(result['y_test'], result['y_pred'])
                st.plotly_chart(fig_cm, use_container_width=True)
            
                fig_roc = plot_roc_curve(result['y_test'], result['y_proba'])
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with st.expander("Classification Report"):
                report = classification_report(result['y_test'], result['y_pred'], output_dict=True)
                report_df = pd.DataFrame(report).T
                st.dataframe(report_df.round(3), use_container_width=True)
            
            if hasattr(result['pipeline'].named_steps['classifier'], 'feature_importances_'):
                with st.expander("Feature Importance"):
                    feature_names = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
                    fig_fi = plot_feature_importance(result['pipeline'], feature_names)
                    if fig_fi:
                        st.plotly_chart(fig_fi, use_container_width=True)

def business_recommendations(df, model_results=None):
    """Render Business Recommendations section."""
    st.header(f":{theme_color}[_Business Recommendations_]", divider=theme_color)
    st.write_stream(stream_data("""
    Based on our comprehensive analysis and modeling results, we've developed 
    actionable business recommendations to optimize marketing campaigns and 
    improve conversion rates.
    """))
    
    # Generate recommendations
    recommendations = generate_business_recommendations(df, model_results)
    
    for rec in recommendations:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{rec['recommendation']}**")
            with col2:
                st.markdown(f"**Impact**: {rec['impact']}")
            with col3:
                st.markdown(f"**Effort**: {rec['effort']}")
    
    # Additional recommendations based on insights
    with st.expander("Detailed Recommendations"):
        st.subheader("1. Customer Targeting")
        st.markdown("""
        - **Priority Segments**: Focus on clients aged 45-60, especially those with university education
        - **Job Categories**: Prioritize management, retired, and student clients
        - **Financial Profile**: Target clients with no existing housing or personal loans
        - **Previous Engagement**: Follow up with clients who had successful previous interactions
        """)
        
        st.subheader("2. Campaign Optimization")
        st.markdown("""
        - **Call Duration**: Focus on call quality and engagement rather than call quantity
        - **Contact Limit**: Limit campaign contacts to 3 per client
        - **Timing**: Schedule campaigns during September and Friday for best results
        - **Channel**: Prioritize cellular over telephone contacts
        """)
        
        st.subheader("3. Economic Timing")
        st.markdown("""
        - **Monitor Indicators**: Track euribor rates and employment trends
        - **Campaign Planning**: Launch campaigns when economic conditions are favorable
        - **Seasonal Strategy**: Consider monthly and quarterly patterns in planning
        """)
        
        st.subheader("4. Model Implementation")
        st.markdown("""
        - **Production Model**: Deploy the best performing model for client scoring
        - **Integration**: Integrate with CRM for real-time scoring
        - **Monitoring**: Track model performance monthly to detect drift
        - **Retraining**: Plan quarterly model retraining with new data
        """)

def conclusion(df, model_results=None):
    """Render the Conclusion section."""
    st.header(f":{theme_color}[_Conclusion_]", divider=theme_color)
    st.write_stream(stream_data("""
    This comprehensive analysis of the Bank Marketing dataset has provided valuable 
    insights into the factors that influence term deposit subscription and delivered 
    predictive models to support marketing decision-making.
    """))
    
    col1, col2 = st.columns(2, border=True)
    
    with col1:
        st.subheader(f":{theme_color}[_Key Achievements_]", divider=theme_color)
        st.markdown("""
        - **Complete CRISP-DM Implementation**: Followed the full lifecycle
        - **Comprehensive EDA**: Identified key demographic, campaign, and economic predictors
        - **Multiple Models**: Trained and evaluated multiple classification models
        - **Actionable Insights**: Generated business recommendations for campaign optimization
        - **Interactive Platform**: Created a user-friendly dashboard for stakeholders
        """)
    
    with col2:
        st.subheader(f":{theme_color}[_Business Value_]", divider=theme_color)
        st.markdown("""
        - **Efficiency**: Potential to reduce campaign calls by 30% through better targeting
        - **Effectiveness**: Expected 15-20% improvement in conversion rates
        - **ROI**: Significant return on investment through optimized resource allocation
        - **Insights**: Deep understanding of customer behavior and market dynamics
        - **Competitive Advantage**: Data-driven decision making capability
        """)
    
    # Add model results summary if available
    if model_results:
        st.subheader(f":{theme_color}[_Model Performance Summary_]", divider=theme_color)
        best_model = max(model_results.items(), key=lambda x: x[1]['metrics']['AUC-ROC'])
        st.success(f"**Best Performing Model**: {best_model[0]} (AUC-ROC: {best_model[1]['metrics']['AUC-ROC']:.3f})")
        
        # Show all model rankings
        rankings = []
        for idx, (name, result) in enumerate(sorted(model_results.items(), key=lambda x: x[1]['metrics']['AUC-ROC'], reverse=True)):
            rankings.append(f"{idx+1}. {name}: {result['metrics']['AUC-ROC']:.3f}")
        st.write("**Model Rankings (by AUC-ROC):**")
        for rank in rankings:
            st.write(f"- {rank}")
    
    with st.container(border=True):
        st.subheader(f":{theme_color}[_Key Recommendations Summary_]", divider=theme_color)
        
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            st.info("**Targeting**\nFocus on clients aged 45-60 with university education in management/retired roles")
        
        with summary_cols[1]:
            st.info("**Campaign**\nLimit contacts to 3 per client, prioritize cellular, schedule for September")
        
        with summary_cols[2]:
            st.info("**Implementation**\nDeploy model, monitor performance, retrain quarterly")


# ================
# MAIN APPLICATION
# =================

def main():
    """Main application entry point."""
    
    # Load data
    df = load_dataset()
    
    if df is None:
        st.stop()
    
    # Store in session state
    if 'df' not in st.session_state:
        st.session_state['df'] = df
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("Bank Marketing")
        st.caption("Analytics System")
        container = st.container(border=True, vertical_alignment="top", height=450)
        container.header("Navigate the Dashboard")
        menu = container.selectbox(
            "",
            [
                "Home",
                "Data Understanding",
                "Exploratory Data Analysis (EDA)",
                "Insights",
                "Modeling",
                "Recommendations",
                "Conclusion"
            ]
        )
    
    # Render selected section
    if menu == "Home":
        tab1, tab2 = st.tabs(["Business Understanding", "Business Summary"])
        with tab1:
            business_understanding()
        with tab2:
            business_summary(df)
    
    elif menu == "Data Understanding":
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Dataset Summary", "Dataset Preview", "Data Quality", 
            "Features", "Descriptive Stats"
        ])
        with tab1:
            dataset_summary(df)
        with tab2:
            dataset_preview(df)
        with tab3:
            data_quality(df)
        with tab4:
            features(df)
        with tab5:
            descriptive_stats(df)
    
    elif menu == "Exploratory Data Analysis (EDA)":
        visualizations(df)
    
    elif menu == "Insights":
        insights(df, st.session_state.get('model_results_for_recs'))
    
    elif menu == "Modeling":
        modeling(df)
    
    elif menu == "Recommendations":
        business_recommendations(df, st.session_state.get('model_results_for_recs'))
    
    elif menu == "Conclusion":
        conclusion(df, st.session_state.get('model_results_for_recs'))

# ===============
# RUN APPLICATION
# ===============
if __name__ == "__main__":
    main()