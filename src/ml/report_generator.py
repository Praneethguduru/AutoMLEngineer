import os
import shutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid GUI threads issues
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from ml.explainability import generate_shap_values
import shap

# Set premium styling
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

def _save_dual_paths(fig, dataset_name, suffix):
    """Helper to save a figure to both the dataset-prefixed and generic paths."""
    os.makedirs("outputs", exist_ok=True)
    path_prefixed = f"outputs/{dataset_name}_{suffix}"
    path_generic = f"outputs/{suffix}"
    
    # Save prefixed version
    fig.savefig(path_prefixed, bbox_inches='tight', dpi=300)
    # Copy to generic version
    shutil.copy2(path_prefixed, path_generic)
    print(f"Saved charts to: {path_prefixed} and {path_generic}")

def generate_model_comparison_chart(model_results, problem_type, dataset_name):
    """Generates a professional bar chart comparing trained models."""
    metric_name = "f1_score" if problem_type == "classification" else "r2"
    metric_label = "F1-Score (Macro/Binary)" if problem_type == "classification" else "R² (Coefficient of Determination)"
    
    # Extract data
    models = []
    scores = []
    for model_name, res in model_results.items():
        score = res["metrics"].get(metric_name, 0.0)
        models.append(model_name)
        scores.append(score)
        
    if not models:
        return
        
    # Sort descending
    sorted_idx = np.argsort(scores)[::-1]
    models = [models[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Theme colors: highlight the best model in premium steel blue/teal, others in soft gray
    best_color = "#1f77b4" if problem_type == "classification" else "#2ca02c"
    other_color = "#d3d3d3"
    colors = [best_color] + [other_color] * (len(models) - 1)
    
    bars = ax.barh(models[::-1], scores[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    
    # Visual polishing
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Add titles & labels
    ax.set_title(f"Model Performance Comparison ({dataset_name})", pad=20, weight='bold')
    ax.set_xlabel(metric_label, labelpad=10)
    
    # Add values on/next to the bars
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.4f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),  # 5 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='semibold',
                    color='#333333')
                    
    # Adjust boundaries
    if problem_type == "classification":
        ax.set_xlim(0, 1.05)
    else:
        # For regression R2, handle negative values gracefully
        min_val = min(0.0, min(scores) * 1.1)
        max_val = max(1.0, max(scores) * 1.1)
        ax.set_xlim(min_val, max_val)

    plt.tight_layout()
    _save_dual_paths(fig, dataset_name, "model_comparison.png")
    plt.close(fig)

def generate_feature_importance_chart(feature_importance, dataset_name):
    """Generates a horizontal bar chart for the top 10 feature importances."""
    if not feature_importance:
        return
        
    # Get top 10 features
    top_n = list(feature_importance.items())[:10]
    features = [item[0] for item in top_n]
    scores = [item[1] for item in top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors: gradient of teal/blue
    colors = sns.color_range = sns.color_palette("viridis_r", len(features))
    
    bars = ax.barh(features[::-1], scores[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    
    # Visual polishing
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    ax.set_title(f"Top 10 Feature Importances ({dataset_name})", pad=20, weight='bold')
    ax.set_xlabel("Relative Importance Score", labelpad=10)
    
    # Add values
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.4f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='semibold',
                    color='#333333')
                    
    # Adjust boundaries
    max_score = max(scores) if scores else 1.0
    ax.set_xlim(0, max_score * 1.15)
    
    plt.tight_layout()
    _save_dual_paths(fig, dataset_name, "feature_importance.png")
    plt.close(fig)

def generate_confusion_matrix_chart(y_test, predictions, dataset_name):
    """Generates a heatmap visualization of the Confusion Matrix (classification only)."""
    cm = confusion_matrix(y_test, predictions)
    classes = np.unique(np.concatenate([y_test, predictions]))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Heatmap with cool-toned blues colormap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={"size": 12, "weight": "bold"})
                
    ax.set_title(f"Confusion Matrix Heatmap ({dataset_name})", pad=20, weight='bold')
    ax.set_xlabel("Predicted Labels", labelpad=10, weight='semibold')
    ax.set_ylabel("True Labels", labelpad=10, weight='semibold')
    
    plt.tight_layout()
    _save_dual_paths(fig, dataset_name, "confusion_matrix.png")
    plt.close(fig)

def generate_shap_charts(df, target_column, feature_report, problem_type, dataset_name):
    """Generates and saves SHAP summary and SHAP bar charts from explainability module."""
    try:
        # Run explainability runner
        shap_res = generate_shap_values(df, target_column, feature_report, problem_type)
        
        shap_values = shap_res["shap_values"]
        X_preprocessed = shap_res["preprocessed_features"]
        feature_names = shap_res["feature_names"]
        
        # 1. Generate SHAP Summary (Beeswarm style)
        fig_summary = plt.figure(figsize=(10, 6))
        # Handle different SHAP formats safely
        if isinstance(shap_values, list):
            # If list of 2 arrays, pick class 1 (standard positive class for beeswarm)
            if len(shap_values) == 2:
                shap.summary_plot(shap_values[1], X_preprocessed, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_preprocessed, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_preprocessed, feature_names=feature_names, show=False)
            
        plt.title(f"SHAP Feature Impact Summary ({dataset_name})", pad=20, weight='bold', fontsize=14)
        _save_dual_paths(fig_summary, dataset_name, "shap_summary.png")
        plt.close(fig_summary)
        
        # 2. Generate SHAP Bar plot (mean absolute SHAP)
        fig_bar = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_preprocessed, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance (Mean Absolute Value) ({dataset_name})", pad=20, weight='bold', fontsize=14)
        _save_dual_paths(fig_bar, dataset_name, "shap_bar.png")
        plt.close(fig_bar)
        
    except Exception as e:
        print(f"Gracefully skipping SHAP visualization generation: {e}")
