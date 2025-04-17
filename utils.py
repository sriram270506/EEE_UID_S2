import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st
import base64
from io import BytesIO
import tensorflow as tf
import zipfile
import os
import json
import time
from matplotlib.colors import LinearSegmentedColormap

def set_tensorflow_memory_growth():
    """Configure TensorFlow to only allocate necessary GPU memory"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set for {len(gpus)} GPUs")
    except Exception as e:
        print(f"Error setting memory growth: {e}")

def create_color_map(color_list, name='custom_cmap'):
    """Create a custom colormap from a list of colors"""
    return LinearSegmentedColormap.from_list(name, color_list)

def plot_to_image(fig):
    """Convert matplotlib figure to image data"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generate a link to download the plot as an image"""
    buf = plot_to_image(fig)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_csv_download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_model_download_link(model_path, filename="model.zip", text="Download Model"):
    """Generate a link to download saved model files as a ZIP"""
    if not os.path.exists(model_path):
        return "Model files not found"
    
    # Create a ZIP file of the model directory
    zip_path = "model_tmp.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # If model_path is a directory, add all files
        if os.path.isdir(model_path):
            for root, _, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(model_path))
                    zipf.write(file_path, arcname)
        # If it's a single file, just add it
        else:
            zipf.write(model_path, os.path.basename(model_path))
    
    # Read the ZIP file and create a download link
    with open(zip_path, "rb") as f:
        bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">{text}</a>'
    
    # Clean up the temporary zip file
    os.remove(zip_path)
    
    return href

def plot_appliance_contribution(appliance_data, appliance_colors=None, figsize=(10, 6)):
    """Plot the contribution of each appliance to total power"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sum across time to get total consumption
    appliance_totals = appliance_data.sum()
    
    # Sort by value
    appliance_totals = appliance_totals.sort_values(ascending=False)
    
    # Create bar plot
    bars = ax.bar(appliance_totals.index, appliance_totals.values)
    
    # Apply colors if provided
    if appliance_colors:
        for i, bar in enumerate(bars):
            appliance = appliance_totals.index[i]
            if appliance in appliance_colors:
                bar.set_color(appliance_colors[appliance])
    
    # Add labels
    ax.set_title('Total Energy Consumption by Appliance')
    ax.set_ylabel('Total Power (W)')
    ax.set_xlabel('Appliance')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def create_altair_timeseries(df, x_column, y_columns, title="Time Series Data", color_dict=None):
    """Create an Altair time series visualization"""
    # Melt the dataframe to long format for Altair
    melted_df = pd.melt(
        df, 
        id_vars=[x_column], 
        value_vars=y_columns,
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create the base chart
    chart = alt.Chart(melted_df).mark_line().encode(
        x=alt.X(f'{x_column}:T', title='Time'),
        y=alt.Y('Value:Q', title='Value'),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=list(y_columns), range=list(color_dict.values()) if color_dict else None)),
        tooltip=['Metric:N', 'Value:Q', f'{x_column}:T']
    ).properties(
        title=title,
        width=700,
        height=400
    ).interactive()
    
    return chart

def get_model_summary(model):
    """Return a formatted string summary of a Keras model"""
    if not isinstance(model, tf.keras.Model):
        return "Not a Keras model"
    
    # Redirect model.summary() output to a string
    summary_string = []
    model.summary(print_fn=lambda x: summary_string.append(x))
    
    return "\n".join(summary_string)

def format_metrics_for_display(metrics):
    """Format metrics dictionary for nice display in Streamlit"""
    formatted = {}
    
    for model_name, model_metrics in metrics.items():
        # Handle efficiency metrics specially
        if 'efficiency' in model_metrics:
            eff = model_metrics['efficiency']
            formatted[f"{model_name}_training_time"] = f"{eff.get('training_time_seconds', 0):.2f} seconds"
            formatted[f"{model_name}_inference_time"] = f"{eff.get('inference_time_seconds', 0):.2f} seconds"
            formatted[f"{model_name}_model_size"] = f"{eff.get('model_size_parameters', 0):,} parameters"
            if 'model_memory_mb' in eff:
                formatted[f"{model_name}_memory"] = f"{eff.get('model_memory_mb', 0):.2f} MB"
        else:
            # Handle regular metrics
            for metric_name, metric_value in model_metrics.items():
                if isinstance(metric_value, (int, float)):
                    formatted[f"{model_name}_{metric_name}"] = f"{metric_value:.4f}"
    
    return formatted

def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 8)):
    """Plot confusion matrix for classification results"""
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes if classes else "auto",
        yticklabels=classes if classes else "auto",
        ax=ax
    )
    
    # Set labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig

def plot_regression_results(y_true, y_pred, figsize=(10, 6)):
    """Plot actual vs predicted values for regression models"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of actual vs predicted
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    ax_min = min(min(y_true), min(y_pred))
    ax_max = max(max(y_true), max(y_pred))
    margin = (ax_max - ax_min) * 0.1
    ax_min -= margin
    ax_max += margin
    
    ax.plot([ax_min, ax_max], [ax_min, ax_max], 'r--', lw=2)
    
    # Set labels and title
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs Predicted Values')
    
    # Add R² value
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.95), xycoords='axes fraction')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    
    plt.tight_layout()
    return fig

def display_progress_bar(progress, text=""):
    """Display a customized progress bar in Streamlit"""
    # Ensure progress is between 0 and 1
    progress = max(0, min(1, progress))
    
    if text:
        text = f"{text}: {progress*100:.1f}%"
    else:
        text = f"Progress: {progress*100:.1f}%"
    
    # Display progress bar
    st.progress(progress)
    st.text(text)

def load_model_if_exists(model_class, model_path):
    """Load a model if the file exists, otherwise return None"""
    try:
        if os.path.exists(model_path):
            model = model_class()
            model.load(model_path)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None
