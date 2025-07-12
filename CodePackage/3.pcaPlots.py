import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
import os
import json
import joblib

from supporting_functions import get_fp

folder_fp, data_fp, figures_fp = get_fp('man_dtop')

# Load your PCA data
pca_model = joblib.load(os.path.expanduser(data_fp + "PCA/pca_phys_model.pkl"))
X_pca = np.load(os.path.expanduser(data_fp + "PCA/pca_phys.npy"))

# Load rheological data
rawRheo = pd.read_excel(
    os.path.expanduser(data_fp + "Armstrong_tESSTV_simplified.xlsx"),
    sheet_name="Rheology_forML"
)

# Load variable dictionaries
with open(os.path.expanduser(data_fp + "rheology_variables.json"), 'r') as f:
    rheoDict = json.load(f)

# Extract rheological parameters
rheology_params = [col for col in rawRheo.columns if col != 'donors']
Y_rheology = rawRheo[rheology_params].values

# Create donor labels (matching your existing code)
donor_labels = []
for i in range(len(X_pca)):
    letter_index = i if i < 4 else i + 1  # Skip E
    donor_labels.append(chr(65 + letter_index))

DATA_LOADED = True
print("Successfully loaded your actual data!")
    

def calculate_correlation_stats(x_data, y_data):
    if len(x_data) != len(y_data) or len(x_data) < 3:
        return None
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]
    
    if len(x_clean) < 3:
        return None
    
    # Calculate correlation
    correlation_r, p_value = stats.pearsonr(x_clean, y_clean)
    
    # Calculate regression line
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    
    # Calculate confidence interval (matching your approach)
    n = len(x_clean)
    x_mean = np.mean(x_clean)
    sxx = np.sum((x_clean - x_mean)**2)
    sxy = np.sum((x_clean - x_mean) * (y_clean - np.mean(y_clean)))
    syy = np.sum((y_clean - np.mean(y_clean))**2)
    s = np.sqrt((syy - sxy**2/sxx) / (n-2)) if n > 2 else 0
    t_critical = stats.t.ppf(0.975, n-2) if n > 2 else 0
    
    return {
        'correlation': correlation_r,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'n_points': n,
        'std_error': s,
        't_critical': t_critical,
        'x_mean': x_mean,
        'sxx': sxx
    }

def create_correlation_plot(pc_idx, rheo_idx):
    # Extract data
    x_data = X_pca[:, pc_idx]
    y_data = Y_rheology[:, rheo_idx]
    
    # Calculate statistics
    stats_result = calculate_correlation_stats(x_data, y_data)
    
    if stats_result is None:
        # Return empty plot if calculation fails
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for correlation analysis",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font_size=16
        )
        return fig
    
    # Create the main scatter plot
    fig = go.Figure()
    
    # Add scatter points with donor labels
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers+text',
        text=donor_labels,
        textposition='top center',
        marker=dict(
            size=12,
            color='#1f77b4',
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        name='Data Points',
        hovertemplate='<b>Donor %{text}</b><br>' +
                     f'PC{pc_idx+1}: %{{x:.3f}}<br>' +
                     f'{rheology_params[rheo_idx]}: %{{y:.3f}}<extra></extra>'
    ))
    
    # Create regression line
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    y_regression = stats_result['slope'] * x_range + stats_result['intercept']
    
    # Calculate confidence interval
    ci = stats_result['t_critical'] * stats_result['std_error'] * np.sqrt(
        1/stats_result['n_points'] + (x_range - stats_result['x_mean'])**2/stats_result['sxx']
    )
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([y_regression + ci, (y_regression - ci)[::-1]]),
        fill='toself',
        fillcolor='rgba(68, 119, 170, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_regression,
        mode='lines',
        line=dict(color='#4477AA', width=2, dash='dash'),
        name='Regression Line',
        hovertemplate=f'Regression: y = {stats_result["slope"]:.3f}x + {stats_result["intercept"]:.3f}<extra></extra>'
    ))
    
    # Update layout with your styling preferences
    pc_variance = pca_model.explained_variance_ratio_[pc_idx] * 100
    fig.update_layout(
        title=dict(
            text=f'PC{pc_idx+1} vs {rheoDict[rheology_params[rheo_idx]]["latex_name"]} Correlation (r = {stats_result["correlation"]:.3f})',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title=f'PC{pc_idx+1} ({pc_variance:.1f}% variance)',
        yaxis_title=rheoDict[rheology_params[rheo_idx]]["latex_name"],
        width=800,
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12),
        # Add grid matching your matplotlib style
        xaxis=dict(
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=1
        )
    )
    
    # Add correlation statistics as annotation
    significance = "***" if stats_result['p_value'] < 0.001 else "**" if stats_result['p_value'] < 0.01 else "*" if stats_result['p_value'] < 0.05 else "ns"
    
    fig.add_annotation(
        text=f"r = {stats_result['correlation']:.3f} ({significance})<br>" +
             f"p = {stats_result['p_value']:.4f}<br>" +
             f"n = {stats_result['n_points']}",
        x=.9, y=0.85,
        xref="paper", yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=10)
    )
    
    return fig

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Blood Rheology Correlation Explorer"

# Define the app layout
app.layout = html.Div([
    # Header section
    html.Div([
        html.H1("Blood Rheology Correlation Explorer", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Interactive exploration of correlations between Principal Components and rheological parameters",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px', 'marginBottom': '30px'}),
        
        # Data status indicator
        html.Div([
            html.Span("📊 Data Status: ", style={'fontWeight': 'bold'}),
            html.Span("Real Data Loaded" if DATA_LOADED else "Demo Data (Simulated)", 
                     style={'color': '#27ae60' if DATA_LOADED else '#e67e22',
                            'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Select Principal Component (X-axis):", 
                      style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='pc-dropdown',
                options=[{'label': f'PC{i+1} ({pca_model.explained_variance_ratio_[i]*100:.1f}% variance)', 'value': i} 
                        for i in range(min(10, X_pca.shape[1]))],  # Limit to first 10 PCs for clarity
                value=7,  # Default to PC8 (0-indexed)
                style={'marginBottom': '20px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Select Rheological Parameter (Y-axis):", 
                      style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='rheo-dropdown',
                options=[{'label': param, 'value': i} for i, param in enumerate(rheology_params)],
                value=3,  # Default to t_r1 (s)
                style={'marginBottom': '20px'}
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'marginBottom': '30px', 'padding': '0 20px'}),
    
    # Main plot area
    html.Div([
        dcc.Graph(id='correlation-plot', style={'height': '600px'})
    ], style={'padding': '0 20px'}),
    
    # Information panel
    html.Div([
        html.H3("Understanding the Analysis", style={'color': '#2c3e50'}),
        html.P([
            "This interactive tool allows you to explore correlations between Principal Components (PCs) derived from physiological blood panel data and rheological parameters from the t-ESSTV constitutive model. ",
            "Each PC captures a different pattern of variation in the physiological measurements, while rheological parameters describe blood flow behavior under different conditions."
        ], style={'textAlign': 'justify', 'lineHeight': '1.6'}),
        
        html.H4("Key Features:", style={'color': '#34495e', 'marginTop': '20px'}),
        html.Ul([
            html.Li("Scatter plot shows individual donor data points with letter labels"),
            html.Li("Dashed regression line indicates the overall trend"),
            html.Li("Shaded area represents 95% confidence interval"),
            html.Li("Correlation coefficient (r) and significance level displayed"),
            html.Li("PC variance explanation shown in axis labels")
        ], style={'lineHeight': '1.8'}),
        
        html.H4("Statistical Interpretation:", style={'color': '#34495e', 'marginTop': '20px'}),
        html.Ul([
            html.Li("r > 0.7: Strong positive correlation"),
            html.Li("0.3 < r < 0.7: Moderate correlation"), 
            html.Li("r < 0.3: Weak correlation"),
            html.Li("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        ], style={'lineHeight': '1.8'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'margin': '20px', 'borderRadius': '5px'})
])

# Callback to update the plot based on dropdown selections
@app.callback(
    Output('correlation-plot', 'figure'),
    [Input('pc-dropdown', 'value'),
     Input('rheo-dropdown', 'value')]
)
def update_plot(pc_idx, rheo_idx):
    """
    This callback function runs whenever the user changes either dropdown.
    It regenerates the correlation plot with the newly selected parameters.
    """
    return create_correlation_plot(pc_idx, rheo_idx)

# Run the app
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 BLOOD RHEOLOGY CORRELATION EXPLORER")
    print("="*60)
    print(f"📊 Dataset: {X_pca.shape[0]} donors, {X_pca.shape[1]} PCs, {len(rheology_params)} rheological parameters")
    print(f"🔬 Analysis: Principal Component Analysis + Rheological Modeling")
    print(f"📈 Available correlations: {X_pca.shape[1] * len(rheology_params)} combinations")
    print("="*60)
    print("🌐 Starting local web server...")
    print("📱 Open your browser to: http://127.0.0.1:8050")
    print("⚡ The app will automatically update when you change selections")
    print("🛑 Press Ctrl+C to stop the server")
    print("="*60)
    
    # Run the app with debug mode for development
    app.run(debug=True, host='127.0.0.1', port=8050)