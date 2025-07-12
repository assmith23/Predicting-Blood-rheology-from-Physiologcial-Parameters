import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def ensure_dataframe(data, columns=None):
    """Convert numpy array to DataFrame if needed"""
    if hasattr(data, 'columns'):
        return data  # Already a DataFrame
    else:
        if columns is not None:
            return pd.DataFrame(data, columns=columns)
        else:
            return pd.DataFrame(data)

def safe_indexing(data, indices):
    """Safely index data whether it's DataFrame or numpy array"""
    if hasattr(data, 'iloc'):
        return data.iloc[indices]
    else:
        return data[indices]

class GaussianCopulaSynthesizer:
    """
    Synthesizes data using Gaussian copula with KNN-based similarity weighting
    """
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors
        self.fitted = False
        
    def fit(self, X, y):
        """
        Fit the Gaussian copula model
        X: physiological features (n_samples, n_features)
        y: rheological targets (n_samples, n_targets)
        """
        self.X_orig = X.copy()
        self.y_orig = y.copy()
        
        # Combine features and targets
        self.data = np.column_stack([X, y])
        n_samples, n_features = self.data.shape
        
        # Fit KNN for similarity detection
        self.knn = NearestNeighbors(n_neighbors=min(self.k_neighbors, n_samples-1))
        self.knn.fit(X)
        
        # Transform to uniform marginals (empirical CDF)
        self.uniform_data = np.zeros_like(self.data)
        for j in range(n_features):
            # Rank transform to [0,1]
            ranks = np.argsort(np.argsort(self.data[:, j]))
            self.uniform_data[:, j] = (ranks + 1) / (n_samples + 1)
        
        # Transform to normal marginals
        self.normal_data = norm.ppf(self.uniform_data)
        
        # Fit multivariate normal to get correlation structure
        self.mean = np.mean(self.normal_data, axis=0)
        self.cov = np.cov(self.normal_data.T)
        
        # Store marginal distributions for inverse transform
        self.marginals = []
        for j in range(n_features):
            sorted_vals = np.sort(self.data[:, j])
            self.marginals.append(sorted_vals)
            
        self.fitted = True
        return self
    
    def _inverse_transform(self, normal_samples):
        """Transform from normal back to original scale"""
        n_samples, n_features = normal_samples.shape
        uniform_samples = norm.cdf(normal_samples)
        
        # Transform back to original scale using empirical inverse CDF
        original_samples = np.zeros_like(uniform_samples)
        for j in range(n_features):
            # Interpolate using sorted original values
            quantiles = uniform_samples[:, j]
            marginal = self.marginals[j]
            n_orig = len(marginal)
            
            for i in range(n_samples):
                q = quantiles[i]
                idx = q * (n_orig - 1)
                lower_idx = int(np.floor(idx))
                upper_idx = min(lower_idx + 1, n_orig - 1)
                
                if lower_idx == upper_idx:
                    original_samples[i, j] = marginal[lower_idx]
                else:
                    # Linear interpolation
                    weight = idx - lower_idx
                    original_samples[i, j] = (1 - weight) * marginal[lower_idx] + weight * marginal[upper_idx]
        
        return original_samples
    
    def generate(self, n_synthetic, similarity_weight=0.3):
        """
        Generate synthetic samples
        similarity_weight: how much to weight towards similar existing samples
        """
        if not self.fitted:
            raise ValueError("Must fit the model first")
        
        n_orig = len(self.X_orig)
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            if np.random.random() < similarity_weight and n_orig > 1:
                # Generate sample similar to existing data
                # Pick a random original sample
                base_idx = np.random.randint(0, n_orig)
                base_sample = self.X_orig[base_idx:base_idx+1]
                
                # Find its neighbors
                distances, indices = self.knn.kneighbors(base_sample)
                neighbor_data = self.normal_data[indices[0]]
                
                # Sample from neighborhood
                if len(neighbor_data) > 1:
                    local_mean = np.mean(neighbor_data, axis=0)
                    local_cov = np.cov(neighbor_data.T) + 0.01 * np.eye(len(local_mean))
                    sample = multivariate_normal.rvs(local_mean, local_cov)
                else:
                    sample = neighbor_data[0] + 0.1 * np.random.randn(len(neighbor_data[0]))
            else:
                # Generate from global distribution
                sample = multivariate_normal.rvs(self.mean, self.cov)
            
            synthetic_samples.append(sample)
        
        synthetic_samples = np.array(synthetic_samples)
        
        # Transform back to original scale
        synthetic_original = self._inverse_transform(synthetic_samples)
        
        # Split back into X and y
        n_features_X = self.X_orig.shape[1]
        X_synthetic = synthetic_original[:, :n_features_X]
        y_synthetic = synthetic_original[:, n_features_X:]
        
        return X_synthetic, y_synthetic

class PCAGPRPipeline:
    """
    Pipeline combining PCA dimensionality reduction with GPR
    """
    def __init__(self, n_components=None, kernel=None):
        self.n_components = n_components
        self.kernel = kernel if kernel is not None else C(1.0) * RBF(1.0) + WhiteKernel(0.1)
        self.scalers_X = {}
        self.scalers_y = {}
        self.pca_models = {}
        self.gpr_models = {}
        
    def fit(self, X, y, feature_names=None, target_names=None):
        """
        Fit PCA + GPR for each target variable
        """
        # Handle categorical variables
        X_processed = self._preprocess_features(X)
        
        # Set target names
        if target_names is not None:
            self.target_names = target_names
        elif hasattr(y, 'columns'):
            self.target_names = y.columns.tolist()
        else:
            self.target_names = [f'target_{i}' for i in range(y.shape[1])]
        
        # Store feature names for later use
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        for i, target_name in enumerate(self.target_names):
            # Scale features
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_processed)
            self.scalers_X[target_name] = scaler_X
            
            # Scale target
            scaler_y = StandardScaler()
            y_target = y[:, i] if y.ndim > 1 else y
            # Convert to numpy array if it's a pandas Series
            if hasattr(y_target, 'values'):
                y_target = y_target.values
            if y_target.ndim == 0:
                y_target = np.array([y_target])
            y_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1)).ravel()
            self.scalers_y[target_name] = scaler_y
            
            # Fit PCA
            n_comp = min(self.n_components or X_scaled.shape[1], X_scaled.shape[0] - 1, X_scaled.shape[1])
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X_scaled)
            self.pca_models[target_name] = pca
            
            # Fit GPR
            gpr = GaussianProcessRegressor(kernel=self.kernel, random_state=42)
            gpr.fit(X_pca, y_scaled)
            self.gpr_models[target_name] = gpr
            
        return self
    
    def _preprocess_features(self, X):
        """Handle categorical variables"""
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = X.copy() if hasattr(X, 'copy') else np.array(X)
        
        if hasattr(X, 'columns'):
            # Handle 'Sex' column if present
            if 'Sex' in X.columns:
                le = LabelEncoder()
                X_processed['Sex'] = le.fit_transform(X['Sex'])
                if not hasattr(self, 'label_encoders'):
                    self.label_encoders = {}
                self.label_encoders['Sex'] = le
        
        return X_processed.values if hasattr(X_processed, 'values') else X_processed
    
    def predict(self, X, return_std=False):
        """Predict using fitted models"""
        X_processed = self._preprocess_features(X)
        
        predictions = {}
        uncertainties = {}
        for target_name in self.target_names:
            # Transform features
            X_scaled = self.scalers_X[target_name].transform(X_processed)
            X_pca = self.pca_models[target_name].transform(X_scaled)
            
            # Predict
            if return_std:
                y_pred_scaled, y_std_scaled = self.gpr_models[target_name].predict(X_pca, return_std=True)
                # Transform back to original scale
                y_pred = self.scalers_y[target_name].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                y_std = y_std_scaled * self.scalers_y[target_name].scale_[0]
                uncertainties[target_name] = y_std
            else:
                y_pred_scaled = self.gpr_models[target_name].predict(X_pca)
                y_pred = self.scalers_y[target_name].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            predictions[target_name] = y_pred
        
        if return_std:
            return predictions, uncertainties
        return predictions

def evaluate_model_performance(y_true, y_pred, target_names=None):
    """Calculate comprehensive performance metrics"""
    metrics = {}
    
    # Handle target names
    if target_names is not None:
        target_names_list = target_names
    elif hasattr(y_true, 'columns'):
        target_names_list = y_true.columns.tolist()
    else:
        target_names_list = [f'target_{i}' for i in range(y_true.shape[1] if y_true.ndim > 1 else 1)]
    
    for i, target_name in enumerate(target_names_list):
        # Extract target values
        if y_true.ndim > 1:
            y_t = y_true[:, i] if isinstance(y_true, np.ndarray) else y_true.iloc[:, i]
        else:
            y_t = y_true
            
        # Convert to numpy array if it's a pandas Series
        if hasattr(y_t, 'values'):
            y_t = y_t.values
        
        y_p = y_pred[target_name] if isinstance(y_pred, dict) else y_pred[:, i] if y_pred.ndim > 1 else y_pred
        
        metrics[target_name] = {
            'MSE': mean_squared_error(y_t, y_p),
            'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
            'MAE': mean_absolute_error(y_t, y_p),
            'R2': r2_score(y_t, y_p)
        }
    
    return metrics

def run_synthetic_data_experiment(X_phys, y_rheo, target_sizes=[35, 50, 65, 80], n_components=10, 
                                feature_names=None, target_names=None):
    """
    Main experiment function
    """
    print("=== KNN-Gaussian Copula Synthetic Data with PCA-GPR Analysis ===\n")
    
    original_size = len(X_phys)
    print(f"Original dataset size: {original_size}")
    print(f"Physiological features: {X_phys.shape[1]}")
    print(f"Rheological targets: {y_rheo.shape[1] if y_rheo.ndim > 1 else 1}")
    print(f"Target sizes to test: {target_sizes}\n")
    
    results = {}
    
    # Baseline: original data only
    print("Training baseline model on original data...")
    pipeline_baseline = PCAGPRPipeline(n_components=min(n_components, original_size-1))
    pipeline_baseline.fit(X_phys, y_rheo, feature_names=feature_names, target_names=target_names)
    
    # Cross-validation on original data
    cv_scores = []
    kfold = KFold(n_splits=min(5, original_size-1), shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(X_phys):
        X_train_cv, X_val_cv = safe_indexing(X_phys, train_idx), safe_indexing(X_phys, val_idx)
        y_train_cv, y_val_cv = safe_indexing(y_rheo, train_idx), safe_indexing(y_rheo, val_idx)
        
        pipeline_cv = PCAGPRPipeline(n_components=min(n_components, len(train_idx)-1))
        pipeline_cv.fit(X_train_cv, y_train_cv, feature_names=feature_names, target_names=target_names)
        
        y_pred_cv = pipeline_cv.predict(X_val_cv)
        metrics_cv = evaluate_model_performance(y_val_cv, y_pred_cv, target_names=target_names)
        
        # Average R2 across targets
        avg_r2 = np.mean([metrics_cv[target]['R2'] for target in metrics_cv.keys()])
        cv_scores.append(avg_r2)
    
    baseline_cv_score = np.mean(cv_scores)
    results['baseline'] = {
        'size': original_size,
        'cv_r2': baseline_cv_score,
        'cv_r2_std': np.std(cv_scores)
    }
    
    print(f"Baseline CV R² (±std): {baseline_cv_score:.3f} (±{np.std(cv_scores):.3f})")
    
    # Fit copula synthesizer
    print("\nFitting Gaussian copula synthesizer...")
    synthesizer = GaussianCopulaSynthesizer(k_neighbors=min(5, original_size-1))
    
    # Convert to numpy arrays for synthesizer
    X_phys_array = X_phys if isinstance(X_phys, np.ndarray) else X_phys.values
    y_rheo_array = y_rheo if isinstance(y_rheo, np.ndarray) else y_rheo.values
    
    synthesizer.fit(X_phys_array, y_rheo_array)
    
    # Test different target sizes
    for target_size in target_sizes:
        if target_size <= original_size:
            print(f"\nSkipping target size {target_size} (≤ original size)")
            continue
            
        n_synthetic = target_size - original_size
        print(f"\nTesting target size: {target_size} ({n_synthetic} synthetic samples)")
        
        # Generate synthetic data
        X_synthetic, y_synthetic = synthesizer.generate(n_synthetic)
        
        # Combine original and synthetic data
        X_combined = np.vstack([X_phys_array, X_synthetic])
        y_combined = np.vstack([y_rheo_array, y_synthetic])
        
        # Train model on combined data
        pipeline_synthetic = PCAGPRPipeline(n_components=min(n_components, target_size-1))
        pipeline_synthetic.fit(X_combined, y_combined, feature_names=feature_names, target_names=target_names)
        
        # Evaluate on original test data (using original data as test set)
        y_pred_orig = pipeline_synthetic.predict(X_phys_array)
        metrics_orig = evaluate_model_performance(y_rheo_array, y_pred_orig, target_names=target_names)
        
        # Average R2 across targets
        avg_r2_orig = np.mean([metrics_orig[target]['R2'] for target in metrics_orig.keys()])
        
        # Cross-validation on combined data
        cv_scores_synthetic = []
        kfold_synthetic = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kfold_synthetic.split(X_combined):
            X_train_syn, X_val_syn = safe_indexing(X_combined, train_idx), safe_indexing(X_combined, val_idx)
            y_train_syn, y_val_syn = safe_indexing(y_combined, train_idx), safe_indexing(y_combined, val_idx)
            
            pipeline_cv_syn = PCAGPRPipeline(n_components=min(n_components, len(train_idx)-1))
            pipeline_cv_syn.fit(X_train_syn, y_train_syn, feature_names=feature_names, target_names=target_names)
            
            y_pred_cv_syn = pipeline_cv_syn.predict(X_val_syn)
            metrics_cv_syn = evaluate_model_performance(y_val_syn, y_pred_cv_syn, target_names=target_names)
            
            avg_r2_cv_syn = np.mean([metrics_cv_syn[target]['R2'] for target in metrics_cv_syn.keys()])
            cv_scores_synthetic.append(avg_r2_cv_syn)
        
        avg_cv_synthetic = np.mean(cv_scores_synthetic)
        
        results[target_size] = {
            'size': target_size,
            'n_synthetic': n_synthetic,
            'test_on_original_r2': avg_r2_orig,
            'cv_r2': avg_cv_synthetic,
            'cv_r2_std': np.std(cv_scores_synthetic),
            'detailed_metrics': metrics_orig
        }
        
        print(f"  Test on original R²: {avg_r2_orig:.3f}")
        print(f"  CV R² (±std): {avg_cv_synthetic:.3f} (±{np.std(cv_scores_synthetic):.3f})")
    
    return results, synthesizer

def analyze_feature_importance(pipeline, feature_names=None):
    """Analyze PCA component importance for each target"""
    print("\n=== PCA Component Importance Analysis ===")
    
    importance_results = {}
    
    # Get feature names
    if feature_names is not None:
        feature_names_list = feature_names
    elif hasattr(pipeline, 'feature_names'):
        feature_names_list = pipeline.feature_names
    else:
        feature_names_list = [f'feature_{i}' for i in range(len(pipeline.pca_models[pipeline.target_names[0]].components_[0]))]
    
    for target_name in pipeline.target_names:
        pca_model = pipeline.pca_models[target_name]
        
        # Get explained variance ratio
        explained_var = pca_model.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        # Get component loadings (how much each original feature contributes to each component)
        components = pca_model.components_
        
        importance_results[target_name] = {
            'explained_variance_ratio': explained_var,
            'cumulative_variance': cumulative_var,
            'components': components,
            'feature_names': feature_names_list
        }
        
        print(f"\n{target_name}:")
        print(f"  First {min(5, len(explained_var))} components explain: {cumulative_var[min(4, len(cumulative_var)-1)]:.3f} of variance")
        
        # Show top contributing features to first component
        first_component = np.abs(components[0])
        top_features_idx = np.argsort(first_component)[-5:][::-1]
        print(f"  Top features in PC1: {[feature_names_list[i] for i in top_features_idx]}")
    
    return importance_results

def plot_rheological_performance(results, rheo_cols, save_path=None):
    """
    Create a performance plot by rheological target similar to the provided image
    
    Parameters:
    results: Dictionary from run_synthetic_data_experiment
    rheo_cols: List of rheological column names
    save_path: Optional path to save the figure
    """
    
    # Create LaTeX-style labels for rheological targets
    rheo_targets_latex = {
        'mu_0': r'$\mu_0$',
        'mu_inf': r'$\mu_\infty$', 
        'tau_c': r'$\tau_c$',
        'TR1': r'$T_{R^1}$',
        'TR2': r'$T_{R^2}$',
        'mu_R': r'$\mu_R$',
        'sigma_y0': r'$\sigma_{\gamma^0}$',
        'tau_lambda': r'$\tau_\lambda$',
        'G_R': r'$G_R$',
        'G_C': r'$G_C$'
    }
    
    # If exact column names don't match, create generic labels
    if not any(col in rheo_targets_latex for col in rheo_cols):
        rheo_targets_latex = {col: col for col in rheo_cols}
    
    # Extract data for plotting
    plot_data = []
    
    # Extract performance for each synthetic data size
    for key, result in results.items():
        if key == 'baseline':
            continue
            
        sample_size = result['size']
        detailed_metrics = result['detailed_metrics']
        
        for target_name, metrics in detailed_metrics.items():
            plot_data.append({
                'target': target_name,
                'n_samples': sample_size,
                'r2_score': metrics['R2']
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each rheological target
    unique_targets = results_df['target'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_targets)))
    
    for i, target in enumerate(unique_targets):
        data = results_df[results_df['target'] == target]
        data_sorted = data.sort_values('n_samples')
        
        # Get label for legend
        label = rheo_targets_latex.get(target, target)
        
        plt.plot(data_sorted['n_samples'], data_sorted['r2_score'], 
                marker='o', linewidth=2, markersize=6, 
                color=colors[i], label=label)
    
    # Formatting
    plt.xlabel('Sample Size (N)', fontsize=12)
    plt.ylabel(r'$R^2$ Score', fontsize=12)
    plt.title('Performance by Rheological Target', fontsize=14, fontweight='bold')
    
    # Set y-axis limits similar to your image
    plt.ylim(-0.5, 1.5)
    
    # Add horizontal line at R² = 0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Grid
    plt.grid(True, alpha=0.3)
    
    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return results_df

def get_fp(cpu="man_dtop"):
    if (cpu == "man_dtop"):
        data_fp = r"/home/msmitty/Documents/TransientBloodRheo_ML/DATA/"
        folder_fp = r"/home/msmitty/Documents/TransientBloodRheo_ML/"
        figures_fp = r"/home/msmitty/Documents/TransientBloodRheo_ML/Figures/correlationPCA/"
    elif (cpu == "surface"):
        data_fp = r"C:\Users\manni\OneDrive - University of Delaware - o365\Python Projects\TransientBloodRheo_ML\DATA"
        folder_fp = r"C:\Users\manni\OneDrive - University of Delaware - o365\Python Projects\TransientBloodRheo_ML"
        figures_fp = r"C:\Users\manni\OneDrive - University of Delaware - o365\Python Projects\TransientBloodRheo_ML\Figures\synthetic"
    elif (cpu == "sean"):
        data_fp = r" "
    else:
        print("Error getting file path")
    
    return folder_fp, data_fp, figures_fp

def main():
    """
    Main execution function
    """
    print("Loading data...")
    
    # Get Correct Filepath
    #cpu = "surface"
    cpu = "man_dtop"
    #cpu = "sean"
    folder_fp, data_fp, figures_fp = get_fp(cpu)

    # Filepaths
    excel_path = os.path.join(data_fp, r"Armstrong_tESSTV_simplified.xlsx")
    physio_json_path = os.path.join(data_fp, "physiological_variables.json")
    rheo_json_path = os.path.join(data_fp, "rheology_variables.json")

    try:
        # Load Data
        phys_data = pd.read_excel(excel_path, sheet_name="Physiological_forML")
        rheo_data = pd.read_excel(excel_path, sheet_name="Rheology_forML")

        print(f"Physiological data shape: {phys_data.shape}")
        print(f"Rheological data shape: {rheo_data.shape}")

        # Load dictionaries
        with open(physio_json_path, 'r') as f:
            physioDict = json.load(f)
        with open(rheo_json_path, 'r') as f:
            rheoDict = json.load(f)

        # Get column names (excluding 'donors')
        physio_cols = [col for col in phys_data.columns if col != 'donors']
        rheo_cols = [col for col in rheo_data.columns if col != 'donors']

        n_neighbors = 3

        # Apply KNN imputation to rheological data
        rheological_values = rheo_data[rheo_cols].values
        imputer = KNNImputer(n_neighbors=min(n_neighbors, len(rheological_values)//2))
        imputed_rheo_data = imputer.fit_transform(rheological_values)

        # Create imputed DataFrame
        imputed_rheo_df = pd.DataFrame(
            imputed_rheo_data,
            columns=rheo_cols
        )
        imputed_rheo_df['donors'] = rheo_data['donors'].values

        # Merge datasets
        combined_data = pd.merge(phys_data, imputed_rheo_df, on='donors')

        # Extract features and targets
        X = combined_data[physio_cols].select_dtypes(include=[np.number]).values
        y = combined_data[rheo_cols].values

        print(f"Final feature matrix X: {X.shape}")
        print(f"Final target matrix y: {y.shape}")
        print(f"Physiological features: {len(physio_cols)}")
        print(f"Rheological targets: {len(rheo_cols)}")
        print(f"Target names: {rheo_cols}")
        
        # Get numeric physiological column names for feature importance
        numeric_physio_cols = combined_data[physio_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Run the experiment
        results, synthesizer = run_synthetic_data_experiment(
            X, y, 
            target_sizes=[30, 50, 75, 100], 
            n_components=10,
            feature_names=numeric_physio_cols,
            target_names=rheo_cols
        )
        
        # Analyze results
        print("\n=== Summary of Results ===")
        for key, result in results.items():
            if key == 'baseline':
                print(f"Baseline (n={result['size']}): CV R² = {result['cv_r2']:.3f}")
            else:
                print(f"Synthetic n={result['size']}: CV R² = {result['cv_r2']:.3f}, "
                      f"Test on original R² = {result['test_on_original_r2']:.3f}")
        
        # Feature importance analysis
        final_pipeline = PCAGPRPipeline(n_components=10)
        final_pipeline.fit(X, y, feature_names=numeric_physio_cols, target_names=rheo_cols)
        importance_results = analyze_feature_importance(final_pipeline, feature_names=numeric_physio_cols)
        
        # Generate the rheological performance plot
        print("\n=== Generating Rheological Performance Plot ===")
        plot_save_path = os.path.join(figures_fp, "rheological_performance.png")
        plot_data = plot_rheological_performance(results, rheo_cols, save_path=plot_save_path)
        print(f"Plot saved to: {plot_save_path}")
        
        return results, synthesizer, importance_results
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please make sure the Excel and JSON files are in the correct directory.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

if __name__ == "__main__":
    results, synthesizer, importance_results = main()