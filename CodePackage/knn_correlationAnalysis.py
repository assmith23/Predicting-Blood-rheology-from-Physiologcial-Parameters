
import pandas as pd
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings

# Suppress GPR warnings
warnings.filterwarnings('ignore', message='.*k2_noise_level.*')


class KNNSyntheticGenerator:
    """Generate synthetic samples using KNN-based interpolation"""
    
    def __init__(self, k=5, noise_level=0.05, random_state=42):
        self.k = k
        self.noise_level = noise_level
        self.random_state = random_state
        self.nn_model = NearestNeighbors(n_neighbors=k+1)
        np.random.seed(random_state)
        
    def fit(self, X):
        """Fit the KNN model to the original data"""
        self.X_original = X.copy()
        self.nn_model.fit(X)
        self.feature_std = np.std(X, axis=0)
        return self
    
    def generate_synthetic(self, n_synthetic):
        """Generate n_synthetic samples using KNN interpolation"""
        synthetic_samples = []
        
        for i in range(n_synthetic):
            # Randomly select a seed point
            seed_idx = np.random.randint(0, len(self.X_original))
            seed_point = self.X_original[seed_idx]
            
            # Find k nearest neighbors
            distances, indices = self.nn_model.kneighbors([seed_point])
            neighbor_indices = indices[0][1:]  # Exclude seed point
            
            # Interpolate between seed and random neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor_point = self.X_original[neighbor_idx]
            alpha = np.random.uniform(0.2, 0.8)
            synthetic_point = alpha * seed_point + (1 - alpha) * neighbor_point
            
            # Add adaptive noise
            adaptive_noise = np.random.normal(0, self.noise_level * self.feature_std)
            synthetic_point = synthetic_point + adaptive_noise
            
            synthetic_samples.append(synthetic_point)
            
        return np.array(synthetic_samples)


class BloodRheologyAnalysis:
    """Simplified blood rheology ML analysis class"""
    
    def __init__(self, X, y, target_names):
        """
        Initialize analysis
        
        Parameters:
        - X: Physiological features (n_samples x n_features)
        - y: Rheological targets (n_samples x n_targets)
        - target_names: Names of rheological parameters
        """
        self.original_X = X
        self.original_y = y
        self.target_names = target_names
        
        # Fit scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_scaled = self.scaler_X.fit_transform(X)
        self.y_scaled = self.scaler_y.fit_transform(y)
        
    def generate_synthetic_dataset(self, n_samples, k=3):
        """Generate synthetic dataset of specified size"""
        generator = KNNSyntheticGenerator(k=k, noise_level=0.05)
        generator.fit(self.X_scaled)
        
        n_synthetic_X = n_samples - len(self.X_scaled)
        if n_synthetic_X > 0:
            synthetic_X = generator.generate_synthetic(n_synthetic_X)
            combined_X = np.vstack([self.X_scaled, synthetic_X])
        else:
            # Sample from original if n_samples < original size
            indices = np.random.choice(len(self.X_scaled), n_samples, replace=False)
            combined_X = self.X_scaled[indices]
            
        # Generate corresponding synthetic targets
        combined_y = self._generate_synthetic_targets(combined_X)
        
        return combined_X, combined_y
    
    def _generate_synthetic_targets(self, X_synthetic):
        """Generate synthetic targets using simple GPR"""
        synthetic_targets = []
        
        for target_idx in range(self.y_scaled.shape[1]):
            # Fit simple GPR on original data
            kernel = RBF() + WhiteKernel()
            gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gpr.fit(self.X_scaled, self.y_scaled[:, target_idx])
            
            # Predict for synthetic features
            y_pred, y_std = gpr.predict(X_synthetic, return_std=True)
            
            # Add noise proportional to uncertainty
            noise = np.random.normal(0, y_std * 0.1)
            synthetic_targets.append(y_pred + noise)
            
        return np.column_stack(synthetic_targets)
    
    def evaluate_gpr_performance(self, X, y, target_idx=0):
        """Evaluate GPR performance for a specific target"""
        kernel = RBF(length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level_bounds=(1e-5, 1e0))
        
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            random_state=42,
            normalize_y=True,
            n_restarts_optimizer=2
        )
        
        try:
            # Use 3-fold CV or train-test split for small datasets
            if len(X) < 9:  # Too small for 3-fold CV
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y[:, target_idx], test_size=0.3, random_state=42
                )
                gpr.fit(X_train, y_train)
                y_pred = gpr.predict(X_test)
                score = r2_score(y_test, y_pred)
            else:
                cv_scores = cross_val_score(gpr, X, y[:, target_idx], cv=3, scoring='r2')
                score = np.mean(cv_scores)
                
            return score
            
        except Exception as e:
            print(f"Warning: GPR failed for target {target_idx}: {e}")
            return -1.0
    
    def run_power_analysis(self, sample_sizes, n_iterations=3):
        """Run power analysis across different sample sizes"""
        results = []
        
        total_configs = len(sample_sizes) * n_iterations * len(self.target_names)
        current_config = 0
        
        print(f"Running power analysis with {total_configs} configurations...")
        
        for n_samples in sample_sizes:
            print(f"Processing N={n_samples}...")
            
            for iteration in range(n_iterations):
                # Generate synthetic dataset
                X_syn, y_syn = self.generate_synthetic_dataset(n_samples)
                
                # Evaluate each target
                for target_idx, target_name in enumerate(self.target_names):
                    current_config += 1
                    
                    if current_config % 10 == 0:
                        progress = current_config / total_configs * 100
                        print(f"Progress: {progress:.1f}%")
                    
                    r2_score = self.evaluate_gpr_performance(X_syn, y_syn, target_idx)
                    
                    results.append({
                        'n_samples': n_samples,
                        'iteration': iteration,
                        'target': target_name,
                        'r2_score': r2_score
                    })
        
        return pd.DataFrame(results)
    
    def plot_results(self, results_df, fp, ylim=(-1, 2)):
        """Simple plotting of power analysis results with proper y-axis scaling"""
        
        rheo_targets_latex = [r"$\mu_0$", r"$\mu_{\infty}$", r"$\tau_C$", r"$T_{R^1}$", r"$T_{R^2}$", r"$\mu_R$", r"$\sigma_{Y^{0}}$", r"$\tau_{\lambda}$", r"$G_R$", r"$G_C$"]
        
        plt.figure(figsize=(10, 6))

        # Plot: Performance by Target
        target_performance = results_df.groupby(['target', 'n_samples'])['r2_score'].mean().reset_index()

        i = 0
        for target in target_performance['target'].unique():
            data = target_performance[target_performance['target'] == target]
            plt.plot(data['n_samples'], data['r2_score'], marker='o', label=rheo_targets_latex[i])
            i += 1

        plt.xlabel('Sample Size (N)')
        plt.ylabel(r'$R^2$ Score')
        plt.title('Performance by Rheological Target')
        plt.ylim(ylim)  # Set consistent y-axis range
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Add horizontal line at R² = 0 for reference
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print info about clipped values
        min_val = results_df['r2_score'].min()
        max_val = results_df['r2_score'].max()
        if min_val < ylim[0] or max_val > ylim[1]:
            clipped_low = (results_df['r2_score'] < ylim[0]).sum()
            clipped_high = (results_df['r2_score'] > ylim[1]).sum()
            print(f"Plot Info: Y-axis limited to {ylim}")
            print(f"   Data range: {min_val:.3f} to {max_val:.3f}")
            if clipped_low > 0:
                print(f"   {clipped_low} values below {ylim[0]} not shown")
            if clipped_high > 0:
                print(f"   {clipped_high} values above {ylim[1]} not shown")
        
        return