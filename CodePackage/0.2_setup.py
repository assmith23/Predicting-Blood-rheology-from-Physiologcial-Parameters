#Imports
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from pathlib import Path

def setup_paths(data_folder="Data", figures_folder="Figures"):
    """
    Automatically setup file paths based on current working directory.
    
    Parameters:
    -----------
    data_folder : str, default "Data"
        Name of the data folder relative to current directory
    figures_folder : str, default "Figures"
        Name of the figures folder relative to current directory
        
    Returns:
    --------
    tuple: (project_path, data_path, figures_path)
    """
    # Get current working directory
    current_dir = Path.cwd()
    
    # Setup paths
    project_path = current_dir
    data_path = current_dir / data_folder
    figures_path = current_dir / figures_folder
    
    # Create directories if they don't exist
    data_path.mkdir(exist_ok=True)
    figures_path.mkdir(exist_ok=True)
    
    # Convert to strings for compatibility
    project_path_str = str(project_path)
    data_path_str = str(data_path)
    figures_path_str = str(figures_path)
    
    print(f"Project directory: {project_path_str}")
    print(f"Data directory: {data_path_str}")
    print(f"Figures directory: {figures_path_str}")
    
    return project_path_str, data_path_str, figures_path_str

def get_fp(cpu="auto", dataFolder="DATA", figuresFolder=""):
    """
    Enhanced version that supports both manual and automatic path detection.
    
    Parameters:
    -----------
    cpu : str, default "auto"
        Either "auto" for automatic detection or specific computer names
    dataFolder : str, default "DATA"
        Name of data folder
    figuresFolder : str, default ""
        Subfolder within figures directory
        
    Returns:
    --------
    tuple: (folder_fp, data_fp, figures_fp)
    """
    if cpu == "auto":
        project_path, data_path, figures_path = setup_paths(dataFolder, "figures")
        if figuresFolder:
            figures_path = os.path.join(figures_path, figuresFolder)
            os.makedirs(figures_path, exist_ok=True)
        return project_path, data_path, figures_path
    
    # Keep original functionality for specific computers
    elif cpu == "man_dtop":
        data_fp = r"/home/msmitty/Documents/TransientBloodRheo_ML/Data/"
        folder_fp = r"/home/msmitty/Documents/TransientBloodRheo_ML/"
        figures_fp = r"/home/msmitty/Documents/TransientBloodRheo_ML/WritingMaterials/Figures/" + figuresFolder
    elif cpu == "surface":
        data_fp = r"C:\Users\manni\OneDrive - University of Delaware - o365\Python Projects\TransientBloodRheo_ML\Data\\"
        folder_fp = r"C:\Users\manni\OneDrive - University of Delaware - o365\Python Projects\TransientBloodRheo_ML"
        figures_fp = r"C:\Users\manni\OneDrive - University of Delaware - o365\Python Projects\TransientBloodRheo_ML\WritingMaterials\Figures\\" + figuresFolder
    elif cpu == "sean":
        data_fp = r"\TransientBloodRheo_ML\DATA"
        figures_fp = r"\TransientBloodRheo_ML/Writing/Figures" + figuresFolder
        folder_fp = r"\TransientBloodRheo_ML"
    else:
        print("Error getting file path")
        return None, None, None
    
    return folder_fp, data_fp, figures_fp

def check_data_files(data_path):
    """
    Check if required data files exist in the data directory.
    
    Parameters:
    -----------
    data_path : str
        Path to data directory
        
    Returns:
    --------
    dict: Dictionary with file existence status
    """
    required_files = [
        "Armstrong_tESSTV_simplified.xlsx",
        "imputedRheo.pkl",
        "physiological_variables.json",
        "rheology_variables.json"
    ]
    
    file_status = {}
    for file in required_files:
        file_path = os.path.join(data_path, file)
        file_status[file] = os.path.exists(file_path)
        
    # Print status
    print("\nData file status:")
    for file, exists in file_status.items():
        status = "✓ Found" if exists else "✗ Missing"
        print(f"  {file}: {status}")
    
    return file_status

def loadData(data_fp):
    """
    Load data from the specified data folder path.
    
    Parameters:
    -----------
    data_fp : str
        Path to data folder
        
    Returns:
    --------
    tuple: (rawPhys, rawRheo, physDict, rheoDict, imputedRheo)
    """
    try:
        # Check if files exist first
        file_status = check_data_files(data_fp)
        missing_files = [f for f, exists in file_status.items() if not exists]
        
        if missing_files:
            print(f"\nError: Missing required files: {missing_files}")
            print(f"Please ensure these files are in: {data_fp}")
            return None, None, None, None, None
        
        print("\nLoading data files...")
        
        # Load Excel data
        excel_path = os.path.join(data_fp, "Armstrong_tESSTV_simplified.xlsx")
        rawPhys = pd.read_excel(excel_path, sheet_name="Physiological_forML")
        rawRheo = pd.read_excel(excel_path, sheet_name="Rheology_forML")
        print("✓ Loaded Excel data")
        
        # Load pickled data
        pickle_path = os.path.join(data_fp, "imputedRheo.pkl")
        imputedRheo = pd.read_pickle(pickle_path)
        print("✓ Loaded pickled rheology data")
        
        # Load JSON dictionaries
        phys_json_path = os.path.join(data_fp, "physiological_variables.json")
        with open(phys_json_path, 'r') as f:
            physDict = json.load(f)
        print("✓ Loaded physiological variables dictionary")
        
        rheo_json_path = os.path.join(data_fp, "rheology_variables.json")
        with open(rheo_json_path, 'r') as f:
            rheoDict = json.load(f)
        print("✓ Loaded rheology variables dictionary")
        
        print(f"\nData loading complete!")
        print(f"Physiological data shape: {rawPhys.shape}")
        print(f"Rheology data shape: {rawRheo.shape}")
        print(f"Imputed rheology data shape: {imputedRheo.shape}")
        
        return rawPhys, rawRheo, physDict, rheoDict, imputedRheo
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None, None

def saveData(fp="", df=None, npy=None, pca=None):
    """
    Save data in various formats.
    
    Parameters:
    -----------
    fp : str
        File path for saving
    df : pandas.DataFrame, optional
        DataFrame to save as pickle
    npy : numpy.ndarray, optional
        NumPy array to save
    pca : sklearn object, optional
        PCA model to save using joblib
    """
    if df is not None:
        df.to_pickle(fp)
        print("Saved DataFrame as pickle to: ", fp)
    if npy is not None:
        np.save(fp, npy)
        print("Saved NumPy array to: ", fp)
    if pca is not None:
        joblib.dump(pca, fp)
        print("Saved PCA model to: ", fp)
        
def loadPCA(data_fp):
    """
    Load PCA model and data.
    
    Parameters:
    -----------
    data_fp : str
        Path to data folder
        
    Returns:
    --------
    tuple: (pca_model, X_pca)
    """
    try:
        pca_folder = os.path.join(data_fp, "PCA")
        pca_model_path = os.path.join(pca_folder, "pca_phys_model.pkl")
        pca_data_path = os.path.join(pca_folder, "pca_phys.npy")
        
        pca_model = joblib.load(pca_model_path)
        X_pca = np.load(pca_data_path)
        
        print("✓ Loaded PCA model and data")
        return pca_model, X_pca
        
    except Exception as e:
        print(f"Error loading PCA: {str(e)}")
        return None, None

def initialize_project():
    """
    Initialize the project by setting up paths and loading data.
    
    Returns:
    --------
    dict: Dictionary containing all loaded data and paths
    """
    print("=== Project Initialization ===")
    
    # Setup paths automatically
    project_path, data_path, figures_path = setup_paths()
    
    # Load data
    rawPhys, rawRheo, physDict, rheoDict, imputedRheo = loadData(data_path)
    
    # Try to load PCA if available
    pca_model, X_pca = loadPCA(data_path)
    
    # Return everything in a dictionary for easy access
    project_data = {
        'paths': {
            'project': project_path,
            'data': data_path,
            'figures': figures_path
        },
        'data': {
            'rawPhys': rawPhys,
            'rawRheo': rawRheo,
            'physDict': physDict,
            'rheoDict': rheoDict,
            'imputedRheo': imputedRheo
        },
        'models': {
            'pca_model': pca_model,
            'X_pca': X_pca
        }
    }
    
    return project_data

# Quick setup function for immediate use
def quick_setup():
    """
    Quick setup that returns the most commonly used items.
    
    Returns:
    --------
    tuple: (data_path, rawPhys, rawRheo, imputedRheo)
    """
    _, data_path, _ = setup_paths()
    rawPhys, rawRheo, _, _, imputedRheo = loadData(data_path)
    return data_path, rawPhys, rawRheo, imputedRheo

"""
# Example usage
if __name__ == "__main__":
    project_data = initialize_project()
    print("\nProject initialized successfully!")
    print(f"Project path: {project_data['paths']['project']}")
    print(f"Data path: {project_data['paths']['data']}")
    print(f"Figures path: {project_data['paths']['figures']}")
    
    # Access raw data
    rawPhys = project_data['data']['rawPhys']
    rawRheo = project_data['data']['rawRheo']
    print(f"\nRaw Physiological Data Shape: {rawPhys.shape}")
    print(f"Raw Rheology Data Shape: {rawRheo.shape}")
"""

def main():
    """
    Main function to run the project initialization and quick setup.
    """
    print("Running main function...")
    project_data = initialize_project()
    
    # Example of accessing data
    print("\nAccessing raw physiological data:")
    print(project_data['data']['rawPhys'].head())
    
    print("\nAccessing raw rheology data:")
    print(project_data['data']['rawRheo'].head())