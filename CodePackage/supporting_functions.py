# Imports
import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import csv
import warnings
import seaborn as sns
import openpyxl
import joblib

# Global Variables
abrSubset = ['donors', 'HCT', 'FIB', 'CHOL', 'TRIG', 'HDL', 'LDL', 'WBC', 'RBC', 'HEM', 'MCV', 'MCH', 'MCHC']
rheo_targets = ["MU_0", "MU_INF", "TAU_C", "T_R1", "T_R2", "MU_R", "SIGMA_Y0", "TAU_LAM", "G_R", "G_C"]

def get_fp(cpu="man_dtop", dataFolder="DATA", figuresFolder=""):
    if (cpu == "man_dtop"):
        data_fp = r"/home/msmitty/Documents/TransientBloodRheo_ML/DATA/"
        folder_fp = r"/home/msmitty/Documents/TransientBloodRheo_ML/"
        figures_fp = r"/home/msmitty/Documents/TransientBloodRheo_ML/Writing/Figures/" + figuresFolder
    elif (cpu == "surface"):
        data_fp = r"C:\Users\manni\OneDrive - University of Delaware - o365\Python Projects\TransientBloodRheo_ML\DATA\\"
        folder_fp = r"C:\Users\manni\OneDrive - University of Delaware - o365\Python Projects\TransientBloodRheo_ML"
        figures_fp = r"C:\Users\manni\OneDrive - University of Delaware - o365\Python Projects\TransientBloodRheo_ML\Writing\Figures\\" + figuresFolder
    elif (cpu == "sean"):
        data_fp = r"\TransientBloodRheo_ML\DATA"
        figures_fp = r"\TransientBloodRheo_ML/Writing/Figures" + figuresFolder
    
    else:
        print("Error getting file path")
    
    return folder_fp, data_fp, figures_fp

def loadData(data_fp):
    # Load Data
    rawPhys = pd.read_excel(
        os.path.expanduser(data_fp + r"Armstrong_tESSTV_simplified.xlsx"),
        sheet_name="Physiological_forML"
    )
    rawRheo = pd.read_excel(
        os.path.expanduser(data_fp + r"Armstrong_tESSTV_simplified.xlsx"),
        sheet_name="Rheology_forML"
    )
    
    imputedRheo = pd.read_pickle(data_fp + r"imputedRheo.pkl")

    # Load Variable Dictionary from json
    with open(os.path.expanduser(data_fp + r"physiological_variables.json"), 'r') as f:
        physDict = json.load(f)
    with open(os.path.expanduser(data_fp + r"rheology_variables.json"), 'r') as f:
        rheoDict = json.load(f)
    
    return rawPhys, rawRheo, physDict, rheoDict, imputedRheo

def saveData(fp="", df=None, npy=None, pca=None):
    if df is not None:
        df.to_pickle(fp)
        print("Saved DataFrame as pickle to: ", fp)
    if npy is not None:
        np.save(fp, npy)
        print("Saved NumPy array to: ", fp)
    if pca is not None:
        joblib.dump(pca, fp)
        print("Saved PCA model to: ", fp)
        
def loadPCA(fp):
    pca_model = joblib.load(os.path.expanduser(fp + "PCA/pca_phys_model.pkl"))
    X_pca = np.load(os.path.expanduser(fp + "PCA/pca_phys.npy"))
    
    return pca_model, X_pca