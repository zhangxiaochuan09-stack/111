!cp -r /kaggle/input/autogluon-package/* /kaggle/working/
!pip install -f --quiet --no-index --find-links='/kaggle/input/autogluon-package' 'autogluon.tabular-1.3.1-py3-none-any.whl'
!cp -r /kaggle/input/scikit-package/* /kaggle/working/
!pip install -f --quiet --no-index --find-links='/kaggle/input/scikit-package' 'scikit_learn-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl' 
from autogluon.tabular import TabularDataset, TabularPredictor
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/
!rm -rf /kaggle/working/*
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
output_dfs = []


"""This notebook implements a modular and robust machine learning pipeline for polymer property prediction. Key features include:

Config-driven pipeline: All model and preprocessing parameters are controlled via a config object for reproducibility and flexibility.
Data preparation: SMILES strings are canonicalized, deduplicated, and optionally augmented. Features are generated using molecular fingerprints and descriptors.
Feature engineering: Includes robust feature selection, correlation pruning, variance thresholding, and optional PCA.
Modeling: Supports stacking with XGBoost, LightGBM, and CatBoost as base models, and XGBoost as the meta-model. Neural network and traditional ML models are also supported.
Cross-validation: Stratified and standard CV splits are available, with holdout sets for final evaluation."""


import glob
import os
import time
import random
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, ElasticNetCV

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdmolops, Lipinski, Crippen
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator

from mordred import Calculator, descriptors as mordred_descriptors

import shap

from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# torchinfo is optional, only used in show_model_summary
try:
    from torchinfo import summary
except ImportError:
    summary = None

# Data paths
RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Create the NeurIPS directory if it does not exist
os.makedirs("NeurIPS", exist_ok=True)
class Config:
    useAllDataForTraining = False
    use_standard_scaler = True  # Set to True to use StandardScaler, False to skip scaling
    # Set to True to calculate Mordred descriptors in featurization
    use_least_important_features_all_methods = True  # Set to True to call get_least_important_features_all_methods
    use_variance_threshold = False  # Set to True to enable VarianceThreshold feature selection
    enable_param_tuning = False  # Set to True to enable XGB hyperparameter tuning
    debug = False

    use_descriptors = False  # Set to True to include RDKit descriptors, False to skip
    use_mordred = True
    # Control inclusion of AtomPair and TopologicalTorsion fingerprints
    use_maccs_fp = False
    use_morgan_fp = False
    use_atom_pair_fp = False
    use_torsion_fp = False
    use_chemberta = False
    chemberta_pooling = 'max'  # can be 'mean', 'max', 'cls', or 'pooler'
    # Include MACCS keys fingerprint

    search_nn = False
    use_stacking = False
    model_name = 'xgb'  
    # Options: ['autogluon', xgb', 'catboost', 'lgbm', 'extratrees', 'randomforest', 'tabnet', 'hgbm', 'nn']

    # Don't change
    # Choose importance method: 'feature_importances_' or 'permutation_importance'
    feature_importance_method = 'permutation_importance'
    use_cross_validation = True  # Set to False to use a single split for speed
    use_pca = False  # Set to True to enable PCA
    pca_variance = 0.9999  # Fraction of variance to keep in PCA
    use_external_data = True  # Set to True to use external datasets
    use_augmentation = False  # Set to True to use augment_dataset
    add_gaussian = False  # Set to True to enable Gaussian Mixture Model-based augmentation
    random_state = 42


    # Number of least important features to drop
    # Use a nested dictionary for model- and label-specific n_least_important_features
    n_least_important_features = {
        'xgb':     {'Tg': 20, 'FFV': 20, 'Tc': 22, 'Density': 19, 'Rg': 19},
        'catboost':{'Tg': 15, 'FFV': 15, 'Tc': 18, 'Density': 15, 'Rg': 15},
        'lgbm':    {'Tg': 18, 'FFV': 18, 'Tc': 20, 'Density': 17, 'Rg': 17},
        'extratrees':{'Tg': 22, 'FFV': 15, 'Tc': 10, 'Density': 25, 'Rg': 5},
        'randomforest':{'Tg': 21, 'FFV': 19, 'Tc': 21, 'Density': 18, 'Rg': 18},
        'balancedrf':{'Tg': 20, 'FFV': 20, 'Tc': 20, 'Density': 20, 'Rg': 20},
    }

    # Path for permutation importance log file
    permutation_importance_log_path = "log/permutation_importance_log.xlsx"

    correlation_threshold_value = 0.96
    correlation_thresholds = {
        "Tg": correlation_threshold_value,
        "FFV": correlation_threshold_value,
        "Tc": correlation_threshold_value,
        "Density": correlation_threshold_value,
        "Rg": correlation_threshold_value
    }

# Create a single config instance to use everywhere
config = Config()

if config.debug or config.search_nn:
    config.use_cross_validation = False

# --- XGB Hyperparameter Tuning DB Utilities ---
import sqlite3
import hashlib
import json

def init_chemberta():
    model_name = "/kaggle/input/c/transformers/default/1/ChemBERTa-77M-MLM"
    chemberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
    chemberta_model = AutoModel.from_pretrained(model_name)
    chemberta_model.eval()
    return chemberta_tokenizer, chemberta_model

def get_chemberta_embedding(smiles, embedding_dim=384):
    """
    Returns ChemBERTa embedding for a single SMILES string.
    Pads/truncates to embedding_dim if needed.
    """
    if smiles is None or not isinstance(smiles, str) or len(smiles) == 0:
        return np.zeros(embedding_dim)
    try:
        # Add pooling argument with default 'mean'
        pooling = getattr(config, 'chemberta_pooling', 'mean')  # can be 'mean', 'max', 'cls', 'pooler'
        chemberta_tokenizer, chemberta_model = init_chemberta()
        inputs = chemberta_tokenizer([smiles], padding=True, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = chemberta_model(**inputs)
            if pooling == 'pooler' and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                emb = outputs.pooler_output.squeeze(0)
            elif pooling == 'cls' and hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
            elif pooling == 'max' and hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state.max(dim=1).values.squeeze(0)
            elif pooling == 'mean' and hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            else:
                raise ValueError("Cannot extract embedding from model output")
            emb_np = emb.cpu().numpy()
            # Pad or truncate if needed
            if emb_np.shape[0] < embedding_dim:
                emb_np = np.pad(emb_np, (0, embedding_dim - emb_np.shape[0]))
            elif emb_np.shape[0] > embedding_dim:
                emb_np = emb_np[:embedding_dim]
            return emb_np
    except Exception as e:
        print(f"ChemBERTa embedding failed for SMILES '{smiles}': {e}")
        return np.zeros(embedding_dim)
    
def init_xgb_tuning_db(db_path="xgb_tuning.db"):
    """Initialize the XGB tuning database and return all existing results."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS xgb_tuning
                 (param_hash TEXT PRIMARY KEY, params TEXT, score REAL)''')
    c.execute('SELECT params, score FROM xgb_tuning')
    results = c.fetchall()
    conn.close()
    return [(json.loads(params), score) for params, score in results]

def get_param_hash(params):
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()

def check_db_for_params(db_path, param_hash):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT score FROM xgb_tuning WHERE param_hash=?', (param_hash,))
    result = c.fetchone()
    conn.close()
    return result is not None

def save_result_to_db(db_path, param_hash, params, score):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO xgb_tuning (param_hash, params, score)
                 VALUES (?, ?, ?)''', (param_hash, json.dumps(params, sort_keys=True), score))
    conn.commit()
    conn.close()

# --- XGB Hyperparameter Grid Search with DB Caching ---
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid

def xgb_grid_search_with_db(X, y, param_grid, db_path="xgb_tuning.db"):
    """
    For each param set in grid, check DB. If not present, train and save result.
    param_grid: dict of param lists, e.g. {'max_depth':[3,5], 'learning_rate':[0.01,0.1]}
    """
    tried = 0
    best_score = None
    best_params = None
    # Split X, y into train/val for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    for params in ParameterGrid(param_grid):
        param_hash = get_param_hash(params)
        if check_db_for_params(db_path, param_hash):
            print(f"Skipping already tried params: {params}")
            continue
        # print(f"Trying params: {json.dumps(params, sort_keys=True)}")
        model = XGBRegressor(**params)
        # Provide eval_set for early stopping
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        from sklearn.metrics import mean_absolute_error
        score = mean_absolute_error(y_val, y_pred)
        print(f"Result: MAE={score:.6f} for params: {json.dumps(params, sort_keys=True)}")
        # For MAE, lower is better
        if (best_score is None) or (score < best_score):
            best_score = score
            best_params = params.copy()
            print(f"New best MAE: {best_score:.6f} with params: {json.dumps(best_params, sort_keys=True)}")
        save_result_to_db(db_path, param_hash, params, score)
        tried += 1
    print(f"Tried {tried} new parameter sets.")
    if best_score is not None:
        print(f"Best score overall: {best_score:.6f} with params: {json.dumps(best_params, sort_keys=True)}")

from sklearn.linear_model import RidgeCV, ElasticNetCV

def drop_correlated_features(df, threshold=0.95):
    """
    Drops columns in a DataFrame that are highly correlated with other columns.
    Only one of each pair of correlated columns is kept.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Correlation threshold for dropping columns (default 0.95).

    Returns:
        pd.DataFrame: DataFrame with correlated columns dropped.
        list: List of dropped column names.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

def get_canonical_smiles(smiles):
        """Convert SMILES to canonical form for consistency"""
        if not RDKIT_AVAILABLE:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return smiles

"""
Load competition data with complete filtering of problematic polymer notation
"""

print("Loading competition data...")
train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')

if config.debug:
    print("   Debug mode: sampling 1000 training examples")
    train = train.sample(n=1000, random_state=42).reset_index(drop=True)

print(f"Training data shape: {train.shape}, Test data shape: {test.shape}")

def clean_and_validate_smiles(smiles):
    """Completely clean and validate SMILES, removing all problematic patterns"""
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    
    # List of all problematic patterns we've seen
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        # Additional patterns that cause issues
        '([R])', '([R1])', '([R2])', 
    ]
    
    # Check for any bad patterns
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    
    # Additional check: if it contains ] followed by [ without valid atoms, likely polymer notation
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    
    # Try to parse with RDKit if available
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None
    
    # If RDKit not available, return cleaned SMILES
    return smiles

# Clean and validate all SMILES
print("Cleaning and validating SMILES...")
train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)

# Remove invalid SMILES
invalid_train = train['SMILES'].isnull().sum()
invalid_test = test['SMILES'].isnull().sum()

print(f"   Removed {invalid_train} invalid SMILES from training data")
print(f"   Removed {invalid_test} invalid SMILES from test data")

train = train[train['SMILES'].notnull()].reset_index(drop=True)
test = test[test['SMILES'].notnull()].reset_index(drop=True)

print(f"   Final training samples: {len(train)}")
print(f"   Final test samples: {len(test)}")

def add_extra_data_clean(df_train, df_extra, target):
    """Add external data with thorough SMILES cleaning"""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    print(f"      Processing {len(df_extra)} {target} samples...")
    
    # Clean external SMILES
    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    
    # Remove invalid SMILES and missing targets
    before_filter = len(df_extra)
    df_extra = df_extra[df_extra['SMILES'].notnull()]
    df_extra = df_extra.dropna(subset=[target])
    after_filter = len(df_extra)
    
    print(f"      Kept {after_filter}/{before_filter} valid samples")
    
    if len(df_extra) == 0:
        print(f"      No valid data remaining for {target}")
        return df_train
    
    # Group by canonical SMILES and average duplicates
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Fill missing values
    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['SMILES']==smile, target] = \
                df_extra[df_extra['SMILES']==smile][target].values[0]
            filled_count += 1
    
    # Add unique SMILES
    extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()
    if len(extra_to_add) > 0:
        for col in TARGETS:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        
        extra_to_add = extra_to_add[['SMILES'] + TARGETS]
        df_train = pd.concat([df_train, extra_to_add], axis=0, ignore_index=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'      {target}: +{n_samples_after-n_samples_before} samples, +{len(unique_smiles_extra)} unique SMILES')
    print(f"      Filled {filled_count} missing entries in train for {target}")
    print(f"      Added {len(extra_to_add)} new entries for {target}")
    return df_train

# Load external datasets with robust error handling
print("\n📂 Loading external datasets...")

external_datasets = []

# Function to safely load datasets
def safe_load_dataset(path, target, processor_func, description):
    try:
        if path.endswith('.xlsx'):
            data = pd.read_excel(path)
        else:
            data = pd.read_csv(path)
        
        data = processor_func(data)
        external_datasets.append((target, data))
        print(f"   ✅ {description}: {len(data)} samples")
        return True
    except Exception as e:
        print(f"   ⚠️ {description} failed: {str(e)[:100]}")
        return False

# Load each dataset
safe_load_dataset(
    '/kaggle/input/tc-smiles/Tc_SMILES.csv',
    'Tc',
    lambda df: df.rename(columns={'TC_mean': 'Tc'}),
    'Tc data'
)

safe_load_dataset(
    '/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv',
    'Tg', 
    lambda df: df[['SMILES', 'Tg']] if 'Tg' in df.columns else df,
    'TgSS enriched data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    'Tg',
    lambda df: df[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'}),
    'JCIM Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
    'Tg',
    lambda df: df.rename(columns={'Tg [K]': 'Tg'}).assign(Tg=lambda x: x['Tg'] - 273.15),
    'Xlsx Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_dnst1.xlsx',
    'Density',
    lambda df: df.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
                .query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
                .assign(Density=lambda x: x['Density'].astype(float) - 0.118),
    'Density data'
)

safe_load_dataset(
    BASE_PATH + 'train_supplement/dataset4.csv',
    'FFV', 
    lambda df: df[['SMILES', 'FFV']] if 'FFV' in df.columns else df,
    'dataset 4'
)

# Integrate external data
print("\n🔄 Integrating external data...")
train_extended = train[['SMILES'] + TARGETS].copy()

if getattr(config, "use_external_data", True) and  not config.debug:
    for target, dataset in external_datasets:
        print(f"   Processing {target} data...")
        train_extended = add_extra_data_clean(train_extended, dataset, target)

print(f"\n📊 Final training data:")
print(f"   Original samples: {len(train)}")
print(f"   Extended samples: {len(train_extended)}")
print(f"   Gain: +{len(train_extended) - len(train)} samples")

for target in TARGETS:
    count = train_extended[target].notna().sum()
    original_count = train[target].notna().sum() if target in train.columns else 0
    gain = count - original_count
    print(f"   {target}: {count:,} samples (+{gain})")

print(f"\n✅ Data integration complete with clean SMILES!")

def separate_subtables(train_df):
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    subtables = {}
    for label in labels:
        # Filter out NaNs, select columns, reset index
        subtables[label] = train_df[train_df[label].notna()][['SMILES', label]].reset_index(drop=True)

    # Optional: Debugging
    for label in subtables:
        print(f"{label} NaNs per column:")
        print(subtables[label].isna().sum())
        print(subtables[label].shape)
        print("-" * 40)

    return subtables

def augment_smiles_dataset(smiles_list, labels, num_augments=3):
    """
    Augments a list of SMILES strings by generating randomized versions.

    Parameters:
        smiles_list (list of str): Original SMILES strings.
        labels (list or np.array): Corresponding labels.
        num_augments (int): Number of augmentations per SMILES.

    Returns:
        tuple: (augmented_smiles, augmented_labels)
    """
    augmented_smiles = []
    augmented_labels = []

    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        # Add original
        augmented_smiles.append(smiles)
        augmented_labels.append(label)
        # Add randomized versions
        for _ in range(num_augments):
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented_smiles.append(rand_smiles)
            augmented_labels.append(label)

    return augmented_smiles, np.array(augmented_labels)

mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)
def build_mordred_descriptors(smiles_list):
    # Build Mordred descriptors for test
    mols_test = [Chem.MolFromSmiles(s) for s in smiles_list]
    desc_test = mordred_calc.pandas(mols_test, nproc=1)

    # Make columns string & numeric only (no dropping beyond that)
    desc_test.columns = desc_test.columns.map(str)
    desc_test = desc_test.select_dtypes(include=[np.number]).copy()
    desc_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    return desc_test

from rdkit.Chem import Crippen, Lipinski

def smiles_to_combined_fingerprints_with_descriptors(smiles_list):
    # Set fingerprint parameters inside the function
    radius = 2
    n_bits = 128

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits) if getattr(Config, "use_morgan_fp", True) else None
    atom_pair_gen = GetAtomPairGenerator(fpSize=n_bits) if getattr(Config, 'use_atom_pair_fp', False) else None
    torsion_gen = GetTopologicalTorsionGenerator(fpSize=n_bits) if getattr(Config, 'use_torsion_fp', False) else None
    
    fp_len = (n_bits if getattr(Config, 'use_morgan_fp', False) else 0) \
           + (n_bits if getattr(Config, 'use_atom_pair_fp', False) else 0) \
           + (n_bits if getattr(Config, 'use_torsion_fp', False) else 0) \
           + (167 if getattr(Config, 'use_maccs_fp', True) else 0)
    if getattr(Config, 'use_chemberta', False):
        fp_len += 384
        
    fingerprints = []
    descriptors = []
    valid_smiles = []
    invalid_indices = []
    use_any_fp = getattr(Config, "use_morgan_fp", False) or getattr(Config, "use_atom_pair_fp", False) or getattr(Config, "use_torsion_fp", False) or getattr(Config, "use_maccs_fp", False) or getattr(Config, 'use_chemberta', False)

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Fingerprints (No change from your code)
            if use_any_fp:
                fps = []
                if getattr(Config, "use_morgan_fp", True) and generator is not None:
                    fps.append(np.array(generator.GetFingerprint(mol)))
                if atom_pair_gen:
                    fps.append(np.array(atom_pair_gen.GetFingerprint(mol)))
                if torsion_gen:
                    fps.append(np.array(torsion_gen.GetFingerprint(mol)))
                if getattr(Config, 'use_maccs_fp', True):
                    fps.append(np.array(MACCSkeys.GenMACCSKeys(mol)))
                if getattr(Config, "use_chemberta", False):
                    emb = get_chemberta_embedding(smiles)
                    fps.append(emb)
                
                combined_fp = np.concatenate(fps)
                fingerprints.append(combined_fp)

            if getattr(Config, 'use_descriptors', True):
                descriptor_values = {}
                for name, func in Descriptors.descList:
                    try:
                        descriptor_values[name] = func(mol)
                    except:
                        print(f"Descriptor {name} failed for SMILES at index {i}")
                        descriptor_values[name] = None

                # try:
                # --- Features for Rigidity and Complexity (for Tg, FFV) ---
                try:
                    num_heavy_atoms = mol.GetNumHeavyAtoms()
                except Exception as e:
                    num_heavy_atoms = 0

                # --- Features for Rigidity and Complexity (for Tg, FFV) ---
                try:
                    descriptor_values['NumAromaticRings'] = Lipinski.NumAromaticRings(mol)
                except Exception as e:
                    descriptor_values['NumAromaticRings'] = None
                try:
                    num_sp3_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
                    descriptor_values['FractionCSP3'] = num_sp3_carbons / num_heavy_atoms if num_heavy_atoms > 0 else 0
                except Exception as e:
                    descriptor_values['FractionCSP3'] = None

                # --- Features for Bulkiness and Shape (for FFV) ---
                try:
                    descriptor_values['MolMR'] = Crippen.MolMR(mol) # Molar Refractivity (volume)
                except Exception as e:
                    descriptor_values['MolMR'] = None
                try:
                    descriptor_values['LabuteASA'] = Descriptors.LabuteASA(mol) # Accessible surface area
                except Exception as e:
                    descriptor_values['LabuteASA'] = None

                # --- Features for Heavy Atoms (for Density) ---
                try:
                    descriptor_values['NumFluorine'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
                except Exception as e:
                    descriptor_values['NumFluorine'] = None
                try:
                    descriptor_values['NumChlorine'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)
                except Exception as e:
                    descriptor_values['NumChlorine'] = None

                # --- Features for Intermolecular Forces (for Tc) ---
                try:
                    descriptor_values['NumHDonors'] = Lipinski.NumHDonors(mol)
                except Exception as e:
                    descriptor_values['NumHDonors'] = None
                try:
                    descriptor_values['NumHAcceptors'] = Lipinski.NumHAcceptors(mol)
                except Exception as e:
                    descriptor_values['NumHAcceptors'] = None

                # --- Features for Branching and Flexibility (for Rg) ---
                try:
                    descriptor_values['BalabanJ'] = Descriptors.BalabanJ(mol) # Topological index sensitive to branching
                except Exception as e:
                    descriptor_values['BalabanJ'] = None
                try:
                    descriptor_values['Kappa2'] = Descriptors.Kappa2(mol) # Molecular shape index
                except Exception as e:
                    descriptor_values['Kappa2'] = None
                try:
                    descriptor_values['NumRotatableBonds'] = CalcNumRotatableBonds(mol) # Flexibility
                except Exception as e:
                    descriptor_values['NumRotatableBonds'] = None
                
                # Graph-based features
                try:
                    adj = rdmolops.GetAdjacencyMatrix(mol)
                    G = nx.from_numpy_array(adj)
                    if nx.is_connected(G):
                        descriptor_values['graph_diameter'] = nx.diameter(G)
                        descriptor_values['avg_shortest_path'] = nx.average_shortest_path_length(G)
                    else:
                        descriptor_values['graph_diameter'], descriptor_values['avg_shortest_path'] = 0, 0
                    descriptor_values['num_cycles'] = len(list(nx.cycle_basis(G)))
                except:
                    print(f"Graph features failed for SMILES at index {i}")
                    descriptor_values['graph_diameter'], descriptor_values['avg_shortest_path'], descriptor_values['num_cycles'] = None, None, None

                descriptors.append(descriptor_values)
            else:
                descriptors.append(None)
            valid_smiles.append(smiles)
        else:
            if use_any_fp: fingerprints.append(np.zeros(fp_len))
            if getattr(Config, "use_chemberta", False):
                descriptors.append({f'chemberta_emb_{j}': 0.0 for j in range(384)})
            else:
                descriptors.append(None)
            valid_smiles.append(None)
            invalid_indices.append(i)

    fingerprints_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fp_len)]) if use_any_fp else pd.DataFrame()
    descriptors_df = pd.DataFrame([d for d in descriptors if d is not None]) if any(d is not None for d in descriptors) else pd.DataFrame()

    if getattr(Config, 'use_mordred', False):
        mordred_df = build_mordred_descriptors(smiles_list)
        if descriptors_df.empty:
            descriptors_df = mordred_df
        else:
            descriptors_df = pd.concat([descriptors_df.reset_index(drop=True), mordred_df], axis=1)
    
    # Keep only unique columns in descriptors_df
    if not descriptors_df.empty:
        descriptors_df = descriptors_df.loc[:, ~descriptors_df.columns.duplicated()]
    return fingerprints_df, descriptors_df, valid_smiles, invalid_indices

required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}

# Utility function to combine train and val sets into X_all and y_all
def combine_train_val(X_train, X_val, y_train, y_val):
    X_train = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
    X_val = pd.DataFrame(X_val) if isinstance(X_val, np.ndarray) else X_val
    y_train = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
    y_val = pd.Series(y_val) if isinstance(y_val, np.ndarray) else y_val
    X_all = pd.concat([X_train, X_val], axis=0)
    y_all = pd.concat([y_train, y_val], axis=0)
    return X_all, y_all

# --- PCA utility for train/test transformation ---
def apply_pca(X_train, X_test=None, verbose=True):
    pca = PCA(n_components=config.pca_variance, svd_solver='full', random_state=getattr(config, 'random_state', 42))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test) if X_test is not None else None
    if verbose:
        print(f"[PCA] Reduced train shape: {X_train.shape} -> {X_train_pca.shape} (kept {pca.n_components_} components, {100*pca.explained_variance_ratio_.sum():.4f}% variance)")
    return X_train_pca, X_test_pca, pca

def augment_dataset(X, y, n_samples=1000, n_components=5, random_state=None):
    """
    Augments a dataset using Gaussian Mixture Models.

    Parameters:
    - X: pd.DataFrame or np.ndarray — feature matrix
    - y: pd.Series or np.ndarray — target values
    - n_samples: int — number of synthetic samples to generate
    - n_components: int — number of GMM components
    - random_state: int — random seed for reproducibility

    Returns:
    - X_augmented: pd.DataFrame — augmented feature matrix
    - y_augmented: pd.Series — augmented target values
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame or a NumPy array")

    X.columns = X.columns.astype(str)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or a NumPy array")

    df = X.copy()
    df['Target'] = y.values

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)

    synthetic_data, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)

    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

    X_augmented = augmented_df.drop(columns='Target')
    y_augmented = augmented_df['Target']

    return X_augmented, y_augmented

# --- Outlier Detection Summary Function ---
def display_outlier_summary(y, X=None, name="target", z_thresh=3, iqr_factor=1.5, iso_contamination=0.01, lof_contamination=0.01):
    """
    Display the percentage of data flagged as outlier by Z-score, IQR, Isolation Forest, and LOF.
    y: 1D array-like (target or feature)
    X: 2D array-like (feature matrix, required for Isolation Forest/LOF)
    name: str, name of the variable being checked
    """
    print(f"\nOutlier summary for: {name}")
    y = np.asarray(y)
    n = len(y)
    # Z-score
    z_scores = (y - np.mean(y)) / np.std(y)
    z_outliers = np.abs(z_scores) > z_thresh
    print(f"Z-score > {z_thresh}: {np.sum(z_outliers)} / {n} ({100*np.mean(z_outliers):.2f}%)")

    # IQR
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower = Q1 - iqr_factor * IQR
    upper = Q3 + iqr_factor * IQR
    iqr_outliers = (y < lower) | (y > upper)
    print(f"IQR (factor {iqr_factor}): {np.sum(iqr_outliers)} / {n} ({100*np.mean(iqr_outliers):.2f}%)")

    # Isolation Forest (if X provided)
    if X is not None:
        try:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=iso_contamination, random_state=42)
            iso_out = iso.fit_predict(X)
            iso_outliers = iso_out == -1
            print(f"Isolation Forest (contamination={iso_contamination}): {np.sum(iso_outliers)} / {len(iso_outliers)} ({100*np.mean(iso_outliers):.2f}%)")
        except Exception as e:
            print(f"Isolation Forest failed: {e}")
        # Local Outlier Factor
        try:
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(n_neighbors=20, contamination=lof_contamination)
            lof_out = lof.fit_predict(X)
            lof_outliers = lof_out == -1
            print(f"Local Outlier Factor (contamination={lof_contamination}): {np.sum(lof_outliers)} / {len(lof_outliers)} ({100*np.mean(lof_outliers):.2f}%)")
        except Exception as e:
            print(f"Local Outlier Factor failed: {e}")
    else:
        print("Isolation Forest/LOF skipped (X not provided)")


train_df=train_extended
test_df=test
subtables = separate_subtables(train_df)

test_smiles = test_df['SMILES'].tolist()
test_ids = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
#labels = ['Tc']

# Save importance_df to Excel log file, one sheet per label
def save_importance_to_excel(importance_df, label, log_path):
    import os
    from openpyxl import load_workbook
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        with pd.ExcelWriter(log_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            importance_df.to_excel(writer, sheet_name=label, index=False)
    else:
        with pd.ExcelWriter(log_path, engine='openpyxl') as writer:
            importance_df.to_excel(writer, sheet_name=label, index=False)

def get_least_important_features_all_methods(X, y, label, model_name=None):
    """
    Remove features in three steps:
    1. Remove features with model.feature_importances_ <= 0
    2. Remove features with permutation_importance <= 0
    3. Remove features with SHAP importance <= 0
    Returns a list of features to remove (union of all three criteria).
    """    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=config.random_state)
    model_type = (model_name or getattr(config, 'model_name', 'xgb'))
    if model_type == 'xgb':
        model = XGBRegressor(random_state=config.random_state, n_jobs=-1, verbosity=0, early_stopping_rounds=50, eval_metric="mae", objective="reg:absoluteerror")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif model_type == 'catboost':
        model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, loss_function='MAE', eval_metric='MAE', random_seed=config.random_state, verbose=False)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50, use_best_model=True)
    elif model_type == 'lgbm':
        model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, reg_lambda=1.0, objective='mae', random_state=config.random_state, verbose=-1, verbosity=-1)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='mae', callbacks=[lgb.early_stopping(stopping_rounds=50)])
    else:
        model = XGBRegressor(random_state=config.random_state, n_jobs=-1, verbosity=0, early_stopping_rounds=50, eval_metric="rmse")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    feature_names = X_train.columns

    # 1. Remove features with model.feature_importances_ <= 0
    fi_mask = model.feature_importances_ <= 0
    fi_features = set(feature_names[fi_mask])
    # Save feature_importances_ to Excel, sorted by importance
    fi_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': model.feature_importances_,
        'importance_std': [0]*len(feature_names)
    }).sort_values('importance_mean', ascending=False)
    save_importance_to_excel(fi_importance_df, label + '_fi', getattr(Config, 'permutation_importance_log_path', 'log/permutation_importance_log.xlsx'))

    # 2. Remove features with permutation_importance <= 0
    perm_result = permutation_importance(
        model, X_valid, y_valid,
        n_repeats=1 if config.debug else 10,
        random_state=config.random_state,
        scoring='neg_mean_absolute_error'
    )
    # Save permutation importance to Excel, sorted by mean importance descending
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_result.importances_mean,
        'importance_std': perm_result.importances_std
    }).sort_values('importance_mean', ascending=False)
    save_importance_to_excel(perm_importance_df, label + '_perm', getattr(Config, 'permutation_importance_log_path', 'log/permutation_importance_log.xlsx'))
    perm_mask = perm_result.importances_mean <= 0
    perm_features = set(feature_names[perm_mask])

    # 3. Remove features with SHAP importance <= 0
    explainer = shap.Explainer(model, X_valid)
    # For LGBM, disable additivity check to avoid ExplainerError
    if model_type == 'lgbm':
        shap_values = explainer(X_valid, check_additivity=False)
    else:
        shap_values = explainer(X_valid)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_mask = shap_importance <= 0
    shap_features = set(feature_names[shap_mask])
    # Save SHAP importance to Excel, sorted by importance
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': shap_importance,
        'importance_std': [0]*len(feature_names)
    }).sort_values('importance_mean', ascending=False)
    save_importance_to_excel(shap_importance_df, label + '_shap', getattr(Config, 'permutation_importance_log_path', 'log/permutation_importance_log.xlsx'))

    # Union of all features to remove
    features_to_remove = fi_features | perm_features | shap_features
    print(f"Removed {len(features_to_remove)} features for {label} using all methods (fi: {len(fi_features)}, perm: {len(perm_features)}, shap: {len(shap_features)})")

    return list(features_to_remove)

def get_least_important_features(X, y, label, model_name=None):
    # Correct unpacking of train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=config.random_state)

    if (model_name or getattr(config, 'model_name', 'xgb')) == 'xgb':
        model = XGBRegressor(random_state=config.random_state, n_jobs=-1, verbosity=0, early_stopping_rounds=50, eval_metric="mae", objective="reg:absoluteerror")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model = ExtraTreesRegressor(random_state=config.random_state, criterion='absolute_error')
        model.fit(X, y)

    # Use config.feature_importance_method to choose method
    importance_method = getattr(config, 'feature_importance_method', 'feature_importances_')
    if importance_method == 'feature_importances_':
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance_mean': model.feature_importances_,
            'importance_std': [0]*len(X_train.columns)
        })
    else:
    # model = ExtraTreesRegressor(random_state=config.random_state)
        result = permutation_importance(
            model, X_valid, y_valid,
            n_repeats=30,
            random_state=Config.random_state,
            scoring='r2'
        )
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })

    # Use model- and label-specific n_least_important_features
    if model_name is None:
        model_name_used = getattr(config, 'model_name', 'xgb')
    else:
        model_name_used = model_name
    n = config.n_least_important_features.get(model_name_used, {}).get(label, 5)
    # Sort by importance (ascending) and return n least important features
    # Remove all features with importance_mean < 0 first
    negative_importance = importance_df[importance_df['importance_mean'] <= 0]
    num_negative = len(negative_importance)
    least_important = negative_importance

    # If less than n features removed, remove more to reach n
    if num_negative < n:
        # Exclude already selected features
        remaining = importance_df[~importance_df['feature'].isin(negative_importance['feature'])]
        additional = remaining.sort_values(by='importance_mean').head(n - num_negative)
        least_important = pd.concat([least_important, additional], ignore_index=True)
    else:
        # If already removed n or more, just keep the negative ones
        least_important = negative_importance

    print(f"Removed {len(least_important)} least important features for {label} (with {num_negative} <= 0)")

    importance_df = importance_df.sort_values(by='importance_mean', ascending=True)

    # Mark features to be removed
    importance_df['removed'] = importance_df['feature'].isin(least_important['feature'])

    save_importance_to_excel(importance_df, label, Config.permutation_importance_log_path)

    return least_important['feature'].tolist()

# Save model to disk for this fold using a helper function
def save_model(Model, label, fold, model_name):
    model_path = f"models/{label}_fold{fold+1}_{model_name}"
    try:
        if 'torch' in str(type(Model)).lower():
            # Save PyTorch model state_dict
            model_path += ".pt"
            torch.save(Model.state_dict(), model_path)
        else:
            # Save scikit-learn model
            model_path += ".joblib"
            joblib.dump(Model, model_path)
        print(f"Saved model for {label} fold {fold+1} to {model_path}")
    except Exception as e:
        print(f"Failed to save model for {label} fold {fold+1}: {e}")

def train_with_other_models(model_name, label, X_train, y_train, X_val, y_val):
    """
    Train a regression model using the specified model_name, with hyperparameters
    adapted to the data size of the target label.
    """
    print(f"Training {model_name} model for label: {label}")
    if model_name == 'tabnet':
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError:
            raise ImportError("pytorch-tabnet is not installed. Please install it with 'pip install pytorch-tabnet'.")
        
        # --- Define TabNet parameters based on label ---
        if label in ['Rg', 'Tc']: # Low Data
            params = {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3, 'lambda_sparse': 1e-4}
        elif label == 'FFV': # High Data
            params = {'n_d': 24, 'n_a': 24, 'n_steps': 5, 'gamma': 1.5, 'lambda_sparse': 1e-5}
        else: # Medium Data
            params = {'n_d': 16, 'n_a': 16, 'n_steps': 5, 'gamma': 1.5, 'lambda_sparse': 1e-5}

        Model = TabNetRegressor(**params, seed=42, verbose=0)
        Model.fit(
            X_train.values, y_train.values.reshape(-1, 1),
            eval_set=[(X_val.values, y_val.values.reshape(-1, 1))],
            eval_metric=['mae'], # ACTION: Changed from 'rmse' to 'mae'
            max_epochs=200, patience=20, batch_size=1024, virtual_batch_size=128
        )

    elif model_name == 'catboost':
        # --- Define CatBoost parameters based on label ---
        params = {'iterations': 3000, 'learning_rate': 0.05, 'loss_function': 'MAE', 'eval_metric': 'MAE', 'random_seed': Config.random_state, 'verbose': False}
        if label in ['Rg', 'Tc']: # Low Data
            params.update({'depth': 5, 'l2_leaf_reg': 7})
        elif label == 'FFV': # High Data
            params.update({'depth': 7, 'l2_leaf_reg': 2})
        else: # Medium Data
            params.update({'depth': 6, 'l2_leaf_reg': 3})

        Model = CatBoostRegressor(**params)
        Model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, use_best_model=True)

    elif model_name == 'lgbm':
        # --- Define LightGBM parameters based on label ---
        params = {'n_estimators': 3000, 'learning_rate': 0.05, 'objective': 'mae', 'random_state': Config.random_state, 'verbose': -1, 'verbosity': -1}
        if label in ['Rg', 'Tc']: # Low Data
            params.update({'max_depth': 4, 'num_leaves': 20, 'reg_lambda': 5.0})
        elif label == 'FFV': # High Data
            params.update({'max_depth': 7, 'num_leaves': 40, 'reg_lambda': 1.0})
        else: # Medium Data
            params.update({'max_depth': 6, 'num_leaves': 31, 'reg_lambda': 1.0})

        Model = LGBMRegressor(**params)
        Model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', callbacks=[lgb.early_stopping(stopping_rounds=50)])

    elif model_name == 'extratrees':
        # --- Define ExtraTrees parameters based on label ---
        params = {'n_estimators': 300, 'criterion': 'absolute_error', 'random_state': Config.random_state, 'n_jobs': -1}
        if label in ['Rg', 'Tc']: # Low Data: Prevent overfitting by requiring more samples per leaf
            params.update({'min_samples_leaf': 3, 'max_features': 0.8})
        else: # High/Medium Data
            params.update({'min_samples_leaf': 1, 'max_features': 1.0})
            
        Model = ExtraTreesRegressor(**params)
        X_all, y_all = combine_train_val(X_train, X_val, y_train, y_val)
        Model.fit(X_all, y_all)

    elif model_name == 'randomforest':
        # --- Define RandomForest parameters based on label ---
        params = {'n_estimators': 1000, 'criterion': 'absolute_error', 'random_state': Config.random_state, 'n_jobs': -1}
        if label in ['Rg', 'Tc']: # Low Data
            params.update({'min_samples_leaf': 3, 'max_features': 0.8, 'max_depth': 15})
        else: # High/Medium Data
            params.update({'min_samples_leaf': 1, 'max_features': 1.0, 'max_depth': None})

        Model = RandomForestRegressor(**params)
        X_all, y_all = combine_train_val(X_train, X_val, y_train, y_val)
        Model.fit(X_all, y_all)

    elif model_name == 'hgbm':
        from sklearn.ensemble import HistGradientBoostingRegressor
        # --- Define HGBM parameters based on label ---
        params = {'max_iter': 1000, 'learning_rate': 0.05, 'loss': 'absolute_error', 'early_stopping': True, 'random_state': 42} # ACTION: Changed loss to 'absolute_error'
        if label in ['Rg', 'Tc']: # Low Data
            params.update({'max_depth': 4, 'l2_regularization': 1.0})
        elif label == 'FFV': # High Data
            params.update({'max_depth': 7, 'l2_regularization': 0.1})
        else: # Medium Data
            params.update({'max_depth': 6, 'l2_regularization': 0.5})

        Model = HistGradientBoostingRegressor(**params)
        X_all, y_all = combine_train_val(X_train, X_val, y_train, y_val)
        Model.fit(X_all, y_all)

    elif model_name == 'nn':
        # The 'train_with_nn' function already uses different configs per label, which is excellent!
        # Just ensure the loss function inside it is nn.L1Loss()
        Model = train_with_nn(label, X_train, X_val, y_train, y_val)
    else:
        raise ValueError(f"Unknown or unavailable model: {model_name}")
    
    return Model

def train_with_autogluon(label, X_train, y_train):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError("AutoGluon is not installed. Please install it with 'pip install autogluon'.")
    import pandas as pd
    import uuid
    # Prepare data for AutoGluon (must be DataFrame with column names)
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train, name=label)
    train_data = X_train_df.copy()
    train_data[label] = y_train_series.values

    unique_path = f"autogluon_{label}_{int(time.time())}_{uuid.uuid4().hex}"

    hyperparameters = {
        "GBM": {},
        "CAT": {},
        "XGB": {},
        "NN_TORCH": {},
        "RF": {},
        "XT": {}
    }

    hyperparameter_tune_kwargs = {
        "num_trials": 50,
        "scheduler": "local",
        "searcher": "auto"
    }

    time_limit = 300 if getattr(Config, 'debug', False) else 3600

    predictor = TabularPredictor(
        label=label,
        eval_metric="mae",  # Use 'mae' for regression
        path=unique_path
    ).fit(
        train_data,
        presets="best_quality",
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        num_bag_folds=5,
        num_stack_levels=2,
        time_limit=time_limit
    )

    print("\n[AutoGluon] Leaderboard:")
    leaderboard = predictor.leaderboard(silent=False)

    print("\n[AutoGluon] Model Info:")
    print(predictor.info())

    print("\n[AutoGluon] Model Names:")
    model_names = leaderboard["model"].tolist()   # <- FIXED here
    print(model_names)

    # Save feature importance to CSV
    fi_df = predictor.feature_importance(train_data)
    fi_path = f"NeurIPS/autogluon_feature_importance_{label}.csv"  
    fi_df.to_csv(fi_path)
    print(f"[AutoGluon] Feature importance saved to {fi_path}")

    return predictor

def train_with_stacking(label, X_train, y_train):
    """
    Trains XGBoost, ExtraTrees, and CatBoost using sklearn's StackingRegressor.
    Returns the fitted stacking model and base models.
    """
    estimators = [
    ('xgb', XGBRegressor(n_estimators=10000, learning_rate=0.01, max_depth=5, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_lambda=1.0, objective="reg:absoluteerror", random_state=Config.random_state, n_jobs=-1)),
    ('lgbm', LGBMRegressor(n_estimators=10000, learning_rate=0.01, num_leaves=64, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, reg_lambda=1.0, max_depth=-1, objective="mae", random_state=Config.random_state, n_jobs=-1, verbose=-1)),
        ('cb', CatBoostRegressor(iterations=10000, learning_rate=0.01, depth=7, l2_leaf_reg=5, bagging_temperature=0.8, random_seed=Config.random_state, verbose=0)),
    ('rf', RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=3, random_state=Config.random_state, n_jobs=-1, criterion='absolute_error'))
    ]

    # Candidate final estimators
    final_estimators = {
        "ElasticNet": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=100, max_iter=50000, tol=1e-3, cv=5, n_jobs=-1),
        "CatBoost": CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3, random_seed=Config.random_state, verbose=0),
    "LightGBM": LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=64, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, reg_lambda=1.0, max_depth=-1, objective="mae", random_state=Config.random_state, n_jobs=-1, verbose=-1),
    "XGBoost": XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, gamma=0.1, objective="reg:absoluteerror", random_state=Config.random_state, n_jobs=-1)
    }

    final_estimator = final_estimators['ElasticNet']
    
    stacker = StackingRegressor(estimators=estimators, final_estimator=final_estimator, passthrough=True, cv=5, n_jobs=-1)

    stacker.fit(X_train, y_train)
    return stacker

def train_with_xgb(label, X_train, y_train, X_val, y_val):
    print(f"Training XGB model for label: {label}")
    if label=="Tg": # Medium Data (~1.2k samples)
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.01, max_depth=5, # <-- Reduced from 6
            colsample_bytree=1.0, reg_lambda=7.0, gamma=0.1, subsample=0.5, # <-- Increased lambda, reduced subsample
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
    elif label=='Rg': # Very Low Data (~600 samples), HIGHEST PRIORITY
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.06, max_depth=4, 
            colsample_bytree=1.0, reg_lambda=10.0, gamma=0.1, subsample=0.6, 
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
    elif label=='FFV': # High Data (~8k samples), LOWEST PRIORITY
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.06, max_depth=7, 
            colsample_bytree=0.8, reg_lambda=2.0, gamma=0.0, subsample=0.6, 
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
    elif label=='Tc': # Low Data (~900 samples), HIGH PRIORITY
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.01, max_depth=4, 
            colsample_bytree=0.8, reg_lambda=7.0, gamma=0.0, subsample=0.6, 
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
    elif label=='Density': # Medium Data (~1.2k samples)
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.06, max_depth=5, 
            colsample_bytree=1.0, reg_lambda=3.0, gamma=0.0, subsample=0.8, 
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
        
    print(f"Model {label} trained with shape: {X_train.shape}, {y_train.shape}")

    Model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return Model

def preprocess_numerical_features(X, label=None):
    # Ensure numeric types
    X_num = X.select_dtypes(include=[np.number]).copy()
    
    # Replace inf/-inf with NaN
    X_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    valid_cols = X_num.columns
    # # Drop columns with any NaN
    # valid_cols = [col for col in X_num.columns if not X_num[col].isnull().any()]
    
    # dropped_cols = set(X_num.columns) - set(valid_cols)
    # if dropped_cols:
    #     print(f"Dropped columns with NaN/Inf for {label}: {list(dropped_cols)}")

    # # Keep only valid columns
    # X_num = X_num[valid_cols]
    
    # Calculate median for each column (for use in test set)
    median_values = X_num.median()
    # Scale features if enabled
    if getattr(Config, 'use_standard_scaler', False):
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        X_num = pd.DataFrame(X_num_scaled, columns=valid_cols, index=X.index)
    else:
        scaler = None
        X_num = X_num.copy()
    
    # Display categorical features
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        print(f"Categorical (non-numeric) features for {label}: {cat_cols}")
    else:
        print(f"No categorical (non-numeric) features for {label}.")
    return X_num, valid_cols, scaler, median_values

def select_features_with_lasso(X, y, label):
    """
    Performs feature selection using Lasso (L1) regularization.

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series or np.array): The target values.
        label (str): The name of the target property (e.g., 'Rg', 'Tc').

    Returns:
        pd.DataFrame: A DataFrame containing only the selected features.
    """
    # Lasso is sensitive to feature scaling, so we scale the data first.
    # We also need to handle any potential NaN values before scaling.
    X_filled = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # Use LassoCV to automatically find the best alpha (regularization strength)
    # through cross-validation. This is more robust than picking a single alpha.
    lasso_cv = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000)

    # Use SelectFromModel to wrap the LassoCV regressor.
    # This will select features where the Lasso coefficient is non-zero.
    # The 'threshold="median"' can be a good strategy to select the top 50% of features
    # if LassoCV is too lenient and keeps too many. Start with the default (None).
    feature_selector = SelectFromModel(lasso_cv, prefit=False, threshold=None)

    print(f"[{label}] Fitting LassoCV to find optimal features...")
    feature_selector.fit(X_scaled, y)

    # Get the names of the features that were kept
    selected_feature_names = X.columns[feature_selector.get_support()]

    print(f"[{label}] Original number of features: {X.shape[1]}")
    print(f"[{label}] Features selected by Lasso: {len(selected_feature_names)}")

    # Return the original DataFrame with only the selected columns
    return selected_feature_names


def check_inf_nan(X, y, label=None):
    """
    Checks for inf, -inf, and NaN values in X (DataFrame) and y (array/Series).
    Prints summary and returns True if any such values are found.
    """
    X_inf = np.isinf(X.values).any()
    X_nan = np.isnan(X.values).any()
    y_inf = np.isinf(y).any()
    y_nan = np.isnan(y).any()
    if label is None:
        label = ""
    else:
        label = f" [{label}]"
    if X_inf or X_nan or y_inf or y_nan:
        print(f"⚠️ Detected inf/nan in X or y{label}: X_inf={X_inf}, X_nan={X_nan}, y_inf={y_inf}, y_nan={y_nan}")
        if X_inf:
            print(f"  X columns with inf: {X.columns[np.isinf(X.values).any(axis=0)].tolist()}")
        if X_nan:
            print(f"  X columns with nan: {X.columns[np.isnan(X.values).any(axis=0)].tolist()}")
        if y_inf:
            print("  y contains inf values.")
        if y_nan:
            print("  y contains nan values.")
        return True
    else:
        print(f"No inf/nan in X or y{label}.")
        return False

# Utility: Display model summary if torchinfo is available
def show_model_summary(model, input_dim, batch_size=32):
    try:
        from torchinfo import summary
        print(summary(model, input_size=(batch_size, input_dim)))
    except ImportError:
        print("torchinfo is not installed. Install it with 'pip install torchinfo' to see model summaries.")


def train_model(
    model,
    X_train, X_val, y_train, y_val,
    epochs=3000, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=30, verbose=True
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=verbose)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    use_early_stopping = X_val is not None and y_val is not None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        if verbose and (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        if use_early_stopping:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_loss = criterion(val_preds, y_val_tensor).item()
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    return model

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim, neurons, dropouts):
        super().__init__()
        layers = []
        for i, n in enumerate(neurons):
            layers.append(nn.Linear(input_dim, n))
            # layers.append(nn.BatchNorm1d(n))
            layers.append(nn.ReLU())
            if i < len(dropouts) and dropouts[i] > 0:
                layers.append(nn.Dropout(dropouts[i]))
            input_dim = n
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_with_nn(label, X_train, X_val, y_train, y_val):

    input_dim = X_train.shape[1]
    
    if getattr(Config, "search_nn", False):
        print(f"--- Starting targeted NN architecture search for label: {label} ---")
        
        search_configs_by_label = {
            'Low': [ # For Rg, Tc. Simple models with high regularization.
                # --- Single Layer Focus ---
                {"neurons": [32], "dropouts": [0.3]},
                {"neurons": [64], "dropouts": [0.4]},
                {"neurons": [128], "dropouts": [0.5]},
                {"neurons": [256], "dropouts": [0.5]},

                # --- Two Layer Rectangular Focus (based on Rg's winner) ---
                {"neurons": [64, 64], "dropouts": [0.4, 0.4]},
                {"neurons": [96, 96], "dropouts": [0.5, 0.5]},
                {"neurons": [128, 128], "dropouts": [0.5, 0.5]}, # Previous winner
                {"neurons": [192, 192], "dropouts": [0.5, 0.5]},
                {"neurons": [256, 256], "dropouts": [0.5, 0.5]},

                # --- Two Layer Tapering Focus ---
                {"neurons": [128, 32], "dropouts": [0.5, 0.3]},
                {"neurons": [128, 64], "dropouts": [0.5, 0.4]},
                {"neurons": [256, 64], "dropouts": [0.5, 0.4]},
                {"neurons": [256, 128], "dropouts": [0.5, 0.4]},
                {"neurons": [512, 128], "dropouts": [0.5, 0.4]},

                # --- Three Layer Focus ---
                {"neurons": [64, 64, 64], "dropouts": [0.4, 0.4, 0.4]},
                {"neurons": [128, 128, 128], "dropouts": [0.5, 0.5, 0.5]},
                {"neurons": [128, 64, 32], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [256, 128, 64], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [256, 64, 32], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [512, 128, 32], "dropouts": [0.5, 0.4, 0.3]},
            ],

            'Medium': [ # For Tg, Density. Balanced complexity.
                # --- Two Layer Focus ---
                {"neurons": [256, 64], "dropouts": [0.4, 0.3]},
                {"neurons": [256, 128], "dropouts": [0.4, 0.3]},
                {"neurons": [512, 64], "dropouts": [0.5, 0.3]},
                {"neurons": [512, 128], "dropouts": [0.5, 0.4]}, # Previous winner
                {"neurons": [512, 256], "dropouts": [0.5, 0.4]},
                {"neurons": [1024, 128], "dropouts": [0.5, 0.4]},
                {"neurons": [1024, 256], "dropouts": [0.5, 0.4]},
                {"neurons": [256, 256], "dropouts": [0.4, 0.4]},
                {"neurons": [512, 512], "dropouts": [0.5, 0.5]},

                # --- Three Layer Focus ---
                {"neurons": [256, 128, 64], "dropouts": [0.4, 0.3, 0.2]},
                {"neurons": [512, 128, 64], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [512, 256, 64], "dropouts": [0.5, 0.4, 0.2]},
                {"neurons": [512, 256, 128], "dropouts": [0.5, 0.4, 0.3]}, # Previous winner
                {"neurons": [1024, 256, 64], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [1024, 512, 128], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [1024, 512, 256], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [256, 256, 256], "dropouts": [0.4, 0.4, 0.4]},
                {"neurons": [512, 512, 512], "dropouts": [0.5, 0.5, 0.5]},

                # --- Four Layer Focus ---
                {"neurons": [512, 256, 128, 64], "dropouts": [0.5, 0.4, 0.3, 0.2]},
                {"neurons": [1024, 512, 256, 128], "dropouts": [0.5, 0.4, 0.3, 0.2]},
            ],

            'High': [ # For FFV. Exploring width and depth.
                # --- Refining Around Winner ([512, 256]) ---
                {"neurons": [512, 128], "dropouts": [0.3, 0.2]},
                {"neurons": [512, 256], "dropouts": [0.3, 0.2]}, # Previous winner
                {"neurons": [512, 512], "dropouts": [0.3, 0.3]},
                {"neurons": [1024, 256], "dropouts": [0.4, 0.3]},
                {"neurons": [1024, 512], "dropouts": [0.4, 0.3]},
                {"neurons": [1024, 1024], "dropouts": [0.4, 0.4]},
                {"neurons": [2048, 512], "dropouts": [0.5, 0.4]},
                {"neurons": [2048, 1024], "dropouts": [0.5, 0.4]},

                # --- Three Layer Focus ---
                {"neurons": [512, 256, 128], "dropouts": [0.3, 0.2, 0.2]},
                {"neurons": [1024, 256, 64], "dropouts": [0.4, 0.3, 0.2]},
                {"neurons": [1024, 512, 128], "dropouts": [0.4, 0.3, 0.2]},
                {"neurons": [1024, 512, 256], "dropouts": [0.4, 0.3, 0.2]},
                {"neurons": [2048, 512, 128], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [2048, 1024, 512], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [512, 512, 512], "dropouts": [0.3, 0.3, 0.3]},
                {"neurons": [1024, 1024, 1024], "dropouts": [0.4, 0.4, 0.4]},

                # --- Four+ Layer Focus ---
                {"neurons": [512, 256, 256, 128], "dropouts": [0.3, 0.2, 0.2, 0.1]},
                {"neurons": [1024, 512, 256, 128], "dropouts": [0.4, 0.3, 0.2, 0.2]},
                {"neurons": [1024, 512, 512, 256], "dropouts": [0.4, 0.3, 0.3, 0.2]},
                {"neurons": [512, 512, 512, 512], "dropouts": [0.3, 0.3, 0.3, 0.3]},
            ]
        }
        
        # Determine which set of configs to use
        if label in ['Rg', 'Tc']:
            configs_to_search = search_configs_by_label['Low']
            print("Using search space for LOW data targets.")
        elif label == 'FFV':
            configs_to_search = search_configs_by_label['High']
            print("Using search space for HIGH data targets.")
        else: # Tg, Density
            configs_to_search = search_configs_by_label['Medium']
            print("Using search space for MEDIUM data targets.")

        results = []
        for i, cfg in enumerate(configs_to_search):
            print(f"\n---> Searching config {i+1}/{len(configs_to_search)}: Neurons={cfg['neurons']}, Dropouts={cfg['dropouts']}")
            model = FeedforwardNet(input_dim, cfg["neurons"], cfg["dropouts"])
            model = train_model(model, X_train, X_val, y_train, y_val, verbose=False) # Turn off verbose for cleaner search logs
            
            model.eval()
            with torch.no_grad():
                X_val_np = np.asarray(X_val) if isinstance(X_val, pd.DataFrame) else X_val
                device = next(model.parameters()).device
                X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)
                y_pred = model(X_val_tensor).cpu().numpy().flatten()
            
            val_mae = mean_absolute_error(y_val, y_pred)
            print(f"     Resulting Val MAE: {val_mae:.6f}")
            results.append({"neurons": cfg["neurons"], "dropouts": cfg["dropouts"], "val_mae": val_mae})
        
        df = pd.DataFrame(results)
        print("\n--- Neural Network Search Results ---")
        print(df.sort_values(by='val_mae').to_string(index=False))
        
        df.to_csv(f"nn_config_validation_mae_{label}.csv", index=False)
        print(f"\nSaved results to nn_config_validation_mae_{label}.csv")
        
        best_row = df.loc[df['val_mae'].idxmin()]
        print(f"\nBest config: neurons={best_row['neurons']}, dropouts={best_row['dropouts']}, Validation MAE: {best_row['val_mae']:.6f}")
        
        config = best_row.to_dict()
        print("Re-training best model on the full training data...")
    
    else: # If not searching, use a single pre-defined configuration
        best_configs = {
            "Tg":      {"neurons": [256, 128, 64], "dropouts": [0.4, 0.3, 0.2]},
            "Density": {"neurons": [256, 128, 64], "dropouts": [0.4, 0.3, 0.2]},
            "FFV":     {"neurons": [512, 256, 128], "dropouts": [0.3, 0.2, 0.2]},
            "Tc":      {"neurons": [128, 64], "dropouts": [0.4, 0.3]},
            "Rg":      {"neurons": [128, 64], "dropouts": [0.4, 0.3]},
        }
        config = best_configs.get(label)
        print(f"Using pre-defined best config for {label}: Neurons={config['neurons']}, Dropouts={config['dropouts']}")

    # Final model training
    best_model = FeedforwardNet(input_dim, config["neurons"], config["dropouts"])
    show_model_summary(best_model, input_dim)
    best_model = train_model(best_model, X_train, X_val, y_train, y_val, verbose=True)
    return best_model

# Utility to set random state everywhere
def set_global_random_seed(seed, config=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if config is not None:
        config.random_state = seed

import hashlib

def stable_hash(obj, max_value=1_000_000):
    """
    Deterministic hash for objects (e.g. labels).
    Always returns the same value across runs/machines.
    """
    # Convert to string and encode
    s = str(obj).encode("utf-8")
    # Use MD5 (fast & deterministic)
    h = hashlib.md5(s).hexdigest()
    # Convert hex digest to int and limit range
    return int(h, 16) % max_value

def train_and_evaluate_models(label, X_main, y_main, splits, nfold, Config):
    """
    Trains models for the given label using the specified configuration.
    Returns: models, fold_maes, mean_fold_mae, std_fold_mae
    """
    # Use a prime multiplier for folds
    FOLD_PRIME = 9973   # a large prime

    models = []
    fold_maes = []
    mean_fold_mae = None
    std_fold_mae = None

    # Use stacking only if enabled in config
    if getattr(Config, 'use_stacking', False):
        Model = train_with_stacking(label, X_main, y_main)
        models.append(Model)
        save_model(Model, label, 1, Config.model_name)
    elif Config.model_name in ['autogluon']:
        Model = train_with_autogluon(label, X_main, y_main)
        models.append(Model)
        # save_model(Model, label, 1, Config.model_name)
    else:
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n--- Fold {fold+1}/{nfold} ---")
            # Set a different random seed for each fold for best possible result
            # Use a deterministic but varied seed: base + fold + hash(label)
            base_seed = getattr(Config, 'random_state', 42)
            label_hash = stable_hash(label)   # replaces abs(hash(label)) % 10000
            fold_seed = base_seed + fold * FOLD_PRIME  + label_hash

            set_global_random_seed(fold_seed, config=Config)

            # Robustly handle both DataFrame and ndarray
            if isinstance(X_main, np.ndarray):
                X_train, X_val = X_main[train_idx], X_main[val_idx]
            else:
                X_train, X_val = X_main.iloc[train_idx], X_main.iloc[val_idx]
            if isinstance(y_main, np.ndarray):
                y_train, y_val = y_main[train_idx], y_main[val_idx]
            else:
                y_train, y_val = y_main.iloc[train_idx], y_main.iloc[val_idx]

            if Config.model_name == 'xgb':
                Model = train_with_xgb(label, X_train, y_train, X_val, y_val)
            elif Config.model_name in ['catboost', 'lgbm', 'extratrees', 'randomforest', 'balancedrf', 'tabnet', 'hgbm', 'autogluon', 'nn']:
                Model = train_with_other_models(Config.model_name, label, X_train, y_train, X_val, y_val)
            else:
                assert False, "No model present. Set Config.use_train_with_xgb = True to train a model."

            # Save model for later holdout prediction
            models.append(Model)
            save_model(Model, label, fold, Config.model_name)

            # Predict on validation set for this fold
            if hasattr(Model, 'forward') and not hasattr(Model, 'predict'):
                Model.eval()
                X_val_np = np.asarray(X_val) if isinstance(X_val, pd.DataFrame) else X_val
                device = next(Model.parameters()).device
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)
                    y_val_pred = Model(X_val_tensor).cpu().numpy().flatten()
            else:
                y_val_pred = Model.predict(X_val)

            fold_mae = mean_absolute_error(y_val, y_val_pred)
            print(f"Fold {fold+1} MAE (on validation set): {fold_mae}")
            fold_maes.append(fold_mae)
            # Save y_val, y_val_pred, and residuals in sorted order for each fold
            residuals = y_val - y_val_pred
            results_df = pd.DataFrame({
                'y_val': y_val,
                'y_val_pred': y_val_pred,
                'residual': residuals
            })
            results_df = results_df.sort_values(by='residual', ascending=False).reset_index(drop=True)
            os.makedirs(f'NeurIPS/fold_residuals/{label}', exist_ok=True)
            results_df.to_csv(f'NeurIPS/fold_residuals/{label}/fold_{fold+1}_val_pred_residuals.csv', index=False)
        mean_fold_mae = np.mean(fold_maes)
        std_fold_mae = np.std(fold_maes)
        print(f"{label} 5-Fold CV mean_absolute_error (on validation sets): {mean_fold_mae} ± {std_fold_mae}")

    return models, fold_maes, mean_fold_mae, std_fold_mae

def save_feature_selection_info(label, kept_columns, least_important_features, correlated_features_dropped, scaler, X_holdout, y_holdout, median_values):
    holdout_dir = f"NeurIPS/feature_selection/{label}"
    os.makedirs(holdout_dir, exist_ok=True)
    feature_info = {
        "kept_columns": list(kept_columns),
        "least_important_features": list(least_important_features),
        "correlated_features_dropped": list(correlated_features_dropped),
    }
    # Save median_values if provided
    if median_values is not None:
        if hasattr(median_values, 'to_dict'):
            feature_info["median_values"] = median_values.to_dict()
        else:
            feature_info["median_values"] = median_values

    # Save X_holdout and y_holdout for this label
    X_holdout_path = os.path.join(holdout_dir, "X_holdout.csv")
    y_holdout_path = os.path.join(holdout_dir, "y_holdout.csv")
    pd.DataFrame(X_holdout).to_csv(X_holdout_path, index=False)
    pd.DataFrame({"y_holdout": y_holdout}).to_csv(y_holdout_path, index=False)

    feature_info_path = os.path.join(holdout_dir, f"{label}_feature_info.json")
    with open(feature_info_path, "w") as f:
        json.dump(feature_info, f, indent=2)

    # Save scaler object
    scaler_path = os.path.join(holdout_dir, "scaler.joblib")
    if scaler is not None:
        joblib.dump(scaler, scaler_path)

def load_feature_selection_info(label, base_dir):
    """
    Loads feature selection info saved by save_feature_selection_info for a given label.
    Returns a dict with keys: kept_columns, least_important_features, correlated_features_dropped, scaler, X_holdout, y_holdout.
    """

    holdout_dir = os.path.join(base_dir, f"NeurIPS/feature_selection/{label}")
    feature_info_path = os.path.join(holdout_dir, f"{label}_feature_info.json")
    X_holdout_path = os.path.join(holdout_dir, "X_holdout.csv")
    y_holdout_path = os.path.join(holdout_dir, "y_holdout.csv")

    if not os.path.exists(feature_info_path):
        raise FileNotFoundError(f"Feature info file not found: {feature_info_path}")

    with open(feature_info_path, "r") as f:
        feature_info = json.load(f)

    X_holdout = pd.read_csv(X_holdout_path)
    y_holdout = pd.read_csv(y_holdout_path)["y_holdout"].values

    # Note: scaler is not restored as an object (only its params or type string is saved)
    # If you need the actual scaler object, you must save it with joblib or pickle

    # Load scaler object
    scaler_path = os.path.join(holdout_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

    # Try to load median_values if present in feature_info, else set to empty Series
    if "median_values" in feature_info:
        median_values = pd.Series(feature_info["median_values"])
    else:
        median_values = pd.Series(dtype=float)
    return {
        "kept_columns": feature_info.get("kept_columns", []),
        "least_important_features": feature_info.get("least_important_features", []),
        "correlated_features_dropped": feature_info.get("correlated_features_dropped", []),
        "X_holdout": X_holdout,
        "y_holdout": y_holdout,
        "scaler": scaler,
        "median_values": median_values
    }

def load_models_for_label(label, models_dir="models"):
    """
    Loads all models for a given label from the specified directory.
    Model filenames must start with the label (e.g., 'Tg_fold1_xgb.joblib').
    Returns a list of loaded models.
    """

    models = []
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' does not exist.")
        return models

    # Match both .joblib and .pt (for torch) files
    pattern_joblib = os.path.join(models_dir, f"{label}_*.joblib")
    pattern_pt = os.path.join(models_dir, f"{label}_*.pt")
    model_files = glob.glob(pattern_joblib) + glob.glob(pattern_pt)
    if not model_files:
        print(f"No models found for label '{label}' in '{models_dir}'.")
        return models

    for model_file in sorted(model_files):
        if model_file.endswith(".joblib"):
            try:
                model = joblib.load(model_file)
                models.append(model)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
        elif model_file.endswith(".pt"):
            # Torch model loading requires model class and architecture
            print(f"Skipping torch model {model_file} (requires model class definition).")
            # You can implement torch loading here if needed
    print(f"Loaded {len(models)} models for label '{label}'.")
    return models

output_df = pd.DataFrame({
    'id': test_ids
})

# --- Store and display mean_absolute_error for each label ---
mae_results = []

def prepare_label_data(label, subtables, Config):
    print(f"Processing label: {label}")
    print(subtables[label].head())
    print(subtables[label].shape)
    original_smiles = subtables[label]['SMILES'].tolist()
    original_labels = subtables[label][label].values

    # Canonicalize SMILES and deduplicate at molecule level before augmentation
    canonical_smiles = [get_canonical_smiles(s) for s in original_smiles]
    smiles_label_df = pd.DataFrame({
        'SMILES': canonical_smiles,
        'label': original_labels
    })
    before_dedup = len(smiles_label_df)
    smiles_label_df = smiles_label_df.drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
    after_dedup = len(smiles_label_df)
    num_dropped = before_dedup - after_dedup
    print(f"Dropped {num_dropped} duplicate SMILES rows for {label} before augmentation.")
    original_smiles = smiles_label_df['SMILES'].tolist()
    original_labels = smiles_label_df['label'].values

    if Config.use_augmentation and not Config.debug:
        print(f"SMILES before augmentation: {len(original_smiles)}")
        smiles_aug, labels_aug = augment_smiles_dataset(original_smiles, original_labels, num_augments=1)
        print(f"SMILES after augmentation: {len(smiles_aug)} (increase: {len(smiles_aug) - len(original_smiles)})")
        original_smiles, original_labels = smiles_aug, labels_aug

    # After augmentation, deduplicate again at molecule level (canonical SMILES)
    canonical_smiles_aug = [get_canonical_smiles(s) for s in original_smiles]
    smiles_label_aug_df = pd.DataFrame({
        'SMILES': canonical_smiles_aug,
        'label': original_labels
    })
    smiles_label_aug_df = smiles_label_aug_df.drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
    original_smiles = smiles_label_aug_df['SMILES'].tolist()
    original_labels = smiles_label_aug_df['label'].values

    fp_df, descriptor_df, valid_smiles, invalid_indices = smiles_to_combined_fingerprints_with_descriptors(original_smiles)

    print(f"Invalid indices for {label}: {invalid_indices}")
    y = np.delete(original_labels, invalid_indices)
    print(fp_df.shape)
    fp_df.reset_index(drop=True, inplace=True)

    if not descriptor_df.empty:
        X = pd.DataFrame(descriptor_df)
        X, kept_columns, scaler, median_values = preprocess_numerical_features(X, label)
        X.reset_index(drop=True, inplace=True)
        if not fp_df.empty:
            X = pd.concat([X, fp_df], axis=1)
    else:
        kept_columns = []
        scaler = None
        X = fp_df

    # Remove duplicate rows in X and corresponding values in y (feature-level duplicates)
    X_dup = X.duplicated(keep='first')
    if X_dup.any():
        print(f"Found {X_dup.sum()} duplicate rows in X for {label}, removing them.")
        X = X[~X_dup]
        y = y[~X_dup]
    print(f"After concat: {X.shape}")
    # Fill NaN in train with median from train
    # Only fill NaN with median if using neural network
    if Config.model_name == 'nn':
        X = X.fillna(median_values)
    check_inf_nan(X, y, label)

    # display_outlier_summary(y, X=X, name=label, z_thresh=3, iqr_factor=1.5, iso_contamination=0.01, lof_contamination=0.01)

    # Drop least important features from X and test

    least_important_features = []
    if getattr(Config, 'use_least_important_features_all_methods', False):
        for i in range(4):
            print(f"Iteration {i+1} for least important feature removal on {label}")
            least_important_feature = get_least_important_features_all_methods(X, y, label, model_name=Config.model_name)
            least_important_features.extend(least_important_feature)
            if len(least_important_feature) > 0:
                print(f"label: {label} Dropping least important features: {least_important_feature}")
                X = X.drop(columns=least_important_feature)
                print(f"After dropping least important features: {X.shape}")

    check_inf_nan(X, y, label)
    # Drop highly correlated features using label-specific correlation threshold from Config
    correlation_threshold = Config.correlation_thresholds.get(label, 1.0)
    if correlation_threshold < 1.0:
        X, correlated_features_dropped = drop_correlated_features(pd.DataFrame(X), threshold=correlation_threshold)
    else:
        correlated_features_dropped = []

    print(f"After correlation cut (threshold={correlation_threshold}): {X.shape}, dropped columns: {correlated_features_dropped}")
    print(f"After dropping correlated features: {X.shape}")
    check_inf_nan(X, y, label)

    if getattr(Config, 'use_variance_threshold', False):
        threshold = 0.01
        selector = VarianceThreshold(threshold=threshold)
        X_sel = selector.fit_transform(X)
        # Get mask of selected features
        selected_cols_variance = X.columns[selector.get_support()]
        # Convert back to DataFrame with column names
        X = pd.DataFrame(X_sel, columns=selected_cols_variance, index=X.index)
        print(f"After variance cut: {X.shape}")
        print(f'Type of X: {type(X)}')

    if Config.add_gaussian and not Config.debug:
        n_samples = 1000
        X, y = augment_dataset(X, y, n_samples=n_samples)
        print(f"After augment cut: {X.shape}")

    # --- Hold out 10% for final MAE calculation ---
    # X_main, X_holdout, y_main, y_holdout = train_test_split(X, y, test_size=0.1, random_state=Config.random_state)
    # Bin y for stratification
    y_bins = pd.qcut(y, q=5, duplicates='drop', labels=False)
    X_main, X_holdout, y_main, y_holdout = train_test_split(X, y, test_size=0.10, random_state=Config.random_state, stratify=y_bins)
    if Config.useAllDataForTraining == True:
        X_main = X
        y_main = y
    # --- Optionally apply PCA ---
    if getattr(Config, 'use_pca', False):
        X_main, X_holdout, pca = apply_pca(X_main, X_holdout, verbose=True)
    else:
        pca = None

    # --- Cross-Validation or Single Split (for speed) ---
    fold_maes = []
    test_preds = []
    val_preds = np.zeros(len(y_main))
    if getattr(Config, 'use_cross_validation', True):
        nfold = 10
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=Config.random_state)
        # For regression, bin y_main for stratification
        y_bins = pd.qcut(y_main, q=nfold, duplicates='drop', labels=False)
        splits = skf.split(X_main, y_bins)
    else:
        # Use a single split: 80% train, 20% val
        train_idx, val_idx = train_test_split(
            np.arange(len(X_main)), test_size=0.2, random_state=Config.random_state
        )
        splits = [(train_idx, val_idx)]
        nfold = 1

    return {
        "X_main": X_main,
        "X_holdout": X_holdout,
        "y_main": y_main,
        "y_holdout": y_holdout,
        "kept_columns": kept_columns,
        "scaler": scaler,
        "median_values": median_values,
        "least_important_features": least_important_features,
        "correlated_features_dropped": correlated_features_dropped,
        "selector": selector if getattr(Config, 'use_variance_threshold', False) else None,
        "selected_cols_variance": selected_cols_variance if getattr(Config, 'use_variance_threshold', False) else None,
        "pca": pca,
        "splits": splits,
        "nfold": nfold,
        "fold_maes": fold_maes,
        "test_preds": test_preds,
        "val_preds": val_preds
    }

def load_label_data(label, model_dir=None):
    if model_dir is not None:
        # Load model and data for the specified label
        model_path = os.path.join(model_dir, f"{label}_model.pkl")
        data_path = os.path.join(model_dir, f"{label}_data.pkl")
        model = joblib.load(model_path)
        data = joblib.load(data_path)
        return model, data
    return None, None

def train_or_predict(train_model=True, model_dir=None):
    for label in labels:
        if train_model:
            print(f"\n=== Training/Predicting for label: {label} ===")
            label_data = prepare_label_data(label, subtables, config)
            X_main = label_data["X_main"]
            X_holdout = label_data["X_holdout"]
            y_main = label_data["y_main"]
            y_holdout = label_data["y_holdout"]
            kept_columns = label_data["kept_columns"]
            scaler = label_data["scaler"]
            median_values = label_data["median_values"]
            least_important_features = label_data["least_important_features"]
            correlated_features_dropped = label_data["correlated_features_dropped"]
            selector = label_data["selector"]
            selected_cols_variance = label_data["selected_cols_variance"]
            pca = label_data["pca"]
            splits = label_data["splits"]
            nfold = label_data["nfold"]
            fold_maes = label_data["fold_maes"]
            test_preds = label_data["test_preds"]
            val_preds = label_data["val_preds"]

            # --- Save feature selection info for this label ---
            save_feature_selection_info(label, kept_columns, least_important_features, correlated_features_dropped, scaler, X_holdout, y_holdout, median_values)  
            os.makedirs('models', exist_ok=True)
            models = []

            # labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

            # --- Hyperparameter tuning for this label ---
            if Config.enable_param_tuning:
                if label == 'Tg':
                    db_path = f"xgb_tuning_{label}.db"
                    init_xgb_tuning_db(db_path)
                    # Example param_grid (customize as needed)
                    param_grid = {
                        'n_estimators': [3000],
                        'max_depth': [4, 5, 6, 7],
                        'learning_rate': [0.001, 0.01, 0.06, 0.1],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0],
                        'gamma': [0, 0.1],
                        'reg_lambda': [1.0, 5.0, 10.0],
                        'early_stopping_rounds': [50],
                        'objective': ["reg:squarederror"],
                        'eval_metric': ["rmse"]
                    }
                    # You must define X_train and y_train for tuning here
                    xgb_grid_search_with_db(X_main, y_main, param_grid, db_path=db_path)
                else:
                    continue
            # FIXME save features_keep
            # Ensure only features present in X_main are kept

            # if label in ['Tg', 'Tc', 'Density', 'Rg']:
            #     features_keep = select_features_with_lasso(X_main, y_main, label)
            #     X_main = X_main[features_keep]
            models, fold_maes, mean_fold_mae, std_fold_mae = train_and_evaluate_models(label, X_main, y_main, splits, nfold, Config)
        else:
            print(f"\n=== Loading models and data for label: {label} ===")
            mean_fold_mae, std_fold_mae = None, None
            feature_info = load_feature_selection_info(label, model_dir)
            kept_columns = feature_info["kept_columns"]
            least_important_features = feature_info["least_important_features"]
            correlated_features_dropped = feature_info["correlated_features_dropped"]
            scaler = feature_info.get("scaler", None)
            X_holdout = feature_info["X_holdout"]
            y_holdout = feature_info["y_holdout"]
            median_values = feature_info["median_values"]
            selector = None
            selected_cols_variance = None
            pca = None

            models = load_models_for_label(label, os.path.join(model_dir, 'models'))
            test_preds = []

        # Prepare test set once
        fp_df, descriptor_df, valid_smiles, invalid_indices = smiles_to_combined_fingerprints_with_descriptors(test_smiles)
        # median_values = label_data["median_values"]
        if not descriptor_df.empty:
            # Safely align test columns to training columns. 
            # This adds any missing columns and fills them with NaN.
            descriptor_df = descriptor_df.reindex(columns=kept_columns)
            if Config.model_name == 'nn':
                # Fill NaN in test with median from train
                descriptor_df = descriptor_df.fillna(median_values)
            # Scale test set using the same scaler and kept_columns, then convert to DataFrame
            if getattr(Config, 'use_standard_scaler', False) and scaler is not None:
                descriptor_df = pd.DataFrame(scaler.transform(descriptor_df), columns=kept_columns, index=descriptor_df.index)
            descriptor_df.reset_index(drop=True, inplace=True)
            if not fp_df.empty:
                fp_df = fp_df.reset_index(drop=True)
                test = pd.concat([descriptor_df, fp_df], axis=1)
            else:
                test = descriptor_df
        else:
            test = fp_df

        if len(least_important_features) > 0:
            test = test.drop(columns=least_important_features)
        if len(correlated_features_dropped) > 0:
            print(f"Dropping correlated columns from test: {correlated_features_dropped}")
            test = test.drop(correlated_features_dropped, axis=1, errors='ignore')
        if getattr(Config, 'use_variance_threshold', False):
            test_sel = selector.transform(test)
            # Convert back to DataFrame
            test = pd.DataFrame(test_sel, columns=selected_cols_variance, index=test.index)
        # Optionally apply PCA to test set if enabled
        if getattr(Config, 'use_pca', False):
            test = pca.transform(test)
        # if label in ['Tg', 'Tc', 'Density', 'Rg']:
        #     X_holdout = X_holdout[features_keep]
        #     test = test[features_keep]
        # --- Holdout set evaluation with all trained models ---
        holdout_maes = []
        for i, Model in enumerate(models):
            is_torch_model = hasattr(Model, 'forward') and not hasattr(Model, 'predict')

            if is_torch_model:
                Model.eval()
                X_holdout_np = np.asarray(X_holdout) if isinstance(X_holdout, pd.DataFrame) else X_holdout
                test_np = np.asarray(test) if isinstance(test, pd.DataFrame) else test
                device = next(Model.parameters()).device
                with torch.no_grad():
                    X_holdout_tensor = torch.tensor(X_holdout_np, dtype=torch.float32).to(device)
                    test_tensor = torch.tensor(test_np, dtype=torch.float32).to(device)
                    y_holdout_pred = Model(X_holdout_tensor).detach().cpu().numpy().flatten()
                    y_test_pred = Model(test_tensor).detach().cpu().numpy().flatten()
            else:
                y_holdout_pred = Model.predict(X_holdout)
                y_test_pred = Model.predict(test)

            holdout_mae = mean_absolute_error(y_holdout, y_holdout_pred)
            print(f"Model {i+1} holdout MAE: {holdout_mae}")
            holdout_maes.append(holdout_mae)

            if isinstance(y_test_pred, pd.Series):
                y_test_pred = y_test_pred.values.flatten()
            else:
                y_test_pred = y_test_pred.flatten()        
            test_preds.append(y_test_pred)

        mean_holdout_mae = np.mean(holdout_maes)
        std_holdout_mae = np.std(holdout_maes)
        print(f"{label} Holdout MAE (mean ± std over all models): {mean_holdout_mae:.5f} ± {std_holdout_mae:.5f}")

        mae_results.append({
            'label': label,
            'fold_mae_mean': mean_fold_mae,
            'fold_mae_std': std_fold_mae,
            'holdout_mae_mean': mean_holdout_mae,
            'holdout_mae_std': std_holdout_mae
        })

        # Average test predictions across folds
        test_preds = np.array(test_preds)
        y_pred = np.mean(test_preds, axis=0)
        print(y_pred)
        new_column_name = label
        output_df[new_column_name] = y_pred

    # Save MAE results to CSV and display
    mae_df = pd.DataFrame(mae_results)
    mae_df.to_csv('NeurIPS/mae_results.csv', index=False)
    print("\nMean Absolute Error for each label:")
    print(mae_df)

# train_or_predict()

# output_df.to_csv('submission.csv', index=False)


# output_df = pd.DataFrame({
#     'id': test_ids
# })
# MODEL_DIR1 = '/kaggle/input/neurips-2025/nn_v4'
# train_or_predict(train_model=False, model_dir=MODEL_DIR1)
# # output_dfs.append(train_or_predict(output_df, train_model=False, model_dir=MODEL_DIR1))
# print(output_df)
# output_dfs.append(output_df.copy())


output_df = pd.DataFrame({
    'id': test_ids
})
MODEL_DIR1 = '/kaggle/input/neurips-2025/xgb_v3'
train_or_predict(train_model=False, model_dir=MODEL_DIR1)
print(output_df)
output_dfs.append(output_df.copy())











"""GNN"""


!pip install /kaggle/input/torch-geometric-2-6-1/torch_geometric-2.6.1-py3-none-any.whl


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool, global_max_pool
import torch.nn.functional as F
import warnings
import json
import torch
from sklearn.preprocessing import RobustScaler
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

os.makedirs("NeurIPS", exist_ok=True)
class Config:
    debug = False
    use_cross_validation = True  # Set to False to use a single split for speed
    use_external_data = True  # Set to True to use external datasets
    random_state = 42

# Create a single config instance to use everywhere
config = Config()

"""
Load competition data with complete filtering of problematic polymer notation
"""
print("Loading competition data...")
train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')

if config.debug:
    print("   Debug mode: sampling 1000 training examples")
    train = train.sample(n=1000, random_state=42).reset_index(drop=True)

print(f"Training data shape: {train.shape}, Test data shape: {test.shape}")

def clean_and_validate_smiles(smiles):
    """Completely clean and validate SMILES, removing all problematic patterns"""
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    
    # List of all problematic patterns we've seen
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        # Additional patterns that cause issues
        '([R])', '([R1])', '([R2])', 
    ]
    
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    
    # Additional check: if it contains ] followed by [ without valid atoms, likely polymer notation
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    
    # Try to parse with RDKit if available
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None
    
    # If RDKit not available, return cleaned SMILES
    return smiles

# Clean and validate all SMILES
print("Cleaning and validating SMILES...")
train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)

# Remove invalid SMILES
invalid_train = train['SMILES'].isnull().sum()
invalid_test = test['SMILES'].isnull().sum()

print(f"   Removed {invalid_train} invalid SMILES from training data")
print(f"   Removed {invalid_test} invalid SMILES from test data")

train = train[train['SMILES'].notnull()].reset_index(drop=True)
test = test[test['SMILES'].notnull()].reset_index(drop=True)

print(f"   Final training samples: {len(train)}")
print(f"   Final test samples: {len(test)}")

def add_extra_data_clean(df_train, df_extra, target):
    """Add external data with thorough SMILES cleaning"""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    print(f"      Processing {len(df_extra)} {target} samples...")
    
    # Clean external SMILES
    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    
    # Remove invalid SMILES and missing targets
    before_filter = len(df_extra)
    df_extra = df_extra[df_extra['SMILES'].notnull()]
    df_extra = df_extra.dropna(subset=[target])
    after_filter = len(df_extra)
    
    print(f"      Kept {after_filter}/{before_filter} valid samples")
    
    if len(df_extra) == 0:
        print(f"      No valid data remaining for {target}")
        return df_train
    
    # Group by canonical SMILES and average duplicates
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Fill missing values
    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['SMILES']==smile, target] = \
                df_extra[df_extra['SMILES']==smile][target].values[0]
            filled_count += 1
    
    # Add unique SMILES
    extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()
    if len(extra_to_add) > 0:
        for col in TARGETS:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        
        extra_to_add = extra_to_add[['SMILES'] + TARGETS]
        df_train = pd.concat([df_train, extra_to_add], axis=0, ignore_index=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'      {target}: +{n_samples_after-n_samples_before} samples, +{len(unique_smiles_extra)} unique SMILES')
    print(f"      Filled {filled_count} missing entries in train for {target}")
    print(f"      Added {len(extra_to_add)} new entries for {target}")
    return df_train

# Load external datasets with robust error handling
print("\n📂 Loading external datasets...")

external_datasets = []

# Function to safely load datasets
def safe_load_dataset(path, target, processor_func, description):
    try:
        if path.endswith('.xlsx'):
            data = pd.read_excel(path)
        else:
            data = pd.read_csv(path)
        
        data = processor_func(data)
        external_datasets.append((target, data))
        print(f"   ✅ {description}: {len(data)} samples")
        return True
    except Exception as e:
        print(f"   ⚠️ {description} failed: {str(e)[:100]}")
        return False

# Load each dataset
safe_load_dataset(
    '/kaggle/input/tc-smiles/Tc_SMILES.csv',
    'Tc',
    lambda df: df.rename(columns={'TC_mean': 'Tc'}),
    'Tc data'
)

safe_load_dataset(
    '/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv',
    'Tg', 
    lambda df: df[['SMILES', 'Tg']] if 'Tg' in df.columns else df,
    'TgSS enriched data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    'Tg',
    lambda df: df[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'}),
    'JCIM Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
    'Tg',
    lambda df: df.rename(columns={'Tg [K]': 'Tg'}).assign(Tg=lambda x: x['Tg'] - 273.15),
    'Xlsx Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_dnst1.xlsx',
    'Density',
    lambda df: df.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
                .query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
                .assign(Density=lambda x: x['Density'].astype(float) - 0.118),
    'Density data'
)

safe_load_dataset(
    BASE_PATH + 'train_supplement/dataset4.csv',
    'FFV', 
    lambda df: df[['SMILES', 'FFV']] if 'FFV' in df.columns else df,
    'dataset 4'
)

# Integrate external data
print("\n🔄 Integrating external data...")
train_extended = train[['SMILES'] + TARGETS].copy()

if getattr(config, "use_external_data", True) and  not config.debug:
    for target, dataset in external_datasets:
        print(f"   Processing {target} data...")
        train_extended = add_extra_data_clean(train_extended, dataset, target)

print(f"\n📊 Final training data:")
print(f"   Original samples: {len(train)}")
print(f"   Extended samples: {len(train_extended)}")
print(f"   Gain: +{len(train_extended) - len(train)} samples")

for target in TARGETS:
    count = train_extended[target].notna().sum()
    original_count = train[target].notna().sum() if target in train.columns else 0
    gain = count - original_count
    print(f"   {target}: {count:,} samples (+{gain})")

print(f"\n✅ Data integration complete with clean SMILES!")

def separate_subtables(train_df):
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    subtables = {}
    for label in labels:
        # Filter out NaNs, select columns, reset index
        subtables[label] = train_df[train_df[label].notna()][['SMILES', label]].reset_index(drop=True)

    return subtables

def augment_smiles_dataset(smiles_list, labels, num_augments=3):
    """
    Augments a list of SMILES strings by generating randomized versions.

    Parameters:
        smiles_list (list of str): Original SMILES strings.
        labels (list or np.array): Corresponding labels.
        num_augments (int): Number of augmentations per SMILES.

    Returns:
        tuple: (augmented_smiles, augmented_labels)
    """
    augmented_smiles = []
    augmented_labels = []

    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        # Add original
        augmented_smiles.append(smiles)
        augmented_labels.append(label)
        # Add randomized versions
        for _ in range(num_augments):
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented_smiles.append(rand_smiles)
            augmented_labels.append(label)

    return augmented_smiles, np.array(augmented_labels)

required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}

def augment_dataset(X, y, n_samples=1000, n_components=5, random_state=None):
    """
    Augments a dataset using Gaussian Mixture Models.

    Parameters:
    - X: pd.DataFrame or np.ndarray — feature matrix
    - y: pd.Series or np.ndarray — target values
    - n_samples: int — number of synthetic samples to generate
    - n_components: int — number of GMM components
    - random_state: int — random seed for reproducibility

    Returns:
    - X_augmented: pd.DataFrame — augmented feature matrix
    - y_augmented: pd.Series — augmented target values
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame or a NumPy array")

    X.columns = X.columns.astype(str)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or a NumPy array")

    df = X.copy()
    df['Target'] = y.values

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)

    synthetic_data, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)

    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

    X_augmented = augmented_df.drop(columns='Target')
    y_augmented = augmented_df['Target']

    return X_augmented, y_augmented


train_df=train_extended
test_df=test
subtables = separate_subtables(train_df)

test_smiles = test_df['SMILES'].tolist()
test_ids = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# ------------------------------------------------------------------
# --- GNN MODEL AND DATA PREPARATION ---
# ------------------------------------------------------------------

# A dictionary to map atom symbols to integer indices for the GNN
ATOM_MAP = {
    'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'H': 9,
    # --- NEWLY ADDED SYMBOLS ---
    'Si': 10, # Silicon
    'Na': 11, # Sodium
    '*' : 12, # Wildcard atom
    # --- NEWLY ADDED SYMBOLS ---
    'B': 13,  # Boron
    'Ge': 14, # Germanium
    'Sn': 15, # Tin
    'Se': 16, # Selenium
    'Te': 17, # Tellurium
    'Ca': 18, # Calcium
    'Cd': 19, # Cadmium
}

def smiles_to_graph(smiles_str: str, y_val=None):
    """
    Converts a SMILES string to a graph, adding selected global
    molecular features to each node's feature vector.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None: return None

        # 1. Calculate global features once per molecule
        global_features = [
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.MolLogP(mol)
        ]

        node_features = []
        for atom in mol.GetAtoms():
            # Initialize atom-specific features (one-hot encoding)
            atom_features = [0] * len(ATOM_MAP)
            symbol = atom.GetSymbol()
            if symbol in ATOM_MAP:
                atom_features[ATOM_MAP[symbol]] = 1

            # Add other standard atom features
            atom_features.extend([
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic())
            ])
            
            # 2. Append the global features to each atom's feature vector
            atom_features.extend(global_features)
            
            node_features.append(atom_features)
        
        if not node_features: return None
        x = torch.tensor(node_features, dtype=torch.float)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
            bond_type = bond.GetBondTypeAsDouble()
            edge_attrs.extend([[bond_type], [bond_type]])

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        if y_val is not None:
            y_tensor = torch.tensor([[y_val]], dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        else:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as e:
        return None

from rdkit.Chem import Descriptors

# A dictionary mapping labels to their most important global features from XGBoost
LABEL_SPECIFIC_FEATURES = {
    'Tg': [
        "HallKierAlpha", # Topological charge index
        "MolLogP",       # Lipophilicity
        "NumRotatableBonds", # Flexibility
        "TPSA",          # Polarity
    ],
    'FFV': [
        "NHOHCount",     # Count of NH and OH groups (H-bonding)
        "NumRotatableBonds",
        "MolWt",         # Size
        "TPSA",
    ],
    'Tc': [
        "MolLogP",
        "NumValenceElectrons",
        "SPS",           # Molecular shape index
        "MolWt",
    ],
    'Density': [
        "MolWt",
        "MolMR",         # Molar refractivity (related to volume)
        "FractionCSP3",  # Proportion of sp3 hybridized carbons (related to saturation)
        "NumHeteroatoms",
    ],
    'Rg': [
        "HallKierAlpha",
        "MolWt",
        "NumValenceElectrons",
        "qed",           # Quantitative Estimation of Drug-likeness
    ]
}

# A helper dictionary to easily call RDKit functions from their string names
RDKIT_DESC_CALCULATORS = {name: func for name, func in Descriptors.descList}
RDKIT_DESC_CALCULATORS['qed'] = Descriptors.qed # Add qed as it's not in the default list

from rdkit import Chem
import numpy as np

# This ATOM_MAP dictionary must be defined globally in your script (it already is)
# ATOM_MAP = {'C': 0, 'N': 1, ...}

def smiles_to_graph_label_specific(smiles_str: str, label: str, y_val=None):
    """
    (BASELINE VERSION - SIMPLE FEATURES)
    - This is the original hybrid GNN featurizer that produced your best score.
    - Node Features (x): Atom one-hot (20) + 5 atom features = 25 features.
    - Edge Features (edge_attr): Bond type as double = 1 feature.
    - Global Features (u): Label-specific descriptors are stored separately in 'data.u'.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None: 
            return None

        # --- 1. Calculate and store label-specific GLOBAL features ---
        global_features = []
        features_to_calculate = LABEL_SPECIFIC_FEATURES.get(label, [])
        
        for feature_name in features_to_calculate:
            calculator_func = RDKIT_DESC_CALCULATORS.get(feature_name)
            if calculator_func:
                try:
                    val = calculator_func(mol)
                    # Ensure value is valid, replace inf/nan with 0
                    global_features.append(val if np.isfinite(val) else 0.0)
                except Exception as e:
                    global_features.append(0.0)
            else:
                global_features.append(0.0)

        # --- 2. Create Node Features (SIMPLE) ---
        node_features = []
        for atom in mol.GetAtoms():
            # One-Hot Symbol (len 20, from global ATOM_MAP)
            atom_features = [0] * len(ATOM_MAP)
            symbol = atom.GetSymbol()
            if symbol in ATOM_MAP:
                atom_features[ATOM_MAP[symbol]] = 1

            # Standard Features (len 5)
            atom_features.extend([
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic())
            ])
            # Total features = 25
            node_features.append(atom_features)
        
        if not node_features: return None
        x = torch.tensor(node_features, dtype=torch.float)

        # --- 3. Create Edge Features (SIMPLE) ---
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
            bond_type = bond.GetBondTypeAsDouble()
            edge_attrs.extend([[bond_type], [bond_type]]) # 1-dim feature

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float) # Shape (0, 1)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # --- 4. Create Data Object ---
        data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_obj.u = torch.tensor([global_features], dtype=torch.float) # Store globals in 'u'

        if y_val is not None:
            data_obj.y = torch.tensor([[y_val]], dtype=torch.float)
        
        return data_obj
        
    except Exception as e:
        # Catch any other unexpected molecule-level errors
        print(f"CRITICAL ERROR converting SMILES '{smiles_str}': {e}")
        return None
            
class GNNModel(torch.nn.Module):
    """
    Defines the Graph Neural Network architecture.
    """
    def __init__(self, num_node_features, hidden_channels=128):
        super(GNNModel, self).__init__()
        torch.manual_seed(42)
        
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 4)
        self.lin = torch.nn.Linear(hidden_channels * 4, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_max_pool(x, batch) # Aggregate node features to get a graph-level embedding
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin(x)
        
        return x


def predict_with_gnn(trained_model, test_smiles):
    """
    Uses a pre-trained GNN model to make predictions on a list of test SMILES.
    """
    if trained_model is None:
        print("Prediction skipped because the GNN model is invalid.")
        return np.full(len(test_smiles), np.nan)

    print("--- Making predictions with trained GNN... ---")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert test SMILES to graph data
    test_data_list = [smiles_to_graph(s) for s in test_smiles]
    
    # We need to keep track of which original indices are valid
    valid_indices = [i for i, data in enumerate(test_data_list) if data is not None]
    valid_test_data = [data for data in test_data_list if data is not None]

    if not valid_test_data:
        print("Warning: No valid test molecules could be converted to graphs.")
        return np.full(len(test_smiles), np.nan)
        
    test_loader = PyGDataLoader(valid_test_data, batch_size=32, shuffle=False)

    trained_model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = trained_model(data)
            all_preds.append(out.cpu())

    # Combine predictions from all batches
    test_preds_tensor = torch.cat(all_preds, dim=0).numpy().flatten()
    
    # Create a full-sized prediction array and fill in the values at their original positions
    final_predictions = np.full(len(test_smiles), np.nan)
    if len(test_preds_tensor) == len(valid_indices):
        final_predictions[valid_indices] = test_preds_tensor
    else:
        print(f"Warning: Mismatch in GNN prediction count. This can happen with invalid SMILES.")
        fill_count = min(len(valid_indices), len(test_preds_tensor))
        final_predictions[valid_indices[:fill_count]] = test_preds_tensor[:fill_count]

    return final_predictions

import json
import os

def save_gnn_model(model, label, model_dir="models/gnn"):
    """
    (MODIFIED) Saves the GNN model state_dict and its full constructor config.
    """
    if model is None:
        print(f"Skipping save for {label}, model is None.")
        return

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"gnn_model_{label}.pth")
    config_path = os.path.join(model_dir, f"gnn_config_{label}.json")

    # Save the model parameters (the weights)
    torch.save(model.state_dict(), model_path)
    
    # Save the full configuration dictionary
    with open(config_path, 'w') as f:
        json.dump(model.config_args, f, indent=4)
        
    print(f"Saved final model for {label} to {model_path}")


def load_gnn_model(label, model_dir="models/gnn"):
    """
    (MODIFIED) Loads a saved GNN model using its full config file.
    """
    model_path = os.path.join(model_dir, f"gnn_model_{label}.pth")
    config_path = os.path.join(model_dir, f"gnn_config_{label}.json")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"Warning: Model or config file not found for {label}. Cannot load model.")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    try:
        # Re-initialize the model using all saved config args via dictionary unpacking
        model = TaskSpecificGNN(**config).to(DEVICE)
        
        # Load the saved model weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded saved model for {label} from {model_path}")
        return model

    except Exception as e:
        print(f"CRITICAL ERROR loading model for {label}: {e}")
        print("This may be due to a mismatch between the saved model and the current model class definition.")
        return None
    

def create_dynamic_mlp(input_dim, layer_list, dropout_list):
    """
    Helper function to dynamically build the task-specific MLP.
    """
    layers = []
    current_dim = input_dim
    
    for neurons, dropout in zip(layer_list, dropout_list):
        layers.append(torch.nn.Linear(current_dim, neurons))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        current_dim = neurons
        
    # Add the final single-output prediction layer
    layers.append(torch.nn.Linear(current_dim, 1))
    
    return torch.nn.Sequential(*layers)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class TaskSpecificGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features,
                 hidden_channels_gnn, mlp_neurons, mlp_dropouts, heads=8):
        super().__init__()
        torch.manual_seed(42)

        # --- 1. GNN Backbone (Using GATConv, No BatchNorm) ---
        self.convs = torch.nn.ModuleList()

        # Layer 1
        self.convs.append(
            GATConv(num_node_features, hidden_channels_gnn, heads=heads,
                    edge_dim=num_edge_features)
        )

        # Layer 2
        self.convs.append(
            GATConv(hidden_channels_gnn * heads, hidden_channels_gnn * 2, heads=heads,
                    edge_dim=num_edge_features)
        )

        # Layer 3 (Final GNN layer)
        self.convs.append(
            GATConv(hidden_channels_gnn * 2 * heads, hidden_channels_gnn * 4, heads=heads,
                    concat=False, edge_dim=num_edge_features)
        )

        gnn_output_dim = hidden_channels_gnn * 4

        # --- 2. Readout Head ---
        combined_feature_size = gnn_output_dim + num_global_features

        self.readout_mlp = create_dynamic_mlp(
            input_dim=combined_feature_size,
            layer_list=mlp_neurons,
            dropout_list=mlp_dropouts
        )

        # --- 3. Store config for saving/loading ---
        self.config_args = {
            'num_node_features': num_node_features,
            'num_edge_features': num_edge_features,
            'num_global_features': num_global_features,
            'hidden_channels_gnn': hidden_channels_gnn,
            'mlp_neurons': mlp_neurons,
            'mlp_dropouts': mlp_dropouts,
            'heads': heads
        }

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch

        # GNN Layers with ReLU and Dropout
        x = F.relu(self.convs[0](x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.convs[1](x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.convs[2](x, edge_index, edge_attr))

        # Readout
        graph_embedding = global_mean_pool(x, batch)
        combined_features = torch.cat([graph_embedding, u], dim=1)
        output = self.readout_mlp(combined_features)

        return output
        
# This is a new helper, just to make scaling code cleaner inside the loops
def scale_graph_features(data_list, u_scaler, x_scaler, atom_map_len):
    """Applies fitted scalers in-place to a list of Data objects."""
    try:
        for data in data_list:
            # 1. Scale global features (u)
            data.u = torch.tensor(u_scaler.transform(data.u.numpy()), dtype=torch.float)
            
            # 2. Scale continuous part of node features (x)
            x_one_hot = data.x[:, :atom_map_len]
            x_continuous = data.x[:, atom_map_len:]
            
            x_continuous_scaled = x_scaler.transform(x_continuous.numpy())
            x_continuous_scaled_tensor = torch.tensor(x_continuous_scaled, dtype=torch.float)
            
            # Recombine scaled features
            data.x = torch.cat([x_one_hot, x_continuous_scaled_tensor], dim=1)
            
    except Exception as e:
        print(f"CRITICAL ERROR applying scalers: {e}. Check feature dimensions. AtomMapLen={atom_map_len}")
        raise e
    return data_list


def train_gnn_model(label, train_data_list, val_data_list, mlp_neurons, mlp_dropouts, epochs=300): # Increased default epochs
    """
    (REVISED)
    - Accepts both train and val data lists.
    - Implements ReduceLROnPlateau scheduler based on val_loss.
    - Implements Early Stopping based on val_loss patience.
    """
    print(f"--- Training GNN for label: {label} ---")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not train_data_list:
        print(f"Warning: Empty train data list passed for {label}.")
        return None
    if not val_data_list:
        print(f"Warning: Empty validation data list passed for {label}.")
        return None

    # drop_last=True is important for training stability, prevents variance from tiny final batches.
    train_loader = PyGDataLoader(train_data_list, batch_size=32, shuffle=True, drop_last=True) 
    val_loader = PyGDataLoader(val_data_list, batch_size=32, shuffle=False) # No shuffle/drop for val

    # Get feature dimensions from the first data object
    first_data = train_data_list[0]
    num_node_features = first_data.x.shape[1]
    num_global_features = first_data.u.shape[1]
    num_edge_features = first_data.edge_attr.shape[1]
    
    print(f"Model Features (Scaled): Nodes={num_node_features}, Edges={num_edge_features}, Global={num_global_features}")

    model = TaskSpecificGNN(  # This should be your (no-BN) model class
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_global_features=num_global_features,
        hidden_channels_gnn=128, 
        mlp_neurons=mlp_neurons,
        mlp_dropouts=mlp_dropouts
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.L1Loss() 

    # --- 1. ADD SCHEDULER ---
    # This will cut the LR by half (factor=0.5) if val loss doesn't improve for 10 epochs (patience=10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # --- 2. ADD EARLY STOPPING VARS ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    PATIENCE_EPOCHS = 30  # Stop training if val loss doesn't improve for 30 straight epochs

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            if data.x.shape[0] <= 1: # Skip batches with one node (can happen)
                continue
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * data.num_graphs
        
        if len(train_loader.dataset) == 0:
            avg_train_loss = 0
        else:
            avg_train_loss = total_train_loss / len(train_loader.dataset)

        # --- 3. ADD VALIDATION LOOP (INSIDE EPOCH LOOP) ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                out = model(data)
                loss = criterion(out, data.y)
                total_val_loss += loss.item() * data.num_graphs
        
        if len(val_loader.dataset) == 0:
             avg_val_loss = 0
        else:
            avg_val_loss = total_val_loss / len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
             print(f"Epoch: {epoch:03d}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # --- 4. SCHEDULER & EARLY STOPPING LOGIC ---
        scheduler.step(avg_val_loss) # Feed validation loss to the scheduler

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE_EPOCHS and epoch > 50: # Give it at least 50 epochs to warm up
            print(f"--- Early stopping triggered at epoch {epoch} ---")
            break
            
    print(f"--- GNN training for {label} complete. Best Val Loss: {best_val_loss:.6f} ---")
    return model

def predict_with_gnn(trained_model, test_smiles, label, u_scaler, x_scaler, atom_map_len):
    """
    (MODIFIED for Full Scaling)
    - Requires both u_scaler (global) and x_scaler (node) to transform features.
    - Returns SCALED predictions.
    """
    if trained_model is None or u_scaler is None or x_scaler is None:
        print(f"Prediction skipped for {label} due to missing model or scaler.")
        return np.full(len(test_smiles), np.nan)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Featurize test data (features are NOT scaled yet)
    test_data_list = [smiles_to_graph_label_specific(s, label, y_val=None) for s in test_smiles]
    
    valid_indices = [i for i, data in enumerate(test_data_list) if data is not None]
    valid_test_data = [data for data in test_data_list if data is not None]

    if not valid_test_data:
        print(f"Warning: No valid test molecules could be converted for {label}.")
        return np.full(len(test_smiles), np.nan)
        
    # 2. Apply fitted scalers to all valid test features
    try:
        valid_test_data = scale_graph_features(valid_test_data, u_scaler, x_scaler, atom_map_len)
    except Exception as e:
        print(f"CRITICAL ERROR applying scalers during prediction: {e}.")
        return np.full(len(test_smiles), np.nan)

    test_loader = PyGDataLoader(valid_test_data, batch_size=32, shuffle=False) 

    trained_model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = trained_model(data)
            all_preds.append(out.cpu())

    test_preds_tensor = torch.cat(all_preds, dim=0).numpy().flatten()
    
    # Fill predictions array (these are SCALED predictions)
    final_predictions = np.full(len(test_smiles), np.nan)
    if len(test_preds_tensor) == len(valid_indices):
        final_predictions[valid_indices] = test_preds_tensor
    else:
        print(f"Warning: Mismatch in GNN prediction count for {label}.")
        fill_count = min(len(valid_indices), len(test_preds_tensor))
        final_predictions[valid_indices[:fill_count]] = test_preds_tensor[:fill_count]

    return final_predictions # These predictions are on the SCALED range


def train_or_predict_gnn(train_model=True, model_dir="models/gnn", n_splits=10):
    """
    (FINAL COMPLETE VERSION)
    - All data hardening (coerce, filter) and RobustScaler logic is included.
    - CV loop is modified to create a val_data_list.
    - Calls the new, optimized train_gnn_model with scheduler/early stopping.
    - Correctly passes all arguments (config['neurons'], config['dropouts']) to fix the TypeError.
    """
    
    ATOM_MAP_LEN = 20  # Make sure this matches your global ATOM_MAP
    
    # Plausible physical ranges to filter catastrophic outliers BEFORE scaling
    VALID_RANGES = {
        'Tg':      (-100, 500),  
        'FFV':     (0.01, 0.99), 
        'Tc':      (0, 1000),    
        'Density': (0.1, 3.0),   
        'Rg':      (0.1, 200)    
    }

    # MLP configs for the GNN readout head
    best_configs = {
        # Classic funnel, slightly lower final dropout
        "Tg":      {"neurons": [512, 256, 128], "dropouts": [0.5, 0.4, 0.2]},
        # Original wide funnel for this complex feature
        "Density": {"neurons": [1024, 256, 64], "dropouts": [0.5, 0.4, 0.3]},
        # Even wider and deeper, with strong regularization for presumed complexity
        "FFV":     {"neurons": [1024, 512, 64], "dropouts": [0.6, 0.5, 0.4]},
        # Slightly deeper than the simplest model to capture more features
        "Tc":      {"neurons": [128, 64], "dropouts": [0.4, 0.3]},
        # A gentle funnel instead of a pure block to encourage feature compression
        "Rg":      {"neurons": [128, 64, 64], "dropouts": [0.4, 0.3, 0.3]},
    }
    default_config = {"neurons": [128, 64], "dropouts": [0.3, 0.3]}

    output_df = pd.DataFrame({'id': test_df['id']})
    cv_mae_results = []
    os.makedirs(model_dir, exist_ok=True)
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)

    for label in labels: 
        print(f"\n{'='*20} Processing GNN for label: {label} {'='*20}")
        
        config = best_configs.get(label, default_config)
        print(f"Using MLP Config: Neurons={config['neurons']}, Dropouts={config['dropouts']}")
        
        ensemble_models = []
        y_scaler_path = os.path.join(model_dir, f"gnn_yscaler_{label}.joblib")
        u_scaler_path = os.path.join(model_dir, f"gnn_uscaler_{label}.joblib")
        x_scaler_path = os.path.join(model_dir, f"gnn_xscaler_{label}.joblib")
        
        if train_model:
            # --- START DATA HARDENING ---
            all_smiles_raw = subtables[label]['SMILES']
            all_y_raw = subtables[label][label] 
            
            all_y_numeric = pd.to_numeric(all_y_raw, errors='coerce')
            original_count = len(all_y_numeric)

            valid_min, valid_max = VALID_RANGES.get(label, (-np.inf, np.inf))
            valid_mask = (all_y_numeric >= valid_min) & (all_y_numeric <= valid_max) & (all_y_numeric.notna())
            
            all_y = all_y_numeric[valid_mask].reset_index(drop=True)
            all_smiles = all_smiles_raw[valid_mask].reset_index(drop=True)
            
            print(f"FILTERING: Coerced {original_count} rows. Kept {len(all_y)} valid rows within range ({valid_min}, {valid_max}).")
            
            if len(all_y) < (2 * n_splits): 
                print(f"CRITICAL: Not enough valid data ({len(all_y)}) to train for {label} with {n_splits} splits. Skipping.")
                continue
            # --- END DATA HARDENING ---

            # --- 1. FIT Y-SCALER (ROBUST) ---
            print("Using RobustScaler for Y-Scaler.")
            y_scaler = RobustScaler()  
            all_y_scaled = y_scaler.fit_transform(all_y.values.reshape(-1, 1)).flatten()
            joblib.dump(y_scaler, y_scaler_path)
            print(f"Saved Y-Scaler for {label}")

            # --- 2. FIT INPUT SCALERS (ROBUST) ---
            print("Pre-computing all graph features to fit input scalers...")
            all_train_graphs_raw = [smiles_to_graph_label_specific(s, label, None) for s in all_smiles]
            
            # Sync graph list with all data (skipping any SMILES that fail featurization)
            all_train_graphs_synced = []
            all_y_scaled_synced = [] 
            all_y_original_synced = [] # Also sync original Y for the CV split
            all_smiles_synced = []     # Also sync SMILES for the CV split
            
            for i, graph in enumerate(all_train_graphs_raw):
                if graph is not None:
                    all_train_graphs_synced.append(graph)
                    all_y_scaled_synced.append(all_y_scaled[i]) 
                    all_y_original_synced.append(all_y[i]) # Keep the original, unscaled, clean Y
                    all_smiles_synced.append(all_smiles[i]) # Keep the matching SMILES
            
            all_train_graphs = all_train_graphs_synced 
            all_y_scaled = np.array(all_y_scaled_synced)
            all_y_original_df = pd.Series(all_y_original_synced) # Store as Series for .iloc
            all_smiles_df = pd.Series(all_smiles_synced)         # Store as Series for .iloc

            if not all_train_graphs:
                print(f"CRITICAL: No valid training graphs could be featurized for {label}. Skipping.")
                continue
                
            all_u_data = np.concatenate([d.u.numpy() for d in all_train_graphs], axis=0)
            print("Using RobustScaler for U-Scaler.")
            u_scaler = RobustScaler().fit(all_u_data)  # Use RobustScaler
            joblib.dump(u_scaler, u_scaler_path)
            print(f"Saved U-Scaler for {label}")

            all_x_data = torch.cat([d.x for d in all_train_graphs], dim=0)
            all_x_continuous = all_x_data[:, ATOM_MAP_LEN:].numpy()
            print("Using RobustScaler for X-Scaler.")
            x_scaler = RobustScaler().fit(all_x_continuous)  # Use RobustScaler
            joblib.dump(x_scaler, x_scaler_path)
            print(f"Saved X-Scaler for {label}")

            # --- 3. APPLY SCALERS ---
            all_data_objects_scaled = scale_graph_features(all_train_graphs, u_scaler, x_scaler, ATOM_MAP_LEN)
            for i, data_obj in enumerate(all_data_objects_scaled):
                data_obj.y = torch.tensor([[all_y_scaled[i]]], dtype=torch.float)
            
            # --- 4. K-FOLD CV LOOP (MODIFIED) ---
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_val_scores = []
            fold_indices_gen = kf.split(all_data_objects_scaled) # Split the synced, valid, scaled data

            for fold, (train_idx, val_idx) in enumerate(fold_indices_gen):
                print(f"\n--- Fold {fold+1}/{n_splits} for {label} ---")
                
                train_data_list = [all_data_objects_scaled[i] for i in train_idx]
                val_data_list = [all_data_objects_scaled[i] for i in val_idx] # <-- CREATE VAL LIST
                
                val_smiles_list = all_smiles_df.iloc[val_idx].tolist()
                y_val_original = all_y_original_df.iloc[val_idx].values 

                fold_model = train_gnn_model(
                    label,
                    train_data_list, # Pass train data
                    val_data_list,   # <-- Pass val data
                    config['neurons'],    # <-- PASSES mlp_neurons
                    config['dropouts'],   # <-- PASSES mlp_dropouts (FIXES ERROR)
                    epochs=300       # <-- Train longer (will stop early)
                )
                
                if fold_model:
                    print("Running final validation prediction on the best model...")
                    val_preds_scaled = predict_with_gnn(fold_model, val_smiles_list, label, u_scaler, x_scaler, ATOM_MAP_LEN)
                    
                    train_y_scaled_median = 0.0 # RobustScaler median is 0
                    val_preds_scaled_filled = pd.Series(val_preds_scaled).fillna(train_y_scaled_median)
                    
                    val_preds_original = y_scaler.inverse_transform(
                        val_preds_scaled_filled.values.reshape(-1, 1)
                    ).flatten()

                    mae = mean_absolute_error(y_val_original, val_preds_original)
                    print(f"✅ Fold {fold+1} Validation MAE (Original Scale): {mae:.4f}")
                    fold_val_scores.append(mae)
                    
                    model_save_name = f"{label}_fold{fold}"
                    save_gnn_model(fold_model, model_save_name, model_dir)
                    ensemble_models.append(fold_model)
                else:
                    print(f"Warning: Training failed for Fold {fold+1}. Model will be skipped.")
            
            if fold_val_scores:
                avg_cv_mae = np.mean(fold_val_scores)
                print(f"\n{'*'*10} Average CV MAE for {label} (Original Scale): {avg_cv_mae:.4f} {'*'*10}")
                cv_mae_results.append({'label': label, 'avg_cv_mae': avg_cv_mae})

        else:
            # --- PREDICTION-ONLY MODE ---
            print(f"Loading {n_splits} models and ALL 3 RobustScalers for {label} ensemble...")
            model_path = '/kaggle/input/neurips-2025/GATConv_v29/models/gnn/'
            try:
                y_scaler = joblib.load(f'{model_path}gnn_yscaler_{label}.joblib')
                u_scaler = joblib.load(f'{model_path}gnn_uscaler_{label}.joblib')
                x_scaler = joblib.load(f'{model_path}gnn_xscaler_{label}.joblib')
                print("Loaded Y, U, and X RobustScalers.")
            except FileNotFoundError:
                print(f"CRITICAL: Scaler files not found for {label}. Cannot make predictions.")
                continue

            for fold in range(n_splits):
                loaded_model = load_gnn_model(f"{label}_fold{fold}", model_path.rstrip('/'))
                if loaded_model:
                    ensemble_models.append(loaded_model)
            
            if not ensemble_models: print(f"Warning: No models found for label {label}.")
            else: print(f"Successfully loaded {len(ensemble_models)} models for ensemble.")


        # --- ENSEMBLE PREDICTION STEP (Test Set) ---
        test_smiles = test_df['SMILES'].tolist()
        
        if ensemble_models and y_scaler and u_scaler and x_scaler:
            print(f"Making ensemble (scaled) predictions for {label} using {len(ensemble_models)} models...")
            all_fold_preds_scaled = []
            for model in ensemble_models:
                fold_test_preds_scaled = predict_with_gnn(model, test_smiles, label, u_scaler, x_scaler, ATOM_MAP_LEN)
                all_fold_preds_scaled.append(fold_test_preds_scaled)
            
            preds_stack_scaled = np.stack(all_fold_preds_scaled)
            final_ensemble_preds_scaled = np.nanmean(preds_stack_scaled, axis=0) 
            pred_series_scaled = pd.Series(final_ensemble_preds_scaled)
            
            pred_series_scaled_filled = pred_series_scaled.fillna(0.0) # Impute with scaled median (0.0)

            final_preds_original = y_scaler.inverse_transform(
                pred_series_scaled_filled.values.reshape(-1, 1)
            ).flatten()
            
            output_df[label] = final_preds_original
            
        else:
            print(f"No models or scalers available for {label}. Filling with (filtered) training median.")
            # Robust median fallback logic
            fallback_median = 0.0
            try:
                if 'all_y' in locals() and not all_y.empty:
                     fallback_median = all_y.median()
                else: 
                     print("Loading data to calculate fallback median...")
                     fb_y_raw = subtables[label][label]
                     fb_y_num = pd.to_numeric(fb_y_raw, errors='coerce')
                     valid_min, valid_max = VALID_RANGES.get(label, (-np.inf, np.inf))
                     fb_mask = (fb_y_num >= valid_min) & (fb_y_num <= valid_max) & (fb_y_num.notna())
                     fallback_median = fb_y_num[fb_mask].median()
                print(f"Using filtered median fallback: {fallback_median}")
            except Exception as e:
                 print(f"Error getting median, falling back to 0: {e}")
                 fallback_median = 0.0 
                 
            output_df[label] = fallback_median

    # --- Display final CV MAE summary ---
    if train_model and cv_mae_results:
        print("\n" + "="*40)
        print("📊 HYBRID GNN 5-Fold CV MAE Summary (Original Scale):")
        print("="*40)
        mae_df = pd.DataFrame(cv_mae_results)
        print(mae_df.to_string(index=False))
        mae_df.to_csv("gnn_hybrid_cv_mae_results.csv", index=False)
        print("\nCV results saved to gnn_hybrid_cv_mae_results.csv")

    submission_path = 'submission_hybrid_gnn_final.csv'
    output_df.to_csv(submission_path, index=False)
    print(f"\n✅ GNN Ensemble predictions (Original Scale) saved to {submission_path}")
    
    warnings.filterwarnings("default", "Mean of empty slice", RuntimeWarning)
    
    return output_df

# To train the models and then predict:
gnn_submission_df = train_or_predict_gnn(train_model=False)

output_dfs.append(gnn_submission_df)

print("\nGNN Submission Preview:")
print(gnn_submission_df.head())



# Average predictions from all output DataFrames
final_df = pd.concat(output_dfs, axis=0).groupby('id').mean().reset_index()
final_df.to_csv('submission.csv', index=False)
final_df