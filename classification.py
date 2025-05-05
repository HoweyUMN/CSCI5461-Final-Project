#%%
### Imports
import re
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

#%%
### Load Data

# Search local directory for folders containing 'matrix.mtx' files which should be expression data
root = Path.cwd()
sample_paths = [path for path in root.iterdir() if path.is_dir() and (path/'matrix.mtx').exists()]

# Load in samples 
X = []
Y = []
gene_lists = []
for path in tqdm(sample_paths, desc='Loading samples'): # Using tqdm for sanity 
    # Load in the matrix for training
    x = sp.io.mmread(path/'matrix.mtx').T.tocsr()
    # Load in the genes
    genes = pd.read_csv(path/'features.tsv', sep='\t', header=None)[1].to_numpy()
    # Load in the labels
    region = re.search(r'sMMr([A-Z]{3})', path.name).group(1)
    labels = np.repeat(region, x.shape[0])
    # Save to lists
    X.append(x)
    Y.append(labels)
    gene_lists.append(genes)

#%%
### Align genes
common_genes = set(gene_lists[0]).intersection(*gene_lists[1:])
keep_idx = [np.isin(gene_list, list(common_genes)) for gene_list in gene_lists]

# Filter the data
X_clean = [X[:, i] for X, i in zip(X, keep_idx)]
X = sp.sparse.vstack(X_clean, format='csr')
Y = np.concatenate(Y)

print(f'Full Dataset shape: {X.shape}, Labels: {np.unique(Y)}')

#%%
### Preprocess for Classification

# Remove genes which are overly or underly expressed heuristically to speed things up
expr_frac = X.getnnz(axis=0) / X.shape[0]        # vector length == n_genes
mask = (expr_frac >= 0.001) & (expr_frac <= 0.90) # 0.1‑90 % cells
print(f'Keeping {np.sum(mask)} genes out of {X.shape[1]}')
X = X[:, mask]

# Log-transform the data
X.data = np.log1p(X.copy().data)

# Train/Test Split
print('Splitting into train/test...')
Y = LabelEncoder().fit_transform(Y) 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

scaler = StandardScaler(with_mean=False)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#%%
### Define Perceptron Model
NDIM = x_train.shape[1]
input_layer = Input(shape=(NDIM,))
            
dense = Dense(NDIM, activation = 'relu', kernel_initializer='he_normal')(input_layer)
dense = Dropout(0.5)(dense)
dense = Dense(int(NDIM / 100), activation = 'relu', kernel_initializer='he_normal')(dense)
dense = Dropout(0.5)(dense)
dense = Dense(int(NDIM / 1000), activation = 'relu', kernel_initializer='he_normal')(dense)
dense = Dropout(0.5)(dense)

output = Dense(3, activation='softmax', kernel_initializer='he_normal')(dense)


model = Model(input_layer, output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
### Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)
print(classification_report(y_test, y_pred, target_names=['CA1', 'CA3', 'DG']))
ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=['CA1', 'CA3', 'DG'], cmap='Blues', normalize='true')
# %%
