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

indices = np.random.permutation(X.shape[0])
X = X[indices]
Y = Y[indices]

X = X[:700000]
Y = Y[:700000]

print(f'Full Dataset shape: {X.shape}, Labels: {np.unique(Y)}')

#%%
### Preprocess for Classification
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, MaxAbsScaler
# A) keep highly‑variable genes (2‑4 k) -------------------------------
gene_var = X.power(2).mean(0).A1 - np.square(X.mean(0).A1)
top = np.argsort(gene_var)[-4000:]          # top 4 000 HVGs
X = X[:, top]

# B) TF–IDF style normalise rows (optional but helps SVD & linear models)
X = normalize(X, axis=1, norm='l1') * 1e4     # counts per 10k
X.data = np.log1p(X.copy().data)                  # log1p in‑place

# C) MaxAbs scale (keeps sparsity) ------------------------------------
X = MaxAbsScaler(copy=False).fit_transform(X)

# D) Sparse SVD → dense PCs ------------------------------------------
svd = TruncatedSVD(n_components=100, algorithm='randomized',
                   n_iter=5, random_state=0)
X = svd.fit_transform(X)   

class_weight = {}
for i,lab in enumerate(np.unique(Y)):
    class_weight[i] = len(Y) / len(np.where(Y == lab)[0])
    print(f'Class {lab} weighted as {class_weight[i]}')

# Train/Test Split
print('Splitting into train/test...')
Y_enc = LabelEncoder().fit_transform(Y) 
x_train, x_test, y_train, y_test = train_test_split(X, Y_enc, test_size=0.5, random_state=42, stratify=Y)

scaler = StandardScaler(with_mean=False)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#%%
### Define Perceptron Model
NDIM = x_train.shape[1]
input_layer = Input(shape=(NDIM,))
            
dense = Dense(NDIM, activation = 'relu', kernel_initializer='he_normal')(input_layer)
# dense = Dropout(0.5)(dense)
dense = Dense(NDIM, activation = 'relu', kernel_initializer='he_normal')(dense)
# dense = Dropout(0.5)(dense)
dense = Dense(NDIM, activation = 'relu', kernel_initializer='he_normal')(dense)
# dense = Dropout(0.5)(dense)

output = Dense(len(np.unique(Y)), activation='softmax', kernel_initializer='he_normal')(dense)


model = Model(input_layer, output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
### Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.5, verbose=1, class_weight=class_weight)
preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)
print(classification_report(y_test, y_pred), target_names = np.unique(Y).tolist())
ConfusionMatrixDisplay.from_predictions(y_pred, y_test, cmap='Blues', normalize='true', display_labels=np.unique(Y).tolist())
# %%
