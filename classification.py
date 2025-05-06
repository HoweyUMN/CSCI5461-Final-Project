# %% 
### Imports
import re, pandas as pd, numpy as np, scipy as sp, tensorflow as tf
from tqdm import tqdm
from pathlib import Path
import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, MaxAbsScaler

wandb.init()
# %%
### Load the Data
root = Path.cwd()
sample_paths = [p for p in root.iterdir()
                if p.is_dir() and (p/'matrix.mtx').exists()]

X_list = []
Y_list = []
gene_lists = []
# Load the data for each brain region
for path in tqdm(sample_paths, desc='Loading samples'):
    x = sp.io.mmread(path/'matrix.mtx').T.tocsr() # Provides raw count of each gene for given region
    genes = pd.read_csv(path/'features.tsv', sep='\t', header=None)[1].to_numpy()
    region = re.search(r'sMMr([A-Z]{3})', path.name).group(1) # Our class labels
    labels = np.repeat(region, x.shape[0])

    X_list.append(x);  Y_list.append(labels);  gene_lists.append(genes)

# %%
### Align the Genes
common_genes = set(gene_lists[0]).intersection(*gene_lists[1:]) # We need to make sure genes exist in all classes
keep_idx = [np.isin(g, list(common_genes)) for g in gene_lists]
X = sp.sparse.vstack([m[:, k] for m, k in zip(X_list, keep_idx)], format='csr')
Y = np.concatenate(Y_list)
genes = gene_lists[0][keep_idx[0]]         

# Limit to certain cell types using marker genes specific to these cell types
# marker_dict = {
#     'Vip'    : ['Vip'],              
#     'Pvalb'  : ['Pvalb'],            
#     'Sst'    : ['Sst'],              
#     'ItL23'  : ['Rorb'],             
#     'Astro'  : ['Aqp4', 'Gfap']      
# }

# gene2idx = {g: i for i, g in enumerate(genes)} # Get a numeric key for each gene

# # Build masks for each of the classes
# class_masks = {}
# for cname, markers in marker_dict.items():
#     # Keep only markers that exist in all datasets to ensure cells are present in all regions
#     idx = [gene2idx[m] for m in markers if m in gene2idx]
#     if not idx:
#         print(f"None of {markers} present after intersection; skipping {cname}")
#         continue
#     expr = X[:, idx].toarray().sum(1).ravel()   
#     class_masks[cname] = (expr > 0)

# # Concatenate all the cell data into one master list
# mask_matrix = np.column_stack(list(class_masks.values()))
# # Cells with multiple expressions are just sorted into their first marker
# winner = mask_matrix.argmax(1)                    
# keep   = mask_matrix.sum(1) > 0                    
# X, region = X[keep], Y[keep]
# subclass  = np.array(list(class_masks.keys()))[winner[keep]]

# print("class counts:", pd.Series(subclass).value_counts().to_dict())

# # Extract labels from brain region information 
# Y = region 


#%%
### Extract the genes with the most variance to improve performance
gene_var = X.power(2).mean(0).A1 - np.square(X.mean(0).A1)
top = np.argsort(gene_var)[-4000:]
X = X[:, top]

X = normalize(X, axis=1, norm='l1') * 1e4
X = X.copy();  X.data = np.log1p(X.data)
X = MaxAbsScaler(copy=False).fit_transform(X)

# Perform feature extraction using SVG, condense to 100 features
print('Extracting key features')
svd = TruncatedSVD(n_components=100, algorithm='randomized',
                   n_iter=10, random_state=0, )
X = svd.fit_transform(X)

#%%
# ### Load Presaved
# data   = np.load("svd_output/svd_features.npz")
# X  = data["X"]
# Y = data["regions"]

#%%
### Weight the classes
class_weight = {i: len(Y) / np.sum(Y == lab)
                for i, lab in enumerate(np.unique(Y))}
print("class weights:", class_weight)

# %% 
### Preprocess for training with NN
Y_enc = LabelEncoder().fit_transform(Y)
x_train, x_test, y_train, y_test = train_test_split(
        X, Y_enc, test_size=0.5, random_state=42, stratify=Y_enc)

scaler = StandardScaler(with_mean=False)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# %%
### Utilize a multilayer perceptron for classification 
NDIM = x_train.shape[1]
input = Input(shape=(NDIM,))
hidden = Dense(2 * NDIM, activation='relu', kernel_initializer='he_normal')(input)
hidden = Dropout(0.5)(hidden)
hidden = Dense(NDIM, activation='relu', kernel_initializer='he_normal')(input)
hidden = Dropout(0.5)(hidden)
hidden = Dense(NDIM, activation='relu', kernel_initializer='he_normal')(input)
hidden = Dropout(0.5)(hidden)
hidden = Dense(NDIM, activation='relu', kernel_initializer='he_normal')(input)
hidden = Dropout(0.5)(hidden)
hidden = Dense(NDIM, activation='relu', kernel_initializer='he_normal')(input)
hidden = Dropout(0.5)(hidden)
hidden = Dense(NDIM, activation='relu', kernel_initializer='he_normal')(hidden)
hidden = Dropout(0.5)(hidden)
output = Dense(len(np.unique(Y)), activation='softmax',
            kernel_initializer='he_normal')(hidden)

model = Model(input, output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

def schedule(epoch, lr):
    return lr * 0.01

history = model.fit(x_train, y_train, epochs=3, batch_size=16,
                    validation_split=0.5, verbose=1,
                    class_weight=class_weight,
                    callbacks = [WandbMetricsLogger(), tf.keras.callbacks.LearningRateScheduler()])

# %% 
### Evaluate the perceptron's performance
preds = model.predict(x_test, batch_size=32)
y_pred = np.argmax(preds, axis=1)
print(classification_report(y_test, y_pred, target_names=np.unique(Y)))
ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap='Blues', normalize='true',
        display_labels=np.unique(Y))

# %%
### Save preprocessed dataset
out_dir = Path("svd_output")          
out_dir.mkdir(exist_ok=True)         

np.savez_compressed(                 
    out_dir / "svd_features.npz",
    X=X.astype(np.float32),         
    regions=Y                        
)
print(f"SVD features saved ➜ {out_dir/'svd_features.npz'}")
# %%
