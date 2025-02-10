# Install required packages
!pip install rdkit-pypi torch matplotlib numpy pandas

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define chemical components with SMILES and properties
CHEM_COMPONENTS = {
    'Pigment': {
        'smiles': 'O=[Ti]=O',  # TiO2
        'descriptors': {}
    },
    'Extender': {
        'smiles': 'C(=O)([O-])[O-].[Ca+2]',  # Calcium Carbonate
        'descriptors': {}
    },
    'Binder': {
        'smiles': 'CC(=O)OCC(C)(C)O',  # Polyvinyl Acetate
        'descriptors': {}
    }
}

# Calculate molecular descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        'MolWt': Descriptors.MolWt(mol),
        'HeavyAtom': Descriptors.HeavyAtomCount(mol),
        'TPSA': Descriptors.TPSA(mol),
        'LogP': Descriptors.MolLogP(mol)
    }

# Precompute component descriptors
for component in CHEM_COMPONENTS:
    CHEM_COMPONENTS[component]['descriptors'] = calculate_descriptors(
        CHEM_COMPONENTS[component]['smiles']
    )

# Load experimental data from Book OG
data = pd.DataFrame({
    'Run': list(range(1, 21)),
    'Pigment(g)': [80,70,64,64,70,70,76,64,76,70,64,76,70,70,70,70,70,76,60,70],
    'Extender(g)': [500,500,559,559,500,500,559,441,441,400,441,559,500,500,500,500,600,441,500,500],
    'Binder(L)': [0.12,0.12,0.11,0.13,0.14,0.1,0.11,0.13,0.13,0.12,0.11,0.13,0.12,0.12,0.12,0.12,0.12,0.11,0.12,0.12],
    'Viscosity(poise)': [5.9,5.91,5.79,5.71,5.73,5.7,5.82,5.71,5.89,5.91,5.79,5.88,5.92,5.92,5.93,5.91,5.93,5.82,5.69,5.93],
    'Density(g/L)': [1287,1410,1260,1371,1418,1300,1260,1335,1283,1250,1360,1410,1400,1408,1400,1404,1300,1261,1376,1400],
    'pH': [6.31,7.45,7.21,8.77,8.6,7.35,6.55,6.35,6.32,6.59,6.74,7.35,7.75,7.9,7.71,8,7.99,6.9,6.41,7.01]
})

# Feature engineering function
def create_chemical_features(row):
    features = []

    # Get component amounts
    pigment = row['Pigment(g)']
    extender = row['Extender(g)']
    binder = row['Binder(L)']

    # Component descriptor features
    for component, amount in [('Pigment', pigment),
                             ('Extender', extender),
                             ('Binder', binder)]:
        desc = CHEM_COMPONENTS[component]['descriptors']
        features += [
            amount * desc['MolWt'],
            amount * desc['TPSA'],
            amount * desc['LogP'],
            amount * desc['HeavyAtom']
        ]

    # Chemical interaction features
    pig_desc = CHEM_COMPONENTS['Pigment']['descriptors']
    ext_desc = CHEM_COMPONENTS['Extender']['descriptors']
    bin_desc = CHEM_COMPONENTS['Binder']['descriptors']

    features += [
        # Cross-component interactions
        pigment * extender * pig_desc['LogP'] * ext_desc['LogP'],
        binder * pig_desc['TPSA'] + binder * ext_desc['TPSA'],
        (pigment/100) * (extender/100) * (bin_desc['MolWt'] * binder),
        # Solubility parameter approximations
        pig_desc['LogP'] * ext_desc['LogP'] * bin_desc['LogP'],
        (pigment * pig_desc['MolWt']) + (extender * ext_desc['MolWt'])
    ]

    return np.array(features)

# Create feature matrix
X = np.array([create_chemical_features(row) for _, row in data.iterrows()])
y = data[['Viscosity(poise)', 'Density(g/L)', 'pH']].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
X_train = x_scaler.transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.transform(y_train)
y_test = y_scaler.transform(y_test)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Neural network architecture
class ChemFormulationNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
model = ChemFormulationNN(X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop with early stopping
best_loss = float('inf')
patience = 100
counter = 0
train_losses = []
test_losses = []

for epoch in range(5000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_loss = criterion(test_outputs, y_test_t)

    # Track losses
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    # Early stopping
    if test_loss < best_loss:
        best_loss = test_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# Predictions
with torch.no_grad():
    train_pred = y_scaler.inverse_transform(model(X_train_t).numpy())
    test_pred = y_scaler.inverse_transform(model(X_test_t).numpy())
    y_train_true = y_scaler.inverse_transform(y_train)
    y_test_true = y_scaler.inverse_transform(y_test)

# Plot predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ['Viscosity(poise)', 'Density(g/L)', 'pH']
for i, ax in enumerate(axes):
    ax.scatter(y_train_true[:, i], train_pred[:, i], label='Train')
    ax.scatter(y_test_true[:, i], test_pred[:, i], label='Test')
    ax.plot([min(y_train_true[:,i]), max(y_train_true[:,i])],
            [min(y_train_true[:,i]), max(y_train_true[:,i])], 'k--')
    ax.set_title(metrics[i])
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.legend()
plt.tight_layout()
plt.show()

# Feature importance using permutation
def permutation_importance(model, X, y, criterion, n_repeats=5):
    baseline = criterion(model(X), y).item()
    importance = torch.zeros(X.shape[1])

    for i in range(X.shape[1]):
        X_perturbed = X.clone()
        for _ in range(n_repeats):
            X_perturbed[:, i] = X_perturbed[torch.randperm(X.shape[0]), i]
            importance[i] += criterion(model(X_perturbed), y).item()
        importance[i] = (importance[i]/n_repeats - baseline)/baseline

    return importance.numpy()

# Get feature importance
imp = permutation_importance(model, X_train_t, y_train_t, criterion)

# Plot feature importance
feature_names = [
    'Pig:MolWt', 'Pig:TPSA', 'Pig:LogP', 'Pig:HeavyAtom',
    'Ext:MolWt', 'Ext:TPSA', 'Ext:LogP', 'Ext:HeavyAtom',
    'Bin:MolWt', 'Bin:TPSA', 'Bin:LogP', 'Bin:HeavyAtom',
    'Pig-Ext-LogP', 'Bin-TPSA', 'Formulation-Mass', 'Cross-LogP', 'Total-Mass'
]

plt.figure(figsize=(12, 8))
plt.barh(feature_names, imp)
plt.title('Feature Importance Analysis')
plt.xlabel('Normalized Importance Score')
plt.tight_layout()
plt.show()

# Prepare Book1 predictions
book1_data = pd.DataFrame({
    'Run': list(range(1, 21)),
    'Pigment(g)': [80,70,64,64,70,70,76,64,76,70,64,76,70,70,70,70,70,76,60,70],
    'Extender(g)': [500,500,559,559,500,500,559,441,441,400,441,559,500,500,500,500,600,441,500,500],
    'Binder(L)': [0.12,0.12,0.11,0.13,0.14,0.1,0.11,0.13,0.13,0.12,0.11,0.13,0.12,0.12,0.12,0.12,0.12,0.11,0.12,0.12]
})

# Generate predictions
X_book = np.array([create_chemical_features(row) for _, row in book1_data.iterrows()])
X_book = x_scaler.transform(X_book)
X_book_t = torch.tensor(X_book, dtype=torch.float32)

with torch.no_grad():
    book_pred = y_scaler.inverse_transform(model(X_book_t).numpy())

# Format results
book1_data[['Viscosity(poise)', 'Density(g/L)', 'pH']] = np.round(book_pred, 2)
print("\nPredicted Formulation Properties for Book1:")
print(book1_data.to_string(index=False))
