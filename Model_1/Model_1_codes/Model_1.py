!pip install rdkit-pypi torch matplotlib numpy pandas xlsxwriter

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
    'TiO2': {
        'smiles': 'O=[Ti]=O',
        'descriptors': {}
    },
    'Kaolin': {
        'smiles': 'O.O=[Si]=O.[Al+3]',  # Simplified representation
        'descriptors': {}
    },
    'PVA': {
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

# Load experimental training data
experimental_data = pd.DataFrame({
    'Run': list(range(1, 21)),
    'TiO2 (kg)': [0.625,0.75,0.5,0.835224,0.5,0.625,0.625,0.625,0.5,0.625,
                  0.75,0.625,0.625,0.5,0.625,0.75,0.625,0.625,0.414776,0.75],
    'Kaolin (kg)': [0.3011345,0.1,0.1,0.175,0.1,0.0488655,0.175,0.175,0.25,0.175,
                    0.1,0.175,0.175,0.25,0.175,0.25,0.175,0.175,0.175,0.25],
    'PVA (l)': [1.3,1.2,1.2,1.3,1.4,1.3,1.3,1.3,1.4,1.131821,1.4,1.3,1.468179,
                1.2,1.3,1.2,1.3,1.3,1.3,1.4],
    'Viscosity (cP)': [1198,884,1030,1008,943,630,605,387,342,360,600,558,805,
                       840,556,679,960,665,408,555],
    'PBR': [4.1,4.39,4.23,4.15,3.63,3.95,4.02,4.02,3.71,4.62,3.76,4.02,3.56,
            4.33,4.02,4.49,4.02,4.02,3.9,3.85]
})

# Load prediction dataset
prediction_data = pd.DataFrame({
    'Run': list(range(1, 21)),
    'TiO2 (kg)': [0.75,0.625,0.625,0.75,0.414776,0.75,0.625,0.625,0.625,0.5,
                  0.625,0.5,0.625,0.75,0.5,0.5,0.625,0.835224,0.625,0.625],
    'Kaolin (kg)': [0.1,0.175,0.175,0.25,0.175,0.1,0.175,0.175,0.0488655,0.25,
                    0.175,0.25,0.301134,0.25,0.1,0.1,0.175,0.175,0.175,0.175],
    'PVA (l)': [1.4,1.3,1.3,1.4,1.3,1.2,1.3,1.46818,1.3,1.4,1.3,1.2,1.3,1.2,
                1.2,1.4,1.13182,1.3,1.3,1.3]
})

# Feature engineering function
def create_chemical_features(row):
    features = []

    # Component amounts
    tio2 = row['TiO2 (kg)']
    kaolin = row['Kaolin (kg)']
    pva = row['PVA (l)']

    # Component descriptor features
    for component, amount in [('TiO2', tio2),
                             ('Kaolin', kaolin),
                             ('PVA', pva)]:
        desc = CHEM_COMPONENTS[component]['descriptors']
        features += [
            amount * desc['MolWt'],
            amount * desc['TPSA'],
            amount * desc['LogP'],
            amount * desc['HeavyAtom']
        ]

    # Chemical interaction features
    tio2_desc = CHEM_COMPONENTS['TiO2']['descriptors']
    kaolin_desc = CHEM_COMPONENTS['Kaolin']['descriptors']
    pva_desc = CHEM_COMPONENTS['PVA']['descriptors']

    features += [
        tio2 * kaolin * tio2_desc['LogP'] * kaolin_desc['LogP'],
        pva * tio2_desc['TPSA'] + pva * kaolin_desc['TPSA'],
        (tio2/1) * (kaolin/1) * (pva_desc['MolWt'] * pva),
        tio2_desc['LogP'] * kaolin_desc['LogP'] * pva_desc['LogP'],
        (tio2 * tio2_desc['MolWt']) + (kaolin * kaolin_desc['MolWt'])
    ]

    return np.array(features)

# Create feature matrix and targets from experimental data
X = np.array([create_chemical_features(row) for _, row in experimental_data.iterrows()])
y = experimental_data[['Viscosity (cP)', 'PBR']].values

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
class FormulationNN(nn.Module):
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
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
model = FormulationNN(X.shape[1])
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

# Generate predictions for the prediction dataset
X_pred = np.array([create_chemical_features(row) for _, row in prediction_data.iterrows()])
X_pred = x_scaler.transform(X_pred)
X_pred_t = torch.tensor(X_pred, dtype=torch.float32)

with torch.no_grad():
    preds = y_scaler.inverse_transform(model(X_pred_t).numpy())

# Add predictions to prediction dataframe
prediction_data['Predicted Viscosity (cP)'] = np.round(preds[:, 0], 1)
prediction_data['Predicted PBR'] = np.round(preds[:, 1], 2)

# Create final output dataframe
final_predictions = prediction_data[['Run', 'TiO2 (kg)', 'Kaolin (kg)', 'PVA (l)',
                                    'Predicted Viscosity (cP)', 'Predicted PBR']]

# Save to Excel
output_filename = 'final_predictions.xlsx'
with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
    final_predictions.to_excel(writer, sheet_name='Predictions', index=False)
    experimental_data.to_excel(writer, sheet_name='Training Data', index=False)

print(f"Predictions saved to {output_filename}")
print("\nFinal Prediction Results:")
print(final_predictions.to_string(index=False))

# Plotting predicted vs actual for training data
with torch.no_grad():
    train_pred = y_scaler.inverse_transform(model(X_train_t).numpy())
    y_train_true = y_scaler.inverse_transform(y_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
metrics = ['Viscosity (cP)', 'PBR']
for i, ax in enumerate(axes):
    ax.scatter(y_train_true[:, i], train_pred[:, i])
    ax.plot([min(y_train_true[:,i]), max(y_train_true[:,i])],
            [min(y_train_true[:,i]), max(y_train_true[:,i])], 'k--')
    ax.set_title(f'Training Data: {metrics[i]}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
plt.tight_layout()
plt.show()
