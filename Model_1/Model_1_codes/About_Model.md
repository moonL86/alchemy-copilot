Parameters Used for Training:
Raw Ingredient Amounts (from experimental data):

TiO₂ (kg)
Kaolin (kg)
PVA (l)


Molecular Descriptors (calculated using RDKit for each component):

Molecular Weight (MolWt)

Topological Polar Surface Area (TPSA)

LogP (Partition Coefficient)

Heavy Atom Count


Feature Engineering (Interactions between chemical properties):

LogP interactions: TiO₂ * Kaolin * (LogP of both)

TPSA interactions: PVA * (TPSA of TiO₂ + TPSA of Kaolin)

Mass-based interactions: (TiO₂ * Kaolin * Molecular Weight of PVA)

Combined LogP effect: LogP of all three components multiplied

Summed molecular weights: Contribution of TiO₂ and Kaolin molecular weights


Target Variables (Outputs):

Viscosity (cP)

PBR (Pigment-to-Binder Ratio)


Training Setup:

Neural Network with 4 layers:

Input → 32 neurons → 16 neurons → 8 neurons → Output (2 neurons)

Activation: ReLU

Dropout (0.3) for regularization

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam (learning rate = 0.001, weight decay = 1e-4)
Early Stopping: Stops if no improvement in 100 epochs
