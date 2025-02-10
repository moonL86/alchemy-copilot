This model is trained on both experimental data values and computed molecular descriptors to predict Viscosity (cP) and Pigment-to-Binder Ratio (PBR).

Trained_dataset- Table 2a from the research paper is used to train the model.

Predictions- Predicted values are Viscosity(in cP) and Pigment Binder Ratio only,
             the cost is calculated by simple method as price of each Independent variable(TiO2, Kaolin and PVA) are given in Table 1a
             formula for cost calculation used here is
             
             [{Mass(TiO2)*4500 + Mass(Kaolin)*40 + Mass(PVA)*2000}2] 
             multiplied by 2 for cost of 20L
             
Visuals- these visuals will be generated when running the code itself.

