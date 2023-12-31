import streamlit as st
import pandas as pd
if 'TrainModel' not in vars():
    from model_trainer import TrainModel
df = pd.read_csv('./data/processed/planetary_systems/planets.csv')

with st.expander('View Data 🔎'):
    st.dataframe(df)

training_data = df[~df.life_exists.isna()]
test_df = df[df.life_exists.isna()]
training_data['life_exists'] = training_data['life_exists'].astype(int)
training_data = training_data.fillna(0) # Hint: you may want to handle missing values better than this


model_trainer = TrainModel(training_data, [
    # Hint: you may want to add/remove some of these features and/or engineer some new ones
    'num_stars_in_system', 
    'num_planets_in_system',
    'orbital_period', 
    'planet_radius_vs_earth', 
    'planet_mass_vs_earth',
    'planet_equilibrium_temperature', 
    'distance_to_system_in_light_years',
    'stellar_surface_gravity', 
    'stellar_metallicity'
], 'life_exists')

model_trainer.eda_features_vs_target()

log_reg_model = model_trainer.train_log_reg()
tree_model = model_trainer.train_decision_tree()
rf_model = model_trainer.train_rf()

test_df = test_df.fillna(0) # Hint: do this better
test_df = model_trainer.predict(rf_model, test_df, threshold=0.5)