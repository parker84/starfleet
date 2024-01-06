import streamlit as st
import pandas as pd
if 'TrainModel' not in vars():
    from model_trainer import TrainModel

def convert_df(df):
    return df.to_csv().encode('utf-8')

training_data = pd.read_csv('./data/processed/planetary_systems/planets_train_data.csv')
test_df = pd.read_csv('./data/processed/planetary_systems/planets_holdout_data_no_labels.csv')

with st.expander('View Data 🔎'):
    st.dataframe(training_data)

training_data['life_exists'] = training_data['life_exists'].astype(int)
training_data = training_data.fillna(0) # Hint: you may want to handle missing values better than this
test_df = test_df.fillna(0) # Hint: do this better

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


test_df_w_predictions = model_trainer.predict(rf_model, test_df, threshold=0.5)
with st.expander("Holdout Set Predictions 📊", expanded=False):
    st.dataframe(test_df_w_predictions[['planet_name', 'pred_proba', 'pred']])
csv = convert_df(test_df_w_predictions[['planet_name', 'pred_proba', 'pred']])
st.download_button(
    label="Download Holdout Predictions as CSV 📥",
    data=csv,
    file_name='holdout_predictions.csv',
    mime='text/csv',
)