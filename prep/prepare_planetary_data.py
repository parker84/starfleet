import pandas as pd
import coloredlogs, logging
from decouple import config
from life import Life
from plotly import express as px
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', 'INFO'), logger=logger)

# planetary systems data
planets = pd.read_csv('data/raw/planetary_systems/Planetary_Systems_Table_2023.12.28_13.54.25.csv')
logger.info(f'planets shape: {planets.shape}')
planets = planets.rename(columns={
    'sy_snum': 'num_stars_in_system',
    'sy_pnum': 'num_planets_in_system',
    'pl_orbper': 'orbital_period',
    'pl_rade': 'planet_radius_vs_earth',
    'pl_bmasse': 'planet_mass_vs_earth',
    'pl_eqt': 'planet_equilibrium_temperature',
    'sy_dist': 'distance_to_system_in_light_years',
    'st_logg': 'stellar_surface_gravity',
    'st_met': 'stellar_metallicity',
})

# cleaning nulls
predictive_features = [
    'planet_radius_vs_earth',
    'planet_mass_vs_earth',
    'planet_equilibrium_temperature',
    'stellar_surface_gravity'
]
null_counts = planets[[
    'planet_radius_vs_earth',
    'planet_mass_vs_earth',
    'planet_equilibrium_temperature',
    'stellar_surface_gravity'
]].isnull().sum(axis=0)
logger.info(f'null counts:\n{null_counts}')
planets = planets.dropna(axis=0, subset=predictive_features)
logger.info(f'planets shape after dropping nulls: {planets.shape}')

# shuffling and dropping duplicates
planets = planets.sample(frac=1, replace=False).reset_index(drop=True) # shuffle the order
planets.drop_duplicates(subset='pl_name', inplace=True) # drop duplicates
assert planets.pl_name.is_unique, 'planet name is not unique' # check for duplicates
logger.info(f'planets shape after dropping duplicates: {planets.shape}')
planets.index = range(planets.shape[0]) # reset the index



# star trek planets data
ab_planet_names = pd.read_csv('data/raw/planet_names/List of Star Trek planets (A–B).csv')
ab_planet_names = list(ab_planet_names.columns)[1:] # 0 is the name of the list
cf_planet_names = pd.read_csv('data/raw/planet_names/List of Star Trek planets (C–F).csv')
cf_planet_names = list(cf_planet_names.columns)[1:] # 0 is the name of the list
gl_planet_names = pd.read_csv('data/raw/planet_names/List of Star Trek planets (G–L).csv')
gl_planet_names = list(gl_planet_names.columns)[1:] # 0 is the name of the list
mq_planet_names = pd.read_csv('data/raw/planet_names/List of Star Trek planets (M–Q).csv')
mq_planet_names = list(mq_planet_names.columns)[1:] # 0 is the name of the list
rs_planet_names = pd.read_csv('data/raw/planet_names/List of Star Trek planets (R–s).csv')
rs_planet_names = list(rs_planet_names.columns)[1:] # 0 is the name of the list
tz_planet_names = pd.read_csv('data/raw/planet_names/List of Star Trek planets (T–Z).csv')
tz_planet_names = list(tz_planet_names.columns)[1:] # 0 is the name of the list
planet_names = ab_planet_names + cf_planet_names + gl_planet_names + mq_planet_names + rs_planet_names + tz_planet_names
planet_names_df = pd.DataFrame([])
planet_names_df['planet_name'] = planet_names
logger.info(f'planet_names shape: {planet_names_df.shape}')

# combining
planets_df = planets.join(planet_names_df, how='inner')[[
    'planet_name',
    'num_stars_in_system',
    'num_planets_in_system',
    'orbital_period',
    'planet_radius_vs_earth',
    'planet_mass_vs_earth',
    'planet_equilibrium_temperature',
    'distance_to_system_in_light_years',
    'stellar_surface_gravity',
    'stellar_metallicity'
]]
planets_df = planets_df[planets_df.planet_name != 'Earth']
planets_df = planets_df[planets_df.planet_name.str.lower().str.startswith('note') == False]
logger.info(f'planets_df shape: {planets_df.shape} (after combining with planet names)')

for col in planets_df.columns:
    if col != 'planet_name':
        p = px.histogram(planets_df, x=col, title=col)
        p.write_image(f'data/processed/planetary_systems/viz/{col}.png')

# creating life
life_df = pd.DataFrame([
    Life(
        row.planet_radius_vs_earth,
        row.planet_mass_vs_earth,
        row.planet_equilibrium_temperature,
        row.stellar_surface_gravity
    ).does_life_exist() 
    for index, row in planets_df.iterrows()
])
planets_df = planets_df.join(life_df)

logger.info(f'life exists value counts:\n{planets_df.life_exists.value_counts()}')
selected_cols = [
    'planet_name', 'life_exists', 'num_stars_in_system', 'num_planets_in_system',
    'orbital_period', 'planet_radius_vs_earth', 'planet_mass_vs_earth',
    'planet_equilibrium_temperature', 'distance_to_system_in_light_years',
    'stellar_surface_gravity', 'stellar_metallicity'
]
planets_df = planets_df[selected_cols]
train_df = planets_df.sample(frac=0.8, replace=False)
train_df['train'] = True
test_df = planets_df.join(train_df, how='left', rsuffix='_train')
test_df = test_df[test_df.train.isnull()]
train_df.to_csv('data/processed/planetary_systems/planets.csv', index=False)
test_df.to_csv('data/processed/planetary_systems/planets_holdout_data.csv', index=False)