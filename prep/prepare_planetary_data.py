import pandas as pd
import plotly.express as px
from life import Life

# planetary systems data
planets = pd.read_csv('data/raw/planetary_systems/Planetary_Systems_Table_2023.12.28_13.54.25.csv')
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
# planets = planets.dropna(axis=0)
planets = planets.sample(frac=1, replace=False).reset_index(drop=True) # shuffle the order
planets.drop_duplicates(subset='pl_name', inplace=True) # drop duplicates
assert planets.pl_name.is_unique, 'planet name is not unique' # check for duplicates
planets.index = range(planets.shape[0]) # reset the index

# star trek planets data
ab_star_trek_planets = pd.read_csv('data/raw/star_trek_planets/List of Star Trek planets (A–B).csv')
ab_star_trek_planets = list(ab_star_trek_planets.columns)[1:] # 0 is the name of the list
cf_star_trek_planets = pd.read_csv('data/raw/star_trek_planets/List of Star Trek planets (C–F).csv')
cf_star_trek_planets = list(cf_star_trek_planets.columns)[1:] # 0 is the name of the list
gl_star_trek_planets = pd.read_csv('data/raw/star_trek_planets/List of Star Trek planets (G–L).csv')
gl_star_trek_planets = list(gl_star_trek_planets.columns)[1:] # 0 is the name of the list
mq_star_trek_planets = pd.read_csv('data/raw/star_trek_planets/List of Star Trek planets (M–Q).csv')
mq_star_trek_planets = list(mq_star_trek_planets.columns)[1:] # 0 is the name of the list
rs_star_trek_planets = pd.read_csv('data/raw/star_trek_planets/List of Star Trek planets (R–s).csv')
rs_star_trek_planets = list(rs_star_trek_planets.columns)[1:] # 0 is the name of the list
tz_star_trek_planets = pd.read_csv('data/raw/star_trek_planets/List of Star Trek planets (T–Z).csv')
tz_star_trek_planets = list(tz_star_trek_planets.columns)[1:] # 0 is the name of the list
star_trek_planets = ab_star_trek_planets + cf_star_trek_planets + gl_star_trek_planets + mq_star_trek_planets + rs_star_trek_planets + tz_star_trek_planets
star_trek_planets_df = pd.DataFrame([])
star_trek_planets_df['planet_name'] = star_trek_planets

# combining
planets_df = planets.join(star_trek_planets_df, how='inner')[[
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

print(planets_df['life_exists'].value_counts())
planets_df.to_csv('data/processed/planetary_systems/planets.csv', index=False)