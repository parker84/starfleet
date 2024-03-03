
import streamlit as st
from streamlit_ace import st_ace
import pandas as pd
import statsmodels.formula.api as smf
import plotly.express as px

st.set_page_config(page_title='StarFleet', page_icon='🛸', initial_sidebar_state="auto", menu_items=None)

st.title('Shuttle Launch 🚀')

st.markdown(
    """### Your Mission 🛸
We're building a space craft for our mission.

We have collected data on multiple un-manned launches of our space craft and tomorrow is the day we're supposed to launch with everyone on board.

Tomorrow's temperature is supposed to be 31 degrees fareinheit using this and the data below you need decide whether or not we should launch the space craft.
    """
)

st.markdown("### The Data 📊")
COLUMN_NAMES = [
    'num_o_rings_at_risk',
    'num_o_rings_w_thermal_distress',
    'launch_temp',
    'leak_check_pressure',
    'order_of_flight'
]

oring_erosion_or_blowby = pd.read_csv(
    './data/raw/shuttle_launch/o-ring-erosion-or-blowby.data',
    header=None,
    delimiter=r"\s+",
)
oring_erosion_or_blowby.columns = COLUMN_NAMES

oring_erosion_or_blowby['rings_w_thermal_distress'] = 1 * (
    oring_erosion_or_blowby['num_o_rings_w_thermal_distress'] > 0
)

with st.expander('Code 🛠️', expanded=False):
    code = st.code(
        """
COLUMN_NAMES = [
    'num_o_rings_at_risk',
    'num_o_rings_w_thermal_distress',
    'launch_temp',
    'leak_check_pressure',
    'order_of_flight'
]

oring_erosion_or_blowby = pd.read_csv(
    './data/raw/shuttle_launch/o-ring-erosion-or-blowby.data',
    header=None,
    delimiter=r"\s+",
)
oring_erosion_or_blowby.columns = COLUMN_NAMES

oring_erosion_or_blowby['rings_w_thermal_distress'] = 1 * (
    oring_erosion_or_blowby['num_o_rings_w_thermal_distress'] > 0
)
"""
    )

st.dataframe(oring_erosion_or_blowby)

st.markdown("### The Model 🦾")

with st.expander('Code 🛠️', expanded=False):
    code = st.code(
        """
model = smf.logit(
    "rings_w_thermal_distress ~ num_o_rings_at_risk + launch_temp + leak_check_pressure + order_of_flight",
    data=oring_erosion_or_blowby,
).fit()

st.text(model.summary())

avgs = oring_erosion_or_blowby.mean(axis=0)
avgs['launch_temp'] = 31
prob_of_rings_w_distress_or_blowby =  model.predict(avgs)
st.markdown(f"Probability of O-Rings with Distress or Blowby at `31` degress fareinheit: `{int(100 * prob_of_rings_w_distress_or_blowby[0])}%`")

preds_df = pd.DataFrame([avgs for _ in range(80)])
preds_df['launch_temp'] = range(20, 100)
preds_df['prob'] = model.predict(preds_df)

p = px.line(
    preds_df,
    x="launch_temp",
    y="prob",
    title="Probability of O-Rings with Distress or Blowby vs Temperature",
    labels={
        "launch_temp": "Launch Temperature", 
        "prob": "Probability of O-Rings with Distress or Blowby"
    },
)
p.add_trace(
    px.scatter(
        x=oring_erosion_or_blowby['launch_temp'], 
        y=oring_erosion_or_blowby['rings_w_thermal_distress']
    ).data[0]
)
st.plotly_chart(p)
"""
    )

model = smf.logit(
    "rings_w_thermal_distress ~ launch_temp + leak_check_pressure",
    data=oring_erosion_or_blowby,
).fit()

st.text(model.summary())

avgs = oring_erosion_or_blowby.mean(axis=0)
avgs['launch_temp'] = 31
prob_of_rings_w_distress_or_blowby =  model.predict(avgs)
st.markdown(f"Probability of O-Rings with Distress or Blowby at `31` degress fareinheit: `{int(100 * prob_of_rings_w_distress_or_blowby[0])}%`")

preds_df = pd.DataFrame([avgs for _ in range(80)])
preds_df['launch_temp'] = range(20, 100)
preds_df['prob'] = model.predict(preds_df)

p = px.line(
    preds_df,
    x="launch_temp",
    y="prob",
    title="Probability of O-Rings with Distress or Blowby vs Temperature",
    labels={
        "launch_temp": "Launch Temperature", 
        "prob": "Probability of O-Rings with Distress or Blowby"
    },
)
p.add_trace(
    px.scatter(
        x=oring_erosion_or_blowby['launch_temp'], 
        y=oring_erosion_or_blowby['rings_w_thermal_distress']
    ).data[0]
)
st.plotly_chart(p)

st.markdown("### Data Source 📚")
st.markdown(
    """
This data is from an actual space shuttle launch. 

A terminal mistake was made analyzing the data the night before the launch and the Challenger space shuttle on January 28, 1986 and the shuttle exploded 73 seconds after launch.

Here's the source of the data: [Challenger USA Space Shuttle O-Ring](https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring)
"""
)

# TODO: clean this all up