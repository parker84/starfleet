
import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page
from streamlit_ace import st_ace


st.set_page_config(page_title='StarFleet', page_icon='🛸', initial_sidebar_state="auto", menu_items=None)

st.title('Detecting Life 🌱')

st.markdown(
    """### The Mission 🛸
You've been tasked with detecting life on planets in deep space.

This is crucial for us to understand which planets we should explore and which we should avoid.

### Your Objective 🥅
For the planets that we don't know if life exists or not, you need to predict whether or not life exists on those planets.

If you predict accurately enough, you'll be able to move on to the next mission, if not, your crew fire you out of the airlock and you'll die in space.
    """
)

with st.expander('README 📖', expanded=False):
    st.markdown(
        """
        We have data on `953` planets in deep space. We know the following about each planet:
        - The planet's radius relative to Earth
        - The planet's mass relative to Earth
        - The planet's equilibrium temperature (in Kelvin)
        - The star's surface gravity (in log10(cm/s**2))
        - Whether or not life exists on some of the planets (this is what we're trying to predict)

        We know for some of the planets which ones have life and which ones don't -> you can use these for training.
        """
    )

st.markdown("### Predict Life 🌱")

with open('./projects/detect_life/train_model.py', 'r') as f:
    default_code = f.read()

with st.expander('**Code** 🛠️ (edit me!)', expanded=False):
    content = st_ace(
        value=default_code,
        language='python',
        theme='monokai',
        font_size=14,
        tab_size=4,
        wrap=False,
        show_gutter=True,
        show_print_margin=False,
        readonly=False,
        annotations=None,
        markers=None,
        auto_update=False,
        key=None
    )
exec(content)

