import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title='StarFleet', page_icon='🛸', initial_sidebar_state="auto", menu_items=None)

st.title('StarFleet 🛸')


st.markdown(
    """
Congrats on your new role as the Chief Data Science Officer on the USX Enterprise (the best starship in the United Federation of Galaxies fleet).

You and your team are about to embark on a journey to explore the final frontier - space 🌌.

You specifically will be tasked with leveraging data science 🔬 to make a meaningful impact on this journey, do the job well and you’ll play a crucial role in:

- Discovering unknown civilizations in deep space
- Defending those in need
- Ensuring the safety and well being of your crew

... And so much more

Do the job poorly and you and your entire crew will die.
    """)


begin = st.button('Begin Your Journey', type='primary')
if begin:
    switch_page('detect life')