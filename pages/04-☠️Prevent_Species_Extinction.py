
import streamlit as st
from streamlit_ace import st_ace


st.set_page_config(page_title='StarFleet', page_icon='🛸', initial_sidebar_state="auto", menu_items=None)

st.title('☠️ Prevent Species Extinction')

st.markdown(
    """### Your Mission 🛸
You and your crew can not only travel through space, but also time.

You need to forecast the population growth of every planet in the galaxy for the next 1000 years.

So we can identify when which planets are going to hit their carrying capacity and need our help to avoid extinction.

**You'll need to build 🛠️:**
1. A Model to Forecast Population Growth for each planet
2. A Model to Forecast Carry Capacity for each planet
3. Analyze and give recommendations on which planets need our help and when

If you fail, you and everyone on the ship will die.

Edit the **`Execution Code`** 🔧 below to build a robot that will protect the ship.
    """
)

st.markdown("### Coming Soon! ☠️")