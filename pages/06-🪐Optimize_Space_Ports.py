
import streamlit as st
from streamlit_ace import st_ace


st.set_page_config(page_title='StarFleet', page_icon='🛸', initial_sidebar_state="auto", menu_items=None)

st.title('Optimize Space Ports 🪐')

st.markdown(
    """### Your Mission 🛸
On your last mission the Vultans discovered you (one of the top Data Science minds in the galaxy) were onboard this ship.

They've been experiencing an economic travesty due to the chaos in thir space ports (which are used to transport goods and services from this solar system across the galaxy).

Because of this, they've taken you and your entire crew hostage until you are able to bring stability back to their ports.

**You'll need to build 🛠️:**
1. A dashboard to monitor the ports
2. Demand Forecasting Model
3. Supply Chain Optimization

If you fail, they will execute you and everyone else on your ship.

Edit the **`Execution Code`** 🔧 below to build a robot that will protect the ship.
    """
)

st.markdown("### Coming Soon! 🚀")