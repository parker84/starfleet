
import streamlit as st
from streamlit_ace import st_ace


st.set_page_config(page_title='StarFleet', page_icon='🛸', initial_sidebar_state="auto", menu_items=None)

st.title('Defending The Ship 👾')

st.markdown(
    """### Your Mission 🛸
You've been tasked with defending the ship against alien invaders.

**You'll need to build 🛠️:**
- An Environment to simulate the attacks
- Reinforcement Learning Agent to Defend The Ship

If you fail, you and everyone on the ship will die.

Expand the sections below 👇🏻 to begin your journey.
    """
)


with st.expander('RL 🤖 Environment in `43` lines of Python'):

    st.markdown("""The agent does nothing productive, action space is only 2D, and the state remains static.

But the Classes + Methods + Training Loop remains the same when building out an agent to battle invaders in space 👾.
    """)

    with open('projects/defend/dummy_env.py', 'r') as f:
        code = f.read()
        st.code(code, language='python')

st.markdown("### More Coming Soon! 👾")
