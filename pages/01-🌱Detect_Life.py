
import streamlit as st
from streamlit_ace import st_ace


st.set_page_config(page_title='StarFleet', page_icon='🛸', initial_sidebar_state="auto", menu_items=None)

st.title('Detecting Life 🌱')

st.markdown(
    """### Your Mission 🛸
You've been tasked with detecting life on planets in deep space.

This is crucial for us to understand which planets we should explore and which we should avoid.

Edit the **`Execution Code`** 🔧 below to predict whether or not life exists on unknown planets.

    """
)

st.markdown("### Predict Life 👾")

with st.expander('README 📖', expanded=False):
    st.markdown(
        """
### The Code 🛠️
- Model Trainer Class 🦾 -> defines a class to train and evaluate various models (you can edit this)
- **`Execution Code`** 🔧 -> code that executes the model training and evaluation (you'll need to edit this)

### The Data 📊
We have data on `953` planets in deep space. We know the following about each planet:
- The planet's radius relative to Earth
- The planet's mass relative to Earth
- The planet's equilibrium temperature (in Kelvin)
- The star's surface gravity (in log10(cm/s**2))
- Whether or not life exists on some of the planets (this is what we're trying to predict)

We know for some of the planets which ones have life and which ones don't -> you can use these for training.
        """
    )

with open('./projects/detect_life/model_trainer.py', 'r') as f:
    model_trainer_code = f.read()
with open('./projects/detect_life/train_model.py', 'r') as f:
    train_model_code = f.read()

st.markdown("### The Code 🛠️")
with st.expander('Model Trainer Class 🦾 (editable)', expanded=False):
    model_trainer_content = st_ace(
        value=model_trainer_code,
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
        key=14
    )
with st.expander('**`Execution Code`** 🔧 (edit me!)', expanded=False):
    execution_content = st_ace(
        value=train_model_code.replace("""if 'TrainModel' not in vars():
    from model_trainer import TrainModel""", ""),
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
        key=123
    )

st.markdown('### The Results')
exec(model_trainer_content)
exec(execution_content)
