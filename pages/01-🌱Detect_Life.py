
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
import streamlit as st
from streamlit_ace import st_ace
import pandas as pd
from projects.detect_life.model_trainer import TrainModel
import streamlit as st
import pandas as pd


st.set_page_config(page_title='StarFleet', page_icon='🛸', initial_sidebar_state="auto", menu_items=None)

st.title('Detecting Life 🌱')

st.markdown(
    """### Your Mission 🛸
You've been tasked with detecting life on planets in deep space.

This is incredibly important so we can properly evaluate the risks of exploring each new planet.

If you fail it's very likely that you and everyone on the ship will die.

Edit the **`Execution Code`** 🔧 below to predict whether or not life exists on unknown planets.

    """
)

st.markdown("### Predict Life 🌱")

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

st.markdown('### The Modelling Process 🧠')
exec(model_trainer_content)
exec(execution_content)

st.markdown('### Performance on Holdout Data 📊')
uploaded_file = st.file_uploader(
    "Upload Predictions CSV", 
    type="csv",
    help="Upload a CSV file with predictions on the holdout set, include columns `'pred_proba'` and `'pred'`"
)
if uploaded_file is not None:
    # Read the CSV file
    test_df_preds = pd.read_csv(uploaded_file)
    assert 'pred_proba' in test_df_preds.columns and 'pred' in test_df_preds.columns, "You need to include columns 'pred_proba' and 'pred' in your CSV"
    test_df_w_labels = pd.read_csv('./data/processed/planetary_systems/planets_holdout_data.csv')
    test_df = test_df_w_labels.merge(test_df_preds[['pred_proba', 'pred', 'planet_name']], on='planet_name', how='inner')
    assert test_df.shape[0] == test_df_w_labels.shape[0]
    target_col = 'life_exists'
    try:
        auc = round(roc_auc_score(test_df[target_col], test_df['pred_proba']), 2)
    except Exception as err:
        auc = None
    ap = round(average_precision_score(test_df[target_col], test_df['pred_proba']), 2)
    report = classification_report(
        test_df[target_col], 
        test_df['pred'], 
        target_names=['negative', 'positive']
    )
    st.markdown(
        f"""
```
auc: {auc},

average precision: {ap}, 

random precision: {round(test_df[target_col].mean(), 2)}

Classification Report: \n{report}
```
            """
        )