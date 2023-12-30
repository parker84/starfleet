import pickle
import pandas as pd
from matplotlib import pyplot as plt
from plotly import express as px
import statsmodels.formula.api as smf
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
SEED = 42

class TrainModel:

    def __init__(self, data: pd.DataFrame, feature_cols: list, target_col: str) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.train_df, self.val_df = self._train_test_split(data)

    def _train_test_split(self, data):
        return train_test_split(data, test_size=.2, random_state=SEED)
    
    def train_rf(self):
        with st.spinner('Training Random Forest...'):
            model = RandomForestClassifier(
                random_state=SEED,
                n_estimators=500, # number of trees
                max_depth=3, # max depth of each tree
                min_samples_leaf=10, # min samples in each leaf of each tree
            )
            model.fit(self.train_df[self.feature_cols], self.train_df[self.target_col])
            feature_imps = self.get_feature_importances(model)
        with st.expander('Random Forest 🎄'):
            st.markdown(
                f"""
                Random Forest Model:
                `{model}`
                #### Feature Importances:
                """
            )
            st.dataframe(feature_imps)
            self.eval_model(model)
            self.plot_eval_curves(model, self.val_df)
        return model
    
    def train_log_reg(self):
        with st.spinner('Training Logistic Regression...'):
            smf_model = smf.logit(f'{self.target_col} ~ {" + ".join(self.feature_cols)}', data=self.train_df).fit()
            skl_model = LogisticRegression(
                random_state=SEED
            )
            skl_model.fit(self.train_df[self.feature_cols], self.train_df[self.target_col])
            feature_imps = self.get_feature_importances(skl_model)
        with st.expander('Logisitic Regression 📈'):
            st.text(
                f"""
Stats Models Summary:\n
{smf_model.summary()}

Sklearn Model Coefficients:\n
{skl_model.coef_}

Sklearn Model Feature Importantces:
                """
            )
            st.dataframe(feature_imps)
            self.eval_model(skl_model)
            self.plot_eval_curves(skl_model, self.val_df)
        return skl_model
    
    def train_decision_tree(self):
        with st.spinner('Training Decision Tree...'):
            model = DecisionTreeClassifier(max_depth=3, random_state=1234)
            model.fit(self.train_df[self.feature_cols], self.train_df[self.target_col])
            feature_imps = self.get_feature_importances(model)
        with st.expander('Decision Tree 🌲'):
            fig = plt.figure(figsize=(25,20))
            _ = tree.plot_tree(model, 
                            feature_names=self.feature_cols,  
                            class_names=[self.target_col + '_0', self.target_col + '_1'],
                            filled=True,
                            proportion=True)
            st.pyplot(fig)
            st.dataframe(feature_imps)
            self.eval_model(model)
            self.plot_eval_curves(model, self.val_df)
            return model

    def get_feature_importances(self, model):
        imps = permutation_importance(
            model,
            self.train_df[self.feature_cols], 
            self.train_df[self.target_col], 
            scoring=['average_precision', 'roc_auc'],
            random_state=SEED
        )
        imp_df = pd.DataFrame([])
        imp_df['features'] = self.feature_cols
        imp_df['average_precision'] = imps['average_precision']['importances_mean']
        imp_df['roc_auc'] = imps['roc_auc']['importances_mean']
        imp_df.sort_values(by='average_precision', ascending=False, inplace=True)
        return imp_df
    
    def eval_model(self, model):
        st.markdown('### Eval on Training Data:')
        self._eval_df(model, self.train_df)
        st.markdown('### Eval on Validation Data:')
        self._eval_df(model, self.val_df)
    
    def predict(self, model, df, threshold):
        df['pred_proba'] = model.predict_proba(df[self.feature_cols])[:, 1]
        df['pred'] = 1 * (df['pred_proba'] > threshold)
        return df

    def _eval_df(self, model, df, threshold=0.5):
        df = self.predict(model, df, threshold)
        try:
            auc = round(roc_auc_score(df[self.target_col], df['pred_proba']), 2)
        except Exception as err:
            auc = None
        ap = round(average_precision_score(df[self.target_col], df['pred_proba']), 2)
        report = classification_report(
            df[self.target_col], 
            1 * (df['pred_proba'] > threshold), 
            target_names=['negative', 'positive']
        )
        st.markdown(
            f"""
```
auc: {auc},

average precision: {ap}, 

random precision: {round(df[self.target_col].mean(), 2)}

Classification Report (threshold={threshold}): \n{report}
```
            """
        )
        return df
    
    def plot_eval_curves(self, model, df):
        df['pred_proba'] = model.predict_proba(df[self.feature_cols])[:, 1]
        col1, col2 = st.columns(2)
        with col1:
            fig1 = self._plot_roc_curve(df)
            st.pyplot(fig1)
            fig2 = self._plot_threshold_recall_curve(df)
            st.pyplot(fig2)
        with col2:
            fig3 = self._plot_precision_recall_curve(df)
            st.pyplot(fig3)
            fig4 = self._plot_threshold_precision_curve(df)
            st.pyplot(fig4)
        return df

    def _plot_roc_curve(self, df):
        auc = round(roc_auc_score(df[self.target_col], df['pred_proba']), 2)
        fpr, tpr, thresholds = roc_curve(df[self.target_col], df['pred_proba'])
        fig = plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC Curve - AUC: {auc}')
        return fig
    
    def _plot_precision_recall_curve(self, df):
        ap = round(average_precision_score(df[self.target_col], df['pred_proba']), 2)
        precision, recall, thresholds = precision_recall_curve(df[self.target_col], df['pred_proba'])
        fig = plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision Recall Curve - Average Precision: {ap}')
        return fig

    def _plot_threshold_recall_curve(self, df):
        precision, recall, thresholds = precision_recall_curve(df[self.target_col], df['pred_proba'])
        fig = plt.figure()
        plt.plot(thresholds, recall[1:])
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.title('Threshold Recall Curve')
        return fig

    def _plot_threshold_precision_curve(self, df):
        precision, recall, thresholds = precision_recall_curve(df[self.target_col], df['pred_proba'])
        fig = plt.figure()
        plt.plot(thresholds, precision[1:])
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.title('Threshold Precision Curve')
        return fig

    
    def save_model(self, model, path):
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    def eda_features_vs_target(self):
        # TODO: some plots only have one bin - need to fix this
        with st.expander('EDA Features vs. Target 📊', expanded=False):
            df_eda = self.train_df.copy()
            for col in self.feature_cols:
                df_eda[f'{col}_bin'] = pd.qcut(df_eda[col], 5, duplicates='drop')
                df_eda[f'{col}_bin'] = df_eda[f'{col}_bin'].astype(str)

                dfg = df_eda.groupby([f'{col}_bin'])[self.target_col].mean().reset_index().rename(columns={self.target_col: 'target_mean'})
                dfg.sort_values(by='target_mean', inplace=True)

                fig = px.bar(dfg, y=f'{col}_bin', x='target_mean', orientation='h', title=f'{col} vs. {self.target_col}')
                st.plotly_chart(fig)

    def qa_fps_and_fns(self, model, df, extra_cols=[]):
        with st.expander('QAing False Positives and False Negatives', expanded=False):
            df['pred_proba'] = model.predict_proba(df[self.feature_cols])[:, 1]
            false_negatives = (
                df[df[self.target_col] == 1]
                .sort_values(by='pred_proba')
                [[self.target_col, 'pred_proba'] + self.feature_cols + extra_cols]
                .head(30)
            )
            st.text("False Negatives:")
            st.dataframe(false_negatives)
            false_positives = (
                df[df[self.target_col] == 0]
                .sort_values(by='pred_proba', ascending=False)
                [[self.target_col, 'pred_proba'] + self.feature_cols + extra_cols]
                .head(30)
            )
            st.text("False Positives:")
            st.dataframe(false_positives)
            return df


df = pd.read_csv('./data/processed/planetary_systems/planets.csv')

with st.expander('View Data 🔎'):
    st.dataframe(df)

training_data = df[~df.life_exists.isna()]
test_df = df[df.life_exists.isna()]
training_data['life_exists'] = training_data['life_exists'].astype(int)
training_data = training_data.fillna(0) # Hint: you may want to handle missing values better than this


model_trainer = TrainModel(training_data, [
    # Hint: you may want to add/remove some of these features and/or engineer some new ones
    'num_stars_in_system', 
    'num_planets_in_system',
    'orbital_period', 
    'planet_radius_vs_earth', 
    'planet_mass_vs_earth',
    'planet_equilibrium_temperature', 
    'distance_to_system_in_light_years',
    'stellar_surface_gravity', 
    'stellar_metallicity'
], 'life_exists')

model_trainer.eda_features_vs_target()

log_reg_model = model_trainer.train_log_reg()
tree_model = model_trainer.train_decision_tree()
rf_model = model_trainer.train_rf()

test_df = test_df.fillna(0) # Hint: do this better
test_df = model_trainer.predict(rf_model, test_df, threshold=0.5)