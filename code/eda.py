import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from itertools import combinations

st.set_page_config(page_title="Predictive Maintenance App", layout="wide")
st.title("Predictive Maintenance App")

# Sidebar navigation
page = st.sidebar.radio("Select Page", ["EDA", "Model Prediction"])

# --- EDA Page ---
if page == "EDA":
    st.header("Exploratory Data Analysis")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview", df.head())
        st.write("### Descriptive Statistics")
        st.dataframe(df.describe(include='all').transpose())

        selected_columns = st.sidebar.multiselect("Select features to analyze", df.columns)
        categorical_features = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 10]
        target_feature = st.sidebar.selectbox("Select a target feature (for coloring plots)", categorical_features)

        if selected_columns:
            continuous_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10]
            discrete_cols = [col for col in selected_columns if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 10]

            if continuous_cols:
                st.write("## 📈 Continuous Feature Analysis")
                st.write("### Line Plots for Each Continuous Feature")
                for col in continuous_cols:
                    st.write(f"**Line Plot for {col}**")
                    fig, ax = plt.subplots()
                    ax.plot(df.index, df[col], label=col)
                    ax.set_xlabel("Index")
                    ax.set_ylabel(col)
                    ax.set_title(f"{col} over Index")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                st.write("### Box Plot")
                melted_df = df[continuous_cols].melt(var_name="Feature", value_name="Value")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x="Feature", y="Value", data=melted_df, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close(fig)

                if len(continuous_cols) >= 2:
                    st.write("### Correlation Heatmap")
                    fig, ax = plt.subplots()
                    sns.heatmap(df[continuous_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.write(f"### Scatter Plots (Colored by '{target_feature}')")
                    for x_col, y_col in combinations(continuous_cols, 2):
                        st.write(f"**{x_col} vs {y_col}**")
                        fig, ax = plt.subplots()
                        sns.scatterplot(
                            x=x_col,
                            y=y_col,
                            hue=df[target_feature],
                            data=df,
                            ax=ax,
                            palette="tab10"
                        )
                        st.pyplot(fig)
                        plt.close(fig)

            if discrete_cols:
                st.write("## 🧩 Discrete Feature Analysis")
                for col in discrete_cols:
                    st.write(f"**Distribution of {col}**")
                    if df[col].nunique() <= 5:
                        value_counts = df[col].value_counts()
                        fig, ax = plt.subplots(figsize=(6, 6))
                        colors = sns.color_palette("pastel")[0:len(value_counts)]
                        wedges, texts, autotexts = ax.pie(
                            value_counts,
                            labels=value_counts.index,
                            autopct='%1.1f%%',
                            startangle=140,
                            colors=colors,
                            textprops={'fontsize': 10}
                        )
                        ax.axis('equal')
                        plt.setp(autotexts, size=10, weight="bold")
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        fig, ax = plt.subplots()
                        sns.countplot(x=col, data=df, ax=ax)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close(fig)
        else:
            st.info("Please select at least one feature from the sidebar to begin visual analysis.")
    else:
        st.warning("Please upload a CSV file to begin.")

# --- Model Prediction Page ---
elif page == "Model Prediction":
    st.header("Model Prediction")

    model_file = st.file_uploader("Upload a trained model (.pkl)", type=["pkl"])
    data_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if model_file and data_file:
        try:
            model = pickle.load(model_file)
            data = pd.read_csv(data_file)
            data["Type"] = data["Type"].replace({"L": 0, "M": 1, "H": 2})
            st.write("### Input Data", data.head())

            predictions = model.predict(data)
            st.write("### Predictions")
            st.dataframe(pd.DataFrame(predictions, columns=["Prediction"]))
        except Exception as e:
            st.error(f"Error loading model or making predictions: {e}")
    else:
        st.info("Please upload both a model file and a dataset to generate predictions.")
