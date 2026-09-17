import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="NHANES Depressionsanalyse",
    page_icon="🧠",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("NHANES_depression.csv")
    return df


df = load_data()
df_copy = df.copy()


df_copy["BMI_Klasse"] = pd.cut(
    df_copy["BMXBMI"],
    bins=[0, 18.5, 25, 30, 100],
    labels=[
        "Untergewicht",
        "Normal",
        "Übergewicht",
        "Adipositas"
    ]
)

feature_labels = {
    "RIAGENDR": "Geschlecht",
    "RIDAGEYR": "Alter",
    "DMDEDUC2": "Bildungsniveau",
    "DMDMARTZ": "Familienstand",
    "BMXBMI": "BMI",
    "BMI_Klasse": "BMI-Klasse",
    "OCD150": "Art der Arbeit",
    "ASH": "Schlafdauer (Stunden)",
    "MEMPW": "Körperliche Aktivität (Minuten/Woche)",
    "DEPT": "Depressionsschwere"
}

category_labels = {

    "RIAGENDR": {
        1: "Männlich",
        2: "Weiblich"
    },

    "DMDEDUC2": {
        1: "Less than 9th grade",
        2: "9-11th grade (Includes 12th grade with no diploma)",
        3: "High school graduate/GED or equivalent",
        4: "Some college or AA degree",
        5: "College graduate or above",
        9: "Nicht bekannt"
    },

    "DMDMARTZ": {
        1: "Married/Living with partner",
        2: "Widowed/Divorced/Separated",
        3: "Never married",
        99: "Nicht bekannt" 
    },


    "OCD150": {
        1: "Working at a job or business",
        2: "With a job or business but not at work",
        3: "Looking for work",
        4: "Not working at a job or business",
        9: "Nicht bekannt"
    },

    "DEPT": {
        0: "Keine/leichte Symptomatik",
        1: "Behandlungsbedürftig"
    }
}

def get_category_label(value, feature):

    if feature in category_labels:
        return category_labels[feature].get(
            value,
            f"Unbekannt ({value})"
        )

    return str(value)

categorical_features = [
    "RIAGENDR",
    "DMDEDUC2",
    "DMDMARTZ",
    "OCD150",
    "DEPT",
    "BMI_Klasse"
]

numerical_features = [
    "RIDAGEYR",
    "BMXBMI",
    "ASH",
    "MEMPW"
]



st.title("🧠 NHANES – Analyse depressiver Symptomatik")

st.write(
    "Analyse verschiedener soziodemografischer und gesundheitsbezogener Merkmale im Zusammenhang mit depressiver Symptomatik."
)



st.sidebar.header("🔎 Filter")

min_age = int(df_copy["RIDAGEYR"].min())
max_age = int(df_copy["RIDAGEYR"].max())

age_range = st.sidebar.slider(
    "Alter",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

gender_values = sorted(
    df_copy["RIAGENDR"].dropna().unique()
)

gender_options = ["Alle"] + list(gender_values)

selected_gender = st.sidebar.selectbox(
    "Geschlecht",
    gender_options
)

condition = st.sidebar.radio(
    "Gesundheitsstatus",
    ["Alle", "Krank", "Gesund"]
)

filtered_df = df_copy[
    (df_copy["RIDAGEYR"] >= age_range[0]) &
    (df_copy["RIDAGEYR"] <= age_range[1])
].copy()

if selected_gender != "Alle":
    filtered_df = filtered_df[
        filtered_df["RIAGENDR"] == selected_gender
    ]

if condition == "Krank":
    filtered_df = filtered_df[
        filtered_df["DEPT"] == 1
    ]

elif condition == "Gesund":
    filtered_df = filtered_df[
        filtered_df["DEPT"] == 0
    ]




tab1, tab2 = st.tabs([
    "📊 Beschreibung der Strichprobe",
    "⚖️ Krank vs. Gesund"
])



with tab1:

    st.header("📊 Beschreibung der Stichprobe")

    st.write(
        "Hier können einzelne Merkmale der gefilterten Stichprobe untersucht werden."
    )


    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Anzahl Personen",
            len(filtered_df)
        )

    with col2:
        if len(filtered_df) > 0:
            mean_age = filtered_df["RIDAGEYR"].mean()
            st.metric(
                "Durchschnittsalter",
                f"{mean_age:.1f} Jahre"
            )
        else:
            st.metric(
                "Durchschnittsalter",
                "–"
            )

    st.divider()

    show_data = st.checkbox(
        "Rohdaten anzeigen"
    )

    if show_data:
        st.dataframe(
            filtered_df,
            use_container_width=True
        )

    selected_feature = st.selectbox(
        "Merkmal auswählen",
        list(feature_labels.keys()),
        format_func=lambda x: feature_labels[x]
    )

    st.subheader(
        feature_labels[selected_feature]
    )

    if selected_feature in categorical_features:

        plot_df = filtered_df[
            selected_feature
        ].dropna()

        if len(plot_df) == 0:

            st.warning(
                "Für die ausgewählten Filter sind keine Daten vorhanden."
            )

        else:

            counts = plot_df.value_counts().sort_index()


            category_names = [
                get_category_label(
                    value,
                    selected_feature
                )
                for value in counts.index
            ]

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            bars = ax.bar(
                category_names,
                counts.values
            )

            for bar, value in zip(
                bars,
                counts.values
            ):

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(value),
                    ha="center",
                    va="bottom"
                )

            ax.set_xlabel(
                feature_labels[selected_feature]
            )

            ax.set_ylabel(
                "Anzahl Personen"
            )

            ax.set_title(
                f"Verteilung: "
                f"{feature_labels[selected_feature]}"
            )

            plt.xticks(
                rotation=30,
                ha="right"
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            plt.tight_layout()

            st.pyplot(fig)

    elif selected_feature in numerical_features:

        plot_df = filtered_df[
            selected_feature
        ].dropna()

        if len(plot_df) == 0:

            st.warning(
                "Für die ausgewählten Filter sind keine Daten vorhanden."
            )

        else:

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            if selected_feature == "RIDAGEYR":

                min_age_plot = int(plot_df.min())
                max_age_plot = int(plot_df.max())

                ax.hist(
                    plot_df,
                    bins=np.arange(
                        min_age_plot - 0.5,
                        max_age_plot + 1.5,
                        1
                    ),
                    rwidth=0.9
                )

            else:

                ax.hist(
                    plot_df,
                    bins=30
                )

            ax.set_xlabel(
                feature_labels[selected_feature]
            )

            ax.set_ylabel(
                "Anzahl Personen"
            )

            ax.set_title(
                f"Verteilung: "
                f"{feature_labels[selected_feature]}"
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            plt.tight_layout()

            st.pyplot(fig)


with tab2:

    st.header("⚖️ Krank vs. Gesund")

    st.write(
        "Vergleich der Personen mit keiner/leichter Symptomatik und behandlungsbedürftiger Symptomatik."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Anzahl Personen",
            len(filtered_df)
        )

    with col2:
        if len(filtered_df) > 0:
            mean_age = filtered_df["RIDAGEYR"].mean()
            st.metric(
                "Durchschnittsalter",
                f"{mean_age:.1f} Jahre"
            )
        else:
            st.metric(
                "Durchschnittsalter",
                "–"
            )

    comparison_features = [
        "RIDAGEYR",
        "ASH",
        "BMXBMI",
        "BMI_Klasse",
        "MEMPW",
        "RIAGENDR",
        "DMDEDUC2",
        "DMDMARTZ",
        "OCD150"
    ]

    comparison_feature = st.selectbox(
        "Merkmal für den Vergleich auswählen",
        comparison_features,
        format_func=lambda x: feature_labels[x]
    )

    #Filter nach Depressionsschwere nicht verwendet:
    comparison_data = df_copy[
        (df_copy["DEPT"].isin([0, 1])) &
        (df_copy["RIDAGEYR"] >= age_range[0]) &
        (df_copy["RIDAGEYR"] <= age_range[1])
    ].copy()

    if selected_gender != "Alle":

        comparison_data = comparison_data[
            comparison_data["RIAGENDR"] == selected_gender
        ]

    if comparison_feature == "ASH":

        plot_df = comparison_data[
            ["DEPT", "ASH"]
        ].dropna()

        if len(plot_df) == 0:

            st.warning(
                "Für die ausgewählten Filter sind keine Daten vorhanden."
            )

        else:

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            sns.violinplot(
                data=plot_df,
                x="DEPT",
                y="ASH",
                ax=ax,
                order=[0, 1]
            )

            ax.set_xlabel(
                "Depressionsschwere"
            )

            ax.set_ylabel(
                "durchschnittliche Schlafdauer pro Nacht in Stunden"
            )

            ax.set_xticklabels([
                "Keine/leichte Symptomatik",
                "Behandlungsbedürftig"
            ])

            ax.set_title(
                "Durchschnittliche Schlafdauer nach  Schweregrad der depressiven Symptomatik"
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            plt.tight_layout()

            st.pyplot(fig)

            means = (
                plot_df
                .groupby("DEPT")[
                    comparison_feature
                ]
                .mean()
            )

            # Anzahl der Personen je Gruppe
            counts = plot_df.groupby("DEPT").size()


            col1, col2 = st.columns(2)


            with col1:

                if 0 in means.index:

                    st.metric(
                        "Mittelwert - Keine/leichte Symptomatik",
                        f"{means[0]:.2f}"
                    )
                    st.caption(f"n = {counts[0]}")

            with col2:

                if 1 in means.index:

                    st.metric(
                        "Mittelwert - Behandlungsbedürftig",
                        f"{means[1]:.2f}",
                    )
                    st.caption(f"n = {counts[1]}")

    elif comparison_feature == "BMI_Klasse":

        plot_df = comparison_data[
            ["BMI_Klasse", "DEPT"]
        ].dropna()

        if len(plot_df) == 0:

            st.warning(
                "Für die ausgewählten Filter sind keine Daten vorhanden."
            )

        else:

            sick_percentage = (
                plot_df
                .groupby(
                    "BMI_Klasse",
                    observed=True
                )["DEPT"]
                .mean()
                * 100
            )

            category_counts = (
                plot_df
                .groupby("BMI_Klasse", observed=True)
                .size()
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            bars = ax.bar(
                sick_percentage.index.astype(str),
                sick_percentage.values
            )
       
            # Prozentwert und absolute Anzahl über den Balken anzeigen
            for bar, percentage, category in zip(
                bars,
                sick_percentage.values,
                sick_percentage.index
            ):
            
                count = category_counts[category]
            
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{percentage:.1f}% (n={count})",
                    ha="center",
                    va="bottom"
                )

            ax.set_xlabel(
                "BMI-Klasse"
            )

            ax.set_ylabel(
                "Anteil Kranker (%)"
            )

            ax.set_title(
                "Anteil der behandlungsbedürftigen Personen nach BMI-Klasse"
            )

            ax.set_ylim(
                0,
                max(sick_percentage.values) * 1.15
                if len(sick_percentage) > 0
                else 100
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            plt.tight_layout()

            st.pyplot(fig)

            st.info(
                "Die Prozentwerte zeigen innerhalb jeder BMI-Klasse, wie viel Prozent der Personen behandlungsbedürftige depressive Symptomatik aufweisen."
            )

    elif comparison_feature in [
        "RIDAGEYR",
        "BMXBMI",
        "MEMPW"
    ]:

        plot_df = comparison_data[
            ["DEPT", comparison_feature]
        ].dropna()

        if len(plot_df) == 0:

            st.warning(
                "Für die ausgewählten Filter sind keine Daten vorhanden."
            )

        else:

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            sns.boxplot(
                data=plot_df,
                x="DEPT",
                y=comparison_feature,
                ax=ax,
                order=[0, 1]
            )

            ax.set_xlabel(
                "Depressionsschwere"
            )

            ax.set_ylabel(
                feature_labels[comparison_feature]
            )

            ax.set_xticklabels([
                "Keine/leichte Symptomatik",
                "Behandlungsbedürftig"
            ])

            ax.set_title(
                f"{feature_labels[comparison_feature]} "
                "nach Depressionsschwere"
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            plt.tight_layout()

            st.pyplot(fig)
            
            means = (
                plot_df
                .groupby("DEPT")[
                    comparison_feature
                ]
                .mean()
            )

            # Anzahl der Personen je Gruppe
            counts = plot_df.groupby("DEPT").size()

            col1, col2 = st.columns(2)

            with col1:

                if 0 in means.index:

                    st.metric(
                        "Mittelwert - Keine/leichte Symptomatik",
                        f"{means[0]:.2f}"
                    )
                    st.caption(f"n = {counts[0]}")

            with col2:

                if 1 in means.index:

                    st.metric(
                        "Mittelwert - Behandlungsbedürftig",
                        f"{means[1]:.2f}",
                    )
                    st.caption(f"n = {counts[1]}")

    elif comparison_feature in [
        "RIAGENDR",
        "DMDEDUC2",
        "DMDMARTZ",
        "OCD150"
    ]:

        plot_df = comparison_data[
            [comparison_feature, "DEPT"]
        ].dropna()

        if len(plot_df) == 0:

            st.warning(
                "Für die ausgewählten Filter sind keine "
                "Daten vorhanden."
            )

        else:
 
            percentage_sick = (
                plot_df
                .groupby(
                    comparison_feature,
                    observed=True
                )["DEPT"]
                .mean()
                * 100
            )

            # Absolute Anzahl der Personen je Kategorie
            category_counts = (
                plot_df
                .groupby(comparison_feature, observed=True)
                .size()
            )
 
            category_names = [
                get_category_label(
                    value,
                    comparison_feature
                )
                for value in percentage_sick.index
            ]

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            bars = ax.bar(
                category_names,
                percentage_sick.values
            )

            # Prozentwert und absolute Anzahl über den Balken anzeigen
            for bar, percentage, category in zip(
                bars,
                percentage_sick.values,
                percentage_sick.index
            ):
            
                count = category_counts[category]
            
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{percentage:.1f}% (n={count})",
                    ha="center",
                    va="bottom"
                )

            ax.set_xlabel(
                feature_labels[comparison_feature]
            )

            ax.set_ylabel(
                "Anteil Kranker (%)"
            )

            ax.set_title(
                f"Anteil der Kranken innerhalb "
                f"der Kategorien von "
                f"{feature_labels[comparison_feature]}"
            )

            ax.set_ylim(
                0,
                max(percentage_sick.values) * 1.15
                if len(percentage_sick) > 0
                else 100
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            plt.xticks(
                rotation=30,
                ha="right"
            )

            plt.tight_layout()

            st.pyplot(fig)

            st.info(
                "Die Prozentwerte zeigen, wie viel Prozent der Personen innerhalb jeder Kategorie behandlungsbedürftige depressive Symptomatik aufweisen."
            )