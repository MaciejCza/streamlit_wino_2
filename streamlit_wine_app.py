# streamlit_wine_app.py
# Aplikacja Streamlit do zaawansowanej analityki danych wina,
# wizualizacji oraz predykcji jakości (model: GradientBoostingRegressor).

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------
st.set_page_config(page_title="Wine Analytics & Food Pairings (Advanced)", layout="wide")
st.title("🍷 Advanced Wine Analytics & Food Pairings")
st.markdown("Aplikacja wykonana na podstawie plików `winequality-red.csv` i `wine_food_pairings.csv`.\n\n" 
            "Funkcjonalności: eksploracja danych, zaawansowana analityka (PCA, klasteryzacja), co najmniej 3 wizualizacje oraz predykcja jakości wina "
            "przy użyciu modelu GradientBoostingRegressor.")

# ---------------------------------------------------------
# Wczytywanie danych
# ---------------------------------------------------------
@st.cache_data
def load_wine_quality(path: str = "winequality-red.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_pairings(path: str = "wine_food_pairings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# Load
try:
    wine_df = load_wine_quality()
except Exception as e:
    st.error(f"Błąd wczytywania winequality-red.csv: {e}")
    st.stop()

try:
    pairings_df = load_pairings()
except Exception as e:
    st.warning(f"Nie udało się wczytać wine_food_pairings.csv: {e}. Moduł parowań będzie niedostępny.")
    pairings_df = None

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("⚙️ Ustawienia aplikacji")
module = st.sidebar.radio("Wybierz moduł:", ["Analiza jakości wina", "Parowanie wina z jedzeniem"]) 

# =========================================================
# 1. Analiza jakości wina
# =========================================================
if module == "Analiza jakości wina":
    st.subheader("📊 Eksploracja i zaawansowana analityka jakości wina")

    df = wine_df.copy()

    # Podstawowy podgląd
    st.markdown("### Podstawowy podgląd datasetu")
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.write("Kształt:", df.shape)
        st.write(df.dtypes)
    with col2:
        st.write("Opis statystyczny:")
        st.dataframe(df.describe().T)

    # Wizualizacja 1: Histogram jakości
    st.markdown("### 1) Rozkład ocen jakości (histogram)")
    fig1, ax1 = plt.subplots()
    ax1.hist(df['quality'], bins=range(int(df['quality'].min()), int(df['quality'].max())+2), edgecolor='black')
    ax1.set_xlabel('quality')
    ax1.set_ylabel('count')
    st.pyplot(fig1)

    # Wizualizacja 2: Korelacje (heatmap)
    st.markdown("### 2) Macierz korelacji cech (heatmap)")
    corr = df.corr(numeric_only=True)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=False, cmap='vlag', ax=ax2)
    st.pyplot(fig2)

    # PCA: redukcja wymiarów i wykres
    st.markdown("### 3) PCA na cechach fizykochemicznych + wykres 2D")
    features = [c for c in df.columns if c != 'quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1','PC2'])
    pca_df['quality'] = df['quality'].values

    fig3, ax3 = plt.subplots()
    scatter = ax3.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['quality'], cmap='viridis', alpha=0.7)
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('PCA: PC1 vs PC2 (kolor = quality)')
    cbar = fig3.colorbar(scatter, ax=ax3)
    cbar.set_label('quality')
    st.pyplot(fig3)

    # Klasteryzacja KMeans na PCA
    st.markdown("### Klasteryzacja KMeans na zredukowanych danych (PCA)")
    k = st.sidebar.slider('Liczba klastrów KMeans', min_value=2, max_value=8, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    pca_df['cluster'] = clusters
    st.write(pca_df.groupby('cluster')['quality'].agg(['count','mean','std']))

    fig4, ax4 = plt.subplots()
    for cl in sorted(pca_df['cluster'].unique()):
        sub = pca_df[pca_df['cluster']==cl]
        ax4.scatter(sub['PC1'], sub['PC2'], label=f'cluster {cl}', alpha=0.6)
    ax4.legend()
    ax4.set_title('KMeans clusters on PCA space')
    st.pyplot(fig4)

    # Scatter interaktywny: cecha vs quality
    st.markdown('### Interaktywny scatter: cecha vs quality')
    feature_cols = features
    x_feature = st.selectbox('Wybierz cechę (oś X):', feature_cols, index=feature_cols.index('alcohol') if 'alcohol' in feature_cols else 0)
    fig5, ax5 = plt.subplots()
    ax5.scatter(df[x_feature], df['quality'], alpha=0.6)
    ax5.set_xlabel(x_feature)
    ax5.set_ylabel('quality')
    st.pyplot(fig5)

    # Boxplot dla cech względem quality (przydatne do wykrywania trendów)
    st.markdown('### Boxplot: alcohol w zależności od jakości (quality)')
    fig6, ax6 = plt.subplots(figsize=(8,4))
    sns.boxplot(x='quality', y='alcohol', data=df, ax=ax6)
    st.pyplot(fig6)

    # -------------------------
    # Model predykcyjny - GradientBoostingRegressor
    # -------------------------
    st.markdown('### 🤖 Model regresyjny: predykcja jakości (GradientBoostingRegressor)')

    with st.expander('Ustawienia modelu i walidacja'):
        test_size = st.slider('Udział danych testowych', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        n_estimators = st.slider('Liczba drzew (n_estimators)', min_value=50, max_value=1000, value=200, step=50)
        learning_rate = st.number_input('Learning rate', min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
        max_depth = st.slider('Max depth', min_value=1, max_value=10, value=3)
        random_state = st.number_input('Random state', min_value=0, value=42, step=1)

    X = df[features]
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

    model = GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=float(learning_rate), max_depth=int(max_depth), random_state=int(random_state))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.metric('R² (test)', f'{r2:.3f}')
    st.metric('MAE (test)', f'{mae:.3f}')

    # Permutation importance (bardziej odporny wskaźnik ważności cech)
    st.markdown('**Ważność cech — permutation importance (na zbiorze testowym)**')
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)
    perm_importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    st.bar_chart(perm_importances)

    # Krótkie cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    st.write('CV R² (5-fold):', np.round(cv_scores,3))

    # Interaktywna predykcja
    st.markdown('### 🔮 Predykcja jakości dla niestandardowych parametrów')
    with st.form('prediction_form'):
        cols = st.columns(3)
        user_input = {}
        for i, col_name in enumerate(features):
            col = cols[i % 3]
            min_val = float(df[col_name].min())
            max_val = float(df[col_name].max())
            mean_val = float(df[col_name].mean())
            step = (max_val - min_val) / 200 if max_val > min_val else 0.01
            user_input[col_name] = col.slider(col_name, min_value=min_val, max_value=max_val, value=mean_val, step=step)
        submitted = st.form_submit_button('Oblicz przewidywaną jakość')

    if submitted:
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        st.success(f'Przewidywana jakość: {pred:.2f}')

# =========================================================
# 2. Parowanie wina z jedzeniem
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    st.subheader('🍽️ Parowanie wina z jedzeniem — eksploracja')
    if pairings_df is None:
        st.error('Brak danych o parowaniach — załaduj wine_food_pairings.csv w katalogu aplikacji.')
        st.stop()

    dfp = pairings_df.copy()
    st.dataframe(dfp.head())

    with st.expander('Podstawowe informacje o zbiorze parowań'):
        st.write('Kształt:', dfp.shape)
        st.write('Kolumny:', list(dfp.columns))
        if 'pairing_quality' in dfp.columns:
            st.write('Statystyki pairing_quality:')
            st.write(dfp['pairing_quality'].describe())

    # Rozkład ocen parowań
    if 'pairing_quality' in dfp.columns:
        st.markdown('### Rozkład ocen parowań')
        figp, axp = plt.subplots()
        axp.hist(dfp['pairing_quality'].dropna(), bins=20, edgecolor='black')
        axp.set_xlabel('pairing_quality')
        axp.set_ylabel('count')
        st.pyplot(figp)

    # Najlepsze parowania dla wybranego typu wina
    st.markdown('### Najlepsze rekomendacje dla wybranego typu wina')
    if 'wine_type' in dfp.columns:
        wine_types = sorted(dfp['wine_type'].dropna().unique())
        chosen = st.selectbox('Wybierz wine_type', options=['(wszystkie)'] + wine_types)
        sub = dfp.copy()
        if chosen != '(wszystkie)':
            sub = sub[sub['wine_type']==chosen]
        if 'pairing_quality' in sub.columns:
            st.dataframe(sub.sort_values('pairing_quality', ascending=False).head(30))
        else:
            st.dataframe(sub.head(30))

    # Prosta rekomendacja: wyszukaj po nazwie potrawy
    st.markdown('### Szukaj rekomendacji po nazwie potrawy')
    search_food = st.text_input('Wpisz fragment nazwy potrawy (food_item)')
    if search_food:
        res = dfp[dfp['food_item'].str.contains(search_food, case=False, na=False)]
        if res.empty:
            st.warning('Brak wyników')
        else:
            st.dataframe(res.head(50))

    # Podsumowanie: średnie pairing_quality per wine_type
    if 'pairing_quality' in dfp.columns and 'wine_type' in dfp.columns:
        st.markdown('### Średnia ocena parowania per wine_type')
        mean_by_wine = dfp.groupby('wine_type')['pairing_quality'].mean().sort_values(ascending=False)
        st.bar_chart(mean_by_wine)

# ---------------------------------------------------------
# Stopka
# ---------------------------------------------------------
st.sidebar.markdown('---')
st.sidebar.write('Dependencies: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn')
st.sidebar.write('Uruchom: `streamlit run streamlit_wine_app.py` (umieść oba CSV w tym samym katalogu)')
