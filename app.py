"""
Application Streamlit — Prédiction de Retards de Vols
=====================================================
Structure attendue (tout dans le MÊME dossier) :
  mon_dossier/
  ├── app.py                    ← ce fichier
  ├── regression_logistique.pkl
  ├── random_forest.pkl
  ├── xgboost.pkl
  ├── lightgbm.pkl
  ├── le_dep.pkl
  ├── le_arr.pkl
  ├── le_comp.pkl
  ├── feature_cols.pkl
  ├── results_table.json
  ├── stats.json
  └── (images .png optionnelles)

Lancement :
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os

# ─── Chemin BASE = dossier où se trouve app.py ────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def p(filename):
    """Retourne le chemin absolu d'un fichier dans le même dossier que app.py."""
    return os.path.join(BASE, filename)

# ─── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" Prédiction Retards de Vols",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Chargement des ressources ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_files = {
        'Régression Logistique': 'regression_logistique.pkl',
        'Random Forest':         'random_forest.pkl',
        'XGBoost':               'xgboost.pkl',
        'LightGBM':              'lightgbm.pkl',
    }
    loaded = {}
    for name, fname in model_files.items():
        path = p(fname)
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
        else:
            st.error(f" Fichier introuvable : {path}")
    return loaded

@st.cache_resource
def load_encoders():
    return (
        joblib.load(p('le_dep.pkl')),
        joblib.load(p('le_arr.pkl')),
        joblib.load(p('le_comp.pkl')),
        joblib.load(p('feature_cols.pkl')),
    )

@st.cache_data
def load_stats():
    path = p('stats.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        'n_total': 490011, 'n_retardes': 93262, 'taux_retard': 19.03,
        'retard_moyen': 4.92, 'retard_median': -5.0, 'retard_max': 1554,
        'n_aeroports_dep': 318, 'n_compagnies': 12, 'n_avions': 4106,
    }

@st.cache_data
def load_results():
    path = p('results_table.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def show_image(filename, caption=None, use_container_width=True):
    """Affiche une image si elle existe, sinon un message discret."""
    path = p(filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=use_container_width)
    else:
        st.info(f" Image non trouvée : `{filename}` — placez-la dans le même dossier que app.py")

# ─── Chargement ───────────────────────────────────────────────────────────────
models       = load_models()
le_dep, le_arr, le_comp, feature_cols = load_encoders()
stats        = load_stats()
results      = load_results()
model_names  = list(models.keys())
COLORS       = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ✈️ Flight Delay Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [" Accueil", " Prédiction", " Performance Modèles", " Analyse EDA"],
)
st.sidebar.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════════
if page == " Accueil":
    st.title(" Système de Prédiction des Retards de Vols")
    st.markdown("### Plateforme d'analyse et de prédiction des retards aériens par Machine Learning")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vols analysés",       f"{stats['n_total']:,}")
    c2.metric("Vols retardés ≥15min",f"{stats['n_retardes']:,}")
    c3.metric("Taux de retard",       f"{stats['taux_retard']:.1f}%")
    c4.metric("Compagnies",           str(stats['n_compagnies']))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Description du Projet")
        st.markdown("""
Un vol est considéré **retardé** si son retard à l'arrivée est **≥ 15 minutes** (norme IATA/FAA).

**4 modèles comparés :**
- 🔵 Régression Logistique *(baseline)*
- 🟢 Random Forest
- 🟠 XGBoost  *(meilleur)*
- 🔴 LightGBM

**Variable cible :** classification binaire (retardé / non retardé)
        """)

    with col2:
        st.subheader(" Statistiques du Dataset")
        df_stats = pd.DataFrame({
            'Indicateur': [
                'Retard moyen', 'Retard médian', 'Retard maximum',
                'Aéroports de départ', 'Types d\'avions',
            ],
            'Valeur': [
                f"{stats['retard_moyen']:.1f} min",
                f"{stats['retard_median']} min",
                f"{stats['retard_max']:.0f} min",
                str(stats['n_aeroports_dep']),
                str(stats['n_avions']),
            ],
        })
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(" Meilleur Modèle : XGBoost")
    if results and 'XGBoost' in results:
        m = results['XGBoost']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{m.get('Accuracy', m.get('accuracy', 'N/A')):.4f}")
        c2.metric("F1-Score",  f"{m.get('F1-Score', m.get('f1', 'N/A')):.4f}")
        c3.metric("ROC-AUC",   f"{m.get('ROC-AUC',  m.get('roc_auc', 'N/A')):.4f}")
        c4.metric("Rappel",    f"{m.get('Rappel',   m.get('recall', 'N/A')):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Prédiction":
    st.title(" Prédiction de Retard de Vol")
    st.markdown("Renseignez les informations du vol pour obtenir une prédiction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(" Départ")
        aeroport_dep = st.selectbox(
            "Aéroport de départ",
            sorted([c for c in le_dep.classes_ if c != 'nan']),
        )
        depart_programme = st.slider(
            "Heure de départ programmée (HHMM)", 0, 2359, 1200, step=5,
        )
        mois = st.selectbox(
            "Mois du vol", list(range(1, 13)),
            format_func=lambda x: ['Janvier','Février','Mars','Avril','Mai','Juin',
                                    'Juillet','Août','Septembre','Octobre','Novembre','Décembre'][x-1],
        )
        jour_semaine = st.selectbox(
            "Jour de la semaine", list(range(7)),
            format_func=lambda x: ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche'][x],
        )

    with col2:
        st.subheader(" Trajet")
        aeroport_arr = st.selectbox(
            "Aéroport d'arrivée",
            sorted([c for c in le_arr.classes_ if c != 'nan']),
        )
        distance = st.number_input(
            "Distance (km)", min_value=50, max_value=15000, value=1000, step=50,
        )
        temps_programme = st.number_input(
            "Durée de vol programmée (min)", min_value=30, max_value=900, value=150,
        )

    with col3:
        st.subheader(" Vol")
        compagnie = st.selectbox(
            "Compagnie aérienne",
            sorted([c for c in le_comp.classes_ if c != 'nan']),
        )
        niveau_securite = st.slider("Niveau de sécurité", 1, 10, 10)
        modele_choisi   = st.selectbox("Modèle de prédiction", model_names, index=2)

    st.markdown("---")
    if st.button(" Prédire le retard", type="primary", use_container_width=True):

        heure_dep = depart_programme // 100

        # Encodage avec gestion des valeurs inconnues
        try:    dep_enc  = le_dep.transform([aeroport_dep])[0]
        except: dep_enc  = 0
        try:    arr_enc  = le_arr.transform([aeroport_arr])[0]
        except: arr_enc  = 0
        try:    comp_enc = le_comp.transform([compagnie])[0]
        except: comp_enc = 0

        X_input = pd.DataFrame([[
            depart_programme, heure_dep, mois, jour_semaine,
            distance, temps_programme, niveau_securite,
            dep_enc, arr_enc, comp_enc,
        ]], columns=feature_cols)

        model = models[modele_choisi]
        pred  = model.predict(X_input)[0]
        prob  = model.predict_proba(X_input)[0][1]

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if pred == 1:
                st.error(f"### ⚠️ Vol probablement RETARDÉ")
            else:
                st.success(f"### ✅ Vol probablement À L'HEURE")

            st.markdown(f"**Probabilité de retard :** `{prob*100:.1f}%`")
            st.markdown(f"**Modèle utilisé :** {modele_choisi}")
            st.progress(float(prob))

        with res_col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(['À l\'heure', 'Retardé'], [1-prob, prob],
                   color=['#4CAF50', '#F44336'], alpha=0.85, edgecolor='white')
            ax.set_ylim(0, 1); ax.set_ylabel('Probabilité')
            ax.set_title(f'Probabilités — {modele_choisi}', fontweight='bold')
            for x, val in enumerate([1-prob, prob]):
                ax.text(x, val + 0.02, f'{val*100:.1f}%', ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Comparaison tous modèles
        st.markdown("---")
        st.subheader("🔄 Comparaison de tous les modèles")
        rows = []
        for mname, mobj in models.items():
            pp   = mobj.predict_proba(X_input)[0][1]
            rows.append({
                'Modèle':             mname,
                'Prédiction':         '⚠️ Retardé' if pp >= 0.5 else '✅ À l\'heure',
                'Probabilité Retard': f'{pp*100:.1f}%',
                'Confiance':          f'{max(pp, 1-pp)*100:.1f}%',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Performance Modèles":
    st.title(" Performance des Modèles ML")

    if results:
        st.subheader(" Tableau Comparatif")

        # Normalise les clés (majuscules ou minuscules selon la source)
        key_map = {
            'accuracy':  'Accuracy',
            'f1':        'F1-Score',
            'precision': 'Précision',
            'recall':    'Rappel',
            'roc_auc':   'ROC-AUC',
            'avg_prec':  'Avg Precision',
            'Accuracy':  'Accuracy',
            'F1-Score':  'F1-Score',
            'Précision': 'Précision',
            'Rappel':    'Rappel',
            'ROC-AUC':   'ROC-AUC',
            'Avg Precision': 'Avg Precision',
        }
        normalized = {}
        for model_name, metrics in results.items():
            normalized[model_name] = {key_map.get(k, k): v for k, v in metrics.items()}

        df_res = pd.DataFrame(normalized).T.reset_index().rename(columns={'index': 'Modèle'})

        def highlight_max(s):
            return ['background-color:#c8e6c9;font-weight:bold'
                    if v == s.max() else '' for v in s]

        num_cols = [c for c in df_res.columns if c != 'Modèle']
        st.dataframe(
            df_res.style.apply(highlight_max, subset=num_cols),
            use_container_width=True, hide_index=True,
        )
    else:
        st.warning("Fichier `results_table.json` non trouvé dans le dossier.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(" Métriques comparées")
        show_image('model_comparison.png')
    with c2:
        st.subheader(" Courbes ROC")
        show_image('roc_curves.png')

    st.markdown("---")
    st.subheader(" Matrices de Confusion")
    show_image('confusion_matrices.png')

    st.markdown("---")
    st.subheader(" Importance des Variables")
    show_image('feature_importance.png')

    st.markdown("---")
    st.subheader(" Interprétation")
    st.info("""
**XGBoost** obtient le meilleur ROC-AUC (0.664) et la meilleure Average Precision (0.315).

**LightGBM** est quasi-équivalent mais s'entraîne plus rapidement sur de grands volumes.

**Variables les plus prédictives :** distance du vol, durée programmée, heure de départ,
aéroport d'origine et compagnie aérienne.

**Déséquilibre de classes :** 19% seulement des vols sont retardés → les modèles
utilisent la pondération des classes pour compenser.
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Analyse EDA":
    st.title(" Analyse Exploratoire des Données")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vols analysés",   f"{stats['n_total']:,}")
    c2.metric("Taux de retard",  f"{stats['taux_retard']:.1f}%")
    c3.metric("Aéroports",       str(stats['n_aeroports_dep']))
    c4.metric("Compagnies",      str(stats['n_compagnies']))

    st.markdown("---")
    show_image('eda_analysis.png')

    st.markdown("---")
    st.subheader(" Observations Clés")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Tendances temporelles :**
- Les mois d'été (juin–août) présentent plus de retards
- Les vols du soir sont les plus touchés (effet cascade)
- Le vendredi et le dimanche ont les taux les plus élevés
        """)
    with c2:
        st.markdown("""
**Tendances opérationnelles :**
- Certains aéroports hub concentrent les retards
- La compagnie aérienne est un facteur discriminant
- Les longs-courriers accumulent plus facilement des retards
        """)

    st.markdown("---")
    st.subheader(" Statistiques Descriptives")
    df_desc = pd.DataFrame({
        'Métrique':          ['Retard moyen', 'Retard médian', 'Retard maximum'],
        'Valeur (minutes)':  [stats['retard_moyen'], stats['retard_median'], stats['retard_max']],
    })
    st.dataframe(df_desc, use_container_width=True, hide_index=True)

    st.info("""
**Définition officielle :** Un vol est retardé si son retard à l'arrivée est ≥ 15 minutes
(norme FAA / IATA). Les vols annulés et détournés ont été exclus de l'analyse.
    """)
