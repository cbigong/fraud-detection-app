import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🔒",
    layout="centered"
)

# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================
@st.cache_resource
def load_model():
    """Charge le modèle une seule fois (mise en cache)"""
    return joblib.load("fraud_detection_model_improved.pkl")

model = load_model()

# ============================================================================
# FONCTION DE CRÉATION DES FEATURES ET DICTIONNAIRE POUR LA TRADUCTION
# ============================================================================
dic_type ={"PAIEMENT" : "PAYMENT",
      "TRANSFERT" : "TRANSFER",
      "RETRAIT" : "CASH_OUT",
      "DEPOT" : "CASH_IN"}

def create_advanced_features(df):
    """
    Crée les features avancées à partir des données de base
    """
    df = df.copy()
    
    # Différences de balance
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Ratios
    df['amount_to_oldbalance_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_oldbalance_dest'] = df['amount'] / (df['oldbalanceDest'] + 1)
    
    # Incohérences
    df['error_balance_orig'] = df['balance_diff_orig'] - df['amount']
    df['error_balance_dest'] = df['balance_diff_dest'] - df['amount']
    
    # Indicateurs binaires
    df['is_zero_balance_orig'] = (df['newbalanceOrig'] == 0).astype(int)
    df['is_zero_balance_dest'] = (df['oldbalanceDest'] == 0).astype(int)
    df['large_transaction'] = (df['amount'] > 200000).astype(int)
    
    # Features logarithmiques
    df['log_amount'] = np.log1p(df['amount'])
    df['log_oldbalance_orig'] = np.log1p(df['oldbalanceOrg'])
    df['log_oldbalance_dest'] = np.log1p(df['oldbalanceDest'])
    
    return df

# ============================================================================
# INTERFACE UTILISATEUR
# ============================================================================

# En-tête
st.title("🔒 Détection de Fraude Bancaire")
st.markdown("""
    Cette application utilise le **Machine Learning** pour détecter les transactions frauduleuses en temps réel.
    
    **Entrez les détails de la transaction ci-dessous et cliquez sur "Analyser".**
""")

st.divider()

# Formulaire de saisie
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Informations de la transaction")
    
    transaction_type = st.selectbox(
        "Type de transaction",
        ["PAIEMENT", "TRANSFERT", "RETRAIT", "DEPOT"],
        help="Sélectionnez le type de transaction"
    )
    
    amount = st.number_input(
        "Montant (€)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        help="Montant de la transaction en euros"
    )

with col2:
    st.subheader("💳 Soldes - Expéditeur")
    
    # Ajout d'un espace pour l'alignement
    # st.markdown("&nbsp;", unsafe_allow_html=True)
    oldbalanceOrg = st.number_input(
        "Solde avant (€)",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
        key="old_orig"
    )
    
    newbalanceOrig = st.number_input(
        "Solde après (€)",
        min_value=0.0,
        value=9000.0,
        step=1000.0,
        key="new_orig"
    )

st.subheader("🏦 Soldes - Destinataire")

col3, col4 = st.columns(2)

with col3:
    oldbalanceDest = st.number_input(
        "Solde avant (€)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        key="old_dest"
    )

with col4:
    newbalanceDest = st.number_input(
        "Solde après (€)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        key="new_dest"
    )

st.divider()

# ============================================================================
# BOUTON DE PRÉDICTION
# ============================================================================

if st.button("🚀 Analyser la Transaction", type="primary", use_container_width=True):
    
    # Création du DataFrame d'entrée
    input_data = pd.DataFrame({
        "type": [dic_type[transaction_type]],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })
    
    # Création des features avancées
    input_data_enhanced = create_advanced_features(input_data)
    
    # Prédiction
    prediction = model.predict(input_data_enhanced)[0]
    probability = model.predict_proba(input_data_enhanced)[0, 1]
    
    # Affichage des résultats
    st.divider()
    st.subheader("📊 Résultat de l'Analyse")
    
    # Métrique de probabilité
    st.metric(
        label="Probabilité de Fraude",
        value=f"{probability * 100:.1f}%",
        # delta=f"{(probability - 0.5) * 100:.1f}% vs seuil"
    )
    
    # Résultat avec couleur
    if prediction == 1:
        st.error("⚠️ **ALERTE : Cette transaction est prédite comme FRAUDULEUSE**")
        st.warning("""
            **Actions recommandées :**
            - Bloquer temporairement la transaction
            - Contacter le client pour vérification
            - Lancer une investigation
        """)
    else:
        st.success("✅ **Cette transaction est prédite comme LÉGITIME**")
        st.info("La transaction peut être autorisée.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem 0;'>
        <p>🔒 <strong>Application de Détection de Fraude Bancaire</strong></p>
        <p>Développé avec Machine Learning | Random Forest + SMOTE</p>
        <p style='font-size: 0.9em; margin-top: 1rem;'>
            Modèle entraîné sur 6.3M+ transactions | AUC-ROC: 95.4% | F1-Score: 85.2%
        </p>
    </div>
""", unsafe_allow_html=True)
