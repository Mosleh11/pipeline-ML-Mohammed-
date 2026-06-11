"""WebApp Streamlit : prediction de tumeur benigne / maligne.

Lancement :
    pip install streamlit
    streamlit run app.py

Pour garder l'interface simple, on ne saisit que quelques mesures cles ; les autres
features sont remplies avec la moyenne du jeu d'entrainement. On affiche un avertissement
quand une valeur sort de la plage vue a l'entrainement (un modele ne sait pas dire
"je ne sais pas", c'est a nous de le garder).

Auteur : Mohammed MOSLEH
"""
import numpy as np
import joblib
import streamlit as st

BUNDLE = joblib.load("modele.joblib")
FEATURES = BUNDLE["features"]
PLAGES = BUNDLE["plages"]

# Valeur par defaut de chaque feature = milieu de la plage d'entrainement
DEFAUTS = (PLAGES["min"] + PLAGES["max"]) / 2

# Quelques mesures parlantes a exposer a l'utilisateur (les plus discriminantes)
CLES = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean concavity"]

st.title("Detection de tumeur : benigne ou maligne ?")
st.write("Saisissez quelques mesures ; le modele estime la nature de la tumeur.")

valeurs = DEFAUTS.copy()
for nom in CLES:
    i = FEATURES.index(nom)
    valeurs[i] = st.number_input(
        nom, value=float(round(DEFAUTS[i], 2)),
        min_value=0.0, max_value=float(PLAGES["max"][i] * 3),
    )

if st.button("Predire"):
    x = valeurs.reshape(1, -1)
    x_s = BUNDLE["scaler"].transform(x)
    pred = int(BUNDLE["modele"].predict(x_s)[0])
    proba = float(BUNDLE["modele"].predict_proba(x_s)[0].max())
    label = "benigne" if pred == 1 else "maligne"

    if label == "benigne":
        st.success(f"Prediction : tumeur **benigne** (probabilite {proba:.0%})")
    else:
        st.error(f"Prediction : tumeur **maligne** (probabilite {proba:.0%})")
    st.progress(proba)

    # Avertissement entrees hors plage
    hors = int(((x[0] < PLAGES["min"]) | (x[0] > PLAGES["max"])).sum())
    if hors:
        st.warning(f"Attention : {hors} valeur(s) hors de la plage vue a l'entrainement. "
                   "La prediction est moins fiable.")
