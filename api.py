"""API minimale Flask pour servir le modele de prediction de tumeur.

Lancement :
    pip install flask
    python api.py

Puis, dans un autre terminal :
    curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
         -d "{\"features\": [ ... 30 nombres ... ]}"

Reponse : {"prediction": 1, "proba": 0.97, "label": "benigne"}

Auteur : Mohammed MOSLEH
"""
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# On recharge le modele + scaler + plages une seule fois au demarrage.
BUNDLE = joblib.load("modele.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    # Validation : la cle "features" doit exister
    if not data or "features" not in data:
        return jsonify({"error": "cle 'features' manquante"}), 400

    features = data["features"]
    attendu = len(BUNDLE["features"])
    if not isinstance(features, list) or len(features) != attendu:
        return jsonify({"error": f"il faut {attendu} features numeriques"}), 400

    # Conversion en nombres (rejette le texte)
    try:
        x = np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return jsonify({"error": "les features doivent etre des nombres"}), 400

    # Normalisation IDENTIQUE a l'entrainement, puis prediction
    x_s = BUNDLE["scaler"].transform(x)
    pred = int(BUNDLE["modele"].predict(x_s)[0])
    proba = float(BUNDLE["modele"].predict_proba(x_s)[0].max())
    label = "benigne" if pred == 1 else "maligne"

    reponse = {"prediction": pred, "proba": round(proba, 3), "label": label}

    # Avertissement si une valeur sort de la plage vue a l'entrainement
    plages = BUNDLE.get("plages")
    if plages is not None:
        hors = int(((x[0] < plages["min"]) | (x[0] > plages["max"])).sum())
        if hors:
            reponse["avertissement"] = f"{hors} valeurs hors de la plage d'entrainement"

    return jsonify(reponse), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
