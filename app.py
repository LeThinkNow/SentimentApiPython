from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
import numpy as np

app = FastAPI(title="API Sentimiento")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos
modelos = {}
try:
    modelos['es'] = joblib.load("modelo_sentimiento_es.joblib")
    modelos['en'] = joblib.load("model_en.joblib")
    modelos['pt'] = joblib.load("model_pt.joblib")
except Exception as e:
    print(f"Error cargando modelos: {e}")

def limpiar_texto(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-záéíóúñ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TextoRequest(BaseModel):
    text: str
    language: str = "es" # Default es
    threshold: float = None

def get_top_features(model, text, prediction_label):
    try:
        if not hasattr(model, 'named_steps'): return []
        
        vectorizer = model.named_steps['countvectorizer']
        classifier = model.named_steps['logisticregression']
        
        feature_names = vectorizer.get_feature_names_out()
        coefs = classifier.coef_[0] # Assuming binary classification
        
        # Palabras en el texto
        words = text.split()
        word_scores = []
        
        for w in words:
            if w in vectorizer.vocabulary_:
                idx = vectorizer.vocabulary_[w]
                score = coefs[idx]
                word_scores.append((w, score))
        
        # Si es Positivo, buscamos los scores más altos positivos
        # Si es Negativo, buscamos los scores más bajos (más negativos)
        
        target_positive = (prediction_label == "Positivo")
        
        if target_positive:
            # Sort desc
            word_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            # Sort asc (most negative first)
            word_scores.sort(key=lambda x: x[1])
            
        # Tomar top 3
        return [w for w, s in word_scores[:3]]
    except Exception as e:
        print(f"Error explicabilidad: {e}")
        return []


@app.post("/predict")
def predict(request: TextoRequest):
    lang = request.language if request.language in modelos else 'es'
    model = modelos.get(lang)
    
    if not model:
        return {"prevision": "Error", "probabilidad": 0.0, "detalle": "Modelo no cargado"}

    texto_limpio = limpiar_texto(request.text)
    
    try:
        proba = model.predict_proba([texto_limpio])[0]
        # Classes usually: [Negativo, Positivo] alphabetical order? 
        # Check classes_ attribute
        classes = model.classes_
        
        prob_dict = {c: p for c, p in zip(classes, proba)}
        
        # Default Logic
        prob_pos = prob_dict.get("Positivo", 0.0)
        prob_neg = prob_dict.get("Negativo", 0.0)
        
        prevision = "Positivo" if prob_pos >= prob_neg else "Negativo"
        probabilidad = prob_pos if prevision == "Positivo" else prob_neg
        
        # Threshold logic override
        if request.threshold:
            if prob_pos >= request.threshold:
                prevision = "Positivo"
                probabilidad = prob_pos
            else:
                prevision = "Negativo"
                probabilidad = prob_neg # Or 1 - prob_pos
                
        # Explainability
        top_features = get_top_features(model, texto_limpio, prevision)

        return {
            "prevision": prevision,
            "probabilidad": round(float(probabilidad), 4),
            "top_features": top_features
        }
    except Exception as e:
        return {"prevision": "Error", "probabilidad": 0.0, "detalle": str(e)}
