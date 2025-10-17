from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process
import ast
import os
import re
from datetime import datetime
import uuid
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- FASTAPI APP --------------------
app = FastAPI(title="AI-Driven Disease Prediction System")
templates = Jinja2Templates(directory="templates")

# -------------------- ENV & GEMINI SETUP --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("❌ GOOGLE_API_KEY not found in environment. Please set it in .env or system environment.")
    raise RuntimeError("GOOGLE_API_KEY not configured")

# Configure Gemini client (new SDK)
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"  # ✅ Current stable model
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# -------------------- LOAD ML MODELS & DATA --------------------
try:
    with open('model/latest_model.txt', 'r') as f:
        model_files = f.read().splitlines()
    model_path = f"model/{model_files[0]}"
    encoder_path = f"model/{model_files[1]}"

    with open('model/symptoms_list.pkl', 'rb') as f:
        symptoms_list = pickle.load(f)

    sym_des = pd.read_csv("dataset/symptoms_df.csv")
    precautions = pd.read_csv("dataset/precautions_df.csv")
    workout = pd.read_csv("dataset/workout_df.csv")
    description = pd.read_csv("dataset/description.csv")
    medications = pd.read_csv("dataset/medications.csv")
    diets = pd.read_csv("dataset/diets.csv")
    symptom_severity = pd.read_csv("dataset/Symptom-severity.csv")

    Rf = pickle.load(open(model_path, "rb"))
    le = pickle.load(open(encoder_path, "rb"))
    diseases_list = {i: disease for i, disease in enumerate(le.classes_)}
except Exception as e:
    logger.error(f"❌ Error loading local ML models or datasets: {e}")
    exit(1)

# -------------------- DATA CLEANUP --------------------
typo_fixes = {
    'spotting_ urination': 'spotting_urination',
    'foul_smell_ofurine': 'foul_smell_of_urine',
    'dischromic _patches': 'dischromic_patches',
    'fluid_overload.1': 'fluid_overload',
    'cold_hands_and_feets': 'cold_hands_and_feet'
}
symptom_severity['Symptom'] = [typo_fixes.get(s, s) for s in symptom_severity['Symptom']]

# Apply fixes to CSV symptoms
for i in range(1, 5):
    col = f'Symptom_{i}'  # Fixed typo: changed 'col_ode' to 'col'
    if col in sym_des.columns:
        sym_des[col] = sym_des[col].astype(str).apply(lambda x: typo_fixes.get(x.strip() if pd.notna(x) else '', x.strip() if pd.notna(x) else ''))

# Symptom variations mapping
symptom_variations = {
    'headache': ['head pain', 'migraine', 'throbbing head', 'cephalgia'],
    'fever': ['high fever', 'temperature', 'feeling hot'],
    'cough': ['dry cough', 'wet cough', 'coughing'],
    'fatigue': ['tired', 'exhausted', 'weakness'],
    'stomach_pain': ['abdominal pain', 'belly ache', 'tummy pain'],
    'itching': ['itchy skin', 'pruritus', 'scratchy'],
    'rash': ['skin rash', 'red spots', 'skin eruption'],
    'breathlessness': ['shortness of breath', 'dyspnea', 'gasping'],
    'diarrhoea': ['loose stools', 'diarrhea'],
    'nausea': ['feeling sick', 'vomit', 'queasiness']
}
variation_to_symptom = {v.lower(): k for k, vals in symptom_variations.items() for v in vals}

# -------------------- HELPER FUNCTIONS --------------------
def correct_symptom_spelling(symptom):
    s = symptom.lower().strip()
    if s in variation_to_symptom:
        return variation_to_symptom[s]
    result = process.extractOne(s, list(symptoms_list.keys()))
    return result[0] if result and result[1] >= 70 else None

def ml_predict(symptoms):
    input_vector = np.zeros(len(symptoms_list))
    for s in symptoms:
        if s in symptoms_list:
            input_vector[symptoms_list[s]] = 1
    probs = Rf.predict_proba([input_vector])[0]
    top_idx = np.argmax(probs)
    return diseases_list[top_idx], probs[top_idx]

def get_disease_info(predicted_dis):
    desc = description[description['Disease'] == predicted_dis]['Description'].values
    desc = desc[0] if len(desc) > 0 else "No description available."
    prec = precautions[precautions['Disease'] == predicted_dis].iloc[0].dropna().tolist()[1:] if predicted_dis in precautions['Disease'].values else ["Consult a doctor."]
    meds = ast.literal_eval(medications[medications['Disease'] == predicted_dis]['Medication'].values[0]) if predicted_dis in medications['Disease'].values else ["Consult a doctor."]
    diet = ast.literal_eval(diets[diets['Disease'] == predicted_dis]['Diet'].values[0]) if predicted_dis in diets['Disease'].values else ["Follow a balanced diet."]
    work = workout[workout['disease'] == predicted_dis]['workout'].tolist()
    work = work if work else ["Rest and light activity only."]
    return desc, prec, meds, diet, work

# -------------------- GEMINI INTEGRATION --------------------
def call_gemini_api_sync(symptoms, age=None, gender=None, medical_history=None):
    try:
        prompt = f"""
        You are an AI health assistant.
        Analyze these symptoms: {', '.join(symptoms)}.
        Patient details — Age: {age}, Gender: {gender}, History: {medical_history}.
        Provide output ONLY in JSON:
        {{
          "predicted_disease": string,
          "confidence": float,
          "description": string,
          "precautions": [string],
          "medications": [string],
          "diet": [string],
          "workout": string,
          "alternative_diagnoses": [{{"disease": string, "confidence": float}}]
        }}
        """

        logger.info(f"🤖 Calling Gemini Model: {GEMINI_MODEL_NAME}")
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()

        # Extract JSON safely
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("Gemini response not JSON.")
        result = json.loads(match.group(0))
        logger.info(f"✅ Gemini output parsed successfully: {result}")
        return result

    except Exception as e:
        logger.error(f"❌ Gemini API failed: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

async def call_gemini_api(symptoms, age=None, gender=None, medical_history=None):
    loop = asyncio.get_event_loop()
    try:
        with ThreadPoolExecutor() as pool:
            return await asyncio.wait_for(
                loop.run_in_executor(pool, lambda: call_gemini_api_sync(symptoms, age, gender, medical_history)),
                timeout=30.0  # 30-second timeout for Gemini API
            )
    except asyncio.TimeoutError:
        logger.warning("⚠️ Gemini API timed out after 30 seconds, falling back to local ML model")
        raise HTTPException(status_code=504, detail="Gemini API timed out")

# -------------------- ROUTES --------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict(
    request: Request,
    symptoms: str = Form(...),
    age: int = Form(None),
    gender: str = Form(None),
    medical_history: str = Form(None),
    use_gemini: bool = Form(True)  # ✅ Default to Gemini for hackathon
):
    user_symptoms = [correct_symptom_spelling(s) for s in re.split(r'[, ]+', symptoms) if correct_symptom_spelling(s)]
    if not user_symptoms:
        raise HTTPException(status_code=400, detail="Please enter valid symptoms.")

    try:
        if use_gemini:
            result = await call_gemini_api(user_symptoms, age, gender, medical_history)
            return JSONResponse(content={
                "predicted_by": "Gemini AI",
                "symptoms": user_symptoms,
                **result,
                "disclaimer": "This is for informational purposes only. Consult a doctor."
            })
    except Exception as e:
        logger.warning(f"⚠️ Gemini failed, falling back to local ML model: {e}")

    # ---- Local ML fallback ----
    disease, conf = ml_predict(user_symptoms)
    desc, prec, meds, diet, work = get_disease_info(disease)
    return JSONResponse(content={
        "predicted_by": "Local ML Model",
        "symptoms": user_symptoms,
        "predicted_disease": disease,
        "confidence": f"{conf:.2f}",
        "description": desc,
        "precautions": prec,
        "medications": meds,
        "diet": diet,
        "workout": "; ".join(work),
        "disclaimer": "This is for informational purposes only. Consult a doctor."
    })

# -------------------- MAIN --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)