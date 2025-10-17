import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
from datetime import datetime
import ast

# Create directories
os.makedirs('model', exist_ok=True)
os.makedirs('dataset', exist_ok=True)  # Ensure dataset dir exists

# Load and clean datasets
dataset = pd.read_csv('dataset/Training.csv')
symptom_severity = pd.read_csv('dataset/Symptom-severity.csv')
sym_des = pd.read_csv('dataset/symptoms_df.csv')  # For rule-based

# Fix typos in symptom names (standardize across all)
typo_fixes = {
    'spotting_ urination': 'spotting_urination',
    'foul_smell_ofurine': 'foul_smell_of_urine',
    'dischromic _patches': 'dischromic_patches',  # Remove space
    'fluid_overload.1': 'fluid_overload',  # Dedup
    'cold_hands_and_feets': 'cold_hands_and_feet',
    'vomting': 'vomiting'  # Additional fix based on potential typo
}
# Apply fixes to dataset columns
dataset.columns = [typo_fixes.get(col, col) for col in dataset.columns]
# Remove duplicates after renaming
dataset = dataset.loc[:, ~dataset.columns.duplicated()]
symptom_severity['Symptom'] = [typo_fixes.get(s, s) for s in symptom_severity['Symptom']]

# Ensure prognosis is string type
dataset['prognosis'] = dataset['prognosis'].astype(str)

# Symptoms list with fixes
symptoms_list = {k: v for k, v in {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feet': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
    'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of_urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93,
    'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97,
    'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
    'abnormal_menstruation': 101, 'dischromic_patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
    'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
    'blood_in_sputum': 117, 'prominent_veins_on_calf': 118, 'palpitations': 119,
    'painful_walking': 120, 'pus_filled_pimples': 121, 'blackheads': 122, 'scurring': 123,
    'skin_peeling': 124, 'silver_like_dusting': 125, 'small_dents_in_nails': 126,
    'inflammatory_nails': 127, 'blister': 128, 'red_sore_around_nose': 129,
    'yellow_crust_ooze': 130, 'sore_throat': 131, 'wheezing': 132, 'chest_tightness': 133,
    'loss_of_taste': 134, 'confusion': 135, 'severe_headache': 136, 'lightheadedness': 137,
    'arm_pain': 138, 'jaw_pain': 139, 'loose_stools': 140, 'bloating': 141,
    'fever_with_rash': 142, 'sneezing': 143
}.items()}

# Align dataset columns to expected
expected_columns = list(symptoms_list.keys()) + ['prognosis']
for col in expected_columns:
    if col not in dataset.columns:
        if col == 'prognosis':
            dataset[col] = np.nan
        else:
            dataset[col] = 0
dataset = dataset.reindex(columns=expected_columns)
dataset = dataset.loc[:, ~dataset.columns.duplicated()]

print("Dataset columns after alignment:", dataset.columns.tolist())
print("Any duplicates?", dataset.columns.duplicated().any())

# Add targeted synthetic data
common_conditions_samples = []
diseases = sym_des['Disease'].unique()

for disease in diseases:
    disease_rows = sym_des[sym_des['Disease'] == disease]
    for _ in range(20):  # 20 samples per disease
        row = disease_rows.sample(1).iloc[0]
        sample_symptoms = []
        for i in range(1, 5):  # Symptom_1 to Symptom_4
            sym_val = row[f'Symptom_{i}']
            if pd.notna(sym_val) and str(sym_val).strip() != '':
                fixed_sym = typo_fixes.get(str(sym_val).strip(), str(sym_val).strip())
                sample_symptoms.append(fixed_sym)
        
        # Randomly add 1-2 extra from disease's typical symptoms
        if sample_symptoms:
            extra_count = np.random.randint(1, 3)
            extras = np.random.choice(sample_symptoms, size=extra_count, replace=True)
            sample_symptoms.extend(extras)
        
        sample = {'prognosis': disease}
        for symptom in symptoms_list.keys():
            sample[symptom] = 1 if symptom in sample_symptoms else 0
        common_conditions_samples.append(sample)

common_conditions_df = pd.DataFrame(common_conditions_samples)
common_conditions_df = common_conditions_df.reindex(columns=expected_columns, fill_value=0)
common_conditions_df['prognosis'] = common_conditions_df['prognosis'].astype(str)  # Ensure string type

print("Common conditions df columns:", common_conditions_df.columns.tolist())
print("Any duplicates?", common_conditions_df.columns.duplicated().any())

# Concatenate safely
enhanced_dataset = pd.concat([dataset, common_conditions_df], ignore_index=True)
enhanced_dataset['prognosis'] = enhanced_dataset['prognosis'].astype(str)  # Final type enforcement

# Features with severity weighting
X = enhanced_dataset.drop('prognosis', axis=1)
severity_weights = dict(zip(symptom_severity['Symptom'], symptom_severity['weight']))
for col in X.columns:
    if col in severity_weights:
        X[col] *= severity_weights[col]  # Weight symptoms

y = enhanced_dataset['prognosis']
le = LabelEncoder()
le.fit(y)  # Should now work with uniform strings
Y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

Rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight_dict, max_depth=15)
Rf.fit(X_train, y_train)

y_pred = Rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, zero_division=0))

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pickle.dump(Rf, open(f'model/RandomForest_{timestamp}.pkl', 'wb'))
pickle.dump(le, open(f'model/label_encoder_{timestamp}.pkl', 'wb'))
with open('model/latest_model.txt', 'w') as f:
    f.write(f"RandomForest_{timestamp}.pkl\nlabel_encoder_{timestamp}.pkl\n")

# Save cleaned symptoms_list
with open('model/symptoms_list.pkl', 'wb') as f:
    pickle.dump(symptoms_list, f)

print("Training complete with fixes!")