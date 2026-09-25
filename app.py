import os
import re
import pandas as pd
import numpy as np
import requests
import folium
import torch
import gradio as gr
import torchvision.models as models
from torchvision import transforms
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from datetime import datetime

REPORT_FILE = "sih_crop_health_reports.csv"
FOLLOWUP_FILE = "sih_crop_followup.csv"
FEEDBACK_FILE = "sih_ai_feedback.csv"

# =========================
# MODELS
# =========================
disease_model = models.efficientnet_b0(weights=None)
disease_model.classifier[1] = torch.nn.Linear(
    disease_model.classifier[1].in_features, 38
)
disease_model.load_state_dict(
    torch.load("plant_disease_efficientnet_b0.pth", map_location="cpu")
)
disease_model.eval()

pest_model = YOLO("best (1).pt")

disease_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

classes = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
    'Apple___healthy','Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
    'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
CONFIDENCE_THRESHOLD = 0.70


def get_weather_risk(temperature, humidity, rainfall):

    if humidity >= 80 and rainfall >= 10:
        return "🔴 HIGH RISK"

    elif humidity >= 65 or rainfall >= 5:
        return "🟡 MEDIUM RISK"

    else:
        return "🟢 LOW RISK"



def calculate_overall_risk(
    disease_confidence,
    pest_confidence,
    weather_risk
):
    score = 0

    if disease_confidence < 0.70:
        score += 40
    elif disease_confidence < 0.85:
        score += 25
    else:
        score += 10

    if pest_confidence > 0:
        score += 30

    if "HIGH" in weather_risk:
        score += 30
    elif "MEDIUM" in weather_risk:
        score += 20
    else:
        score += 5

    if score >= 70:
        risk = "🔴 HIGH CROP RISK"
    elif score >= 40:
        risk = "🟡 MEDIUM CROP RISK"
    else:
        risk = "🟢 LOW CROP RISK"

    return score, risk



def generate_farmer_alert(risk_level):

    if "HIGH" in risk_level:
        return """
🚨 FARMER ALERT

⚠️ High crop risk detected!

🔍 Increase crop monitoring immediately.
🌱 Inspect affected plants.
🔬 Expert validation is recommended.
📞 Contact the nearest agricultural expert if symptoms spread.
"""

    elif "MEDIUM" in risk_level:
        return """
⚠️ FARMER ALERT

🟡 Medium crop risk detected.

🌱 Monitor the crop regularly.
🔍 Check for disease and pest symptoms.
🌦️ Continue monitoring weather conditions.
"""

    else:
        return """
✅ FARMER STATUS

🟢 Low crop risk detected.

🌱 Continue regular crop monitoring
and preventive crop management.
"""


def generate_advisory(disease, pest, weather_risk):
    advice = []

    if "healthy" not in disease.lower():
        advice.append(
            f"🦠 Disease: {disease}\n"
            "Remove affected plant parts and monitor the crop regularly."
        )
    else:
        advice.append(
            "🦠 Disease: No major disease indicated.\n"
            "Continue regular crop monitoring."
        )

    if pest != "No pest detected":
        advice.append(
            f"🐛 Pest: {pest}\n"
            "Inspect affected plants and consider appropriate integrated pest management."
        )
    else:
        advice.append(
            "🐛 Pest: No pest detected.\n"
            "Continue regular monitoring."
        )

    if "HIGH" in weather_risk:
        advice.append(
            "🌦️ Weather: HIGH RISK\n"
            "Increase field monitoring and consider expert validation."
        )
    elif "MEDIUM" in weather_risk:
        advice.append(
            "🌦️ Weather: MEDIUM RISK\n"
            "Monitor the crop closely during the coming period."
        )
    else:
        advice.append(
            "🌦️ Weather: LOW RISK\n"
            "Continue preventive crop management."
        )

    return "\n\n".join(advice)



language_options = [
    "English",
    "தமிழ்",
    "हिन्दी",
    "తెలుగు",
    "ಕನ್ನಡ",
    "മലയാളം",
    "বাংলা",
    "मराठी",
    "ગુજરાતી",
    "ਪੰਜਾਬੀ",
    "ଓଡ଼ିଆ"
]


def get_multilingual_advisory(disease, pest, weather_risk, language):

    if language == "English":
        return generate_advisory(disease, pest, weather_risk)

    translations = {
        "தமிழ்": ("🌱 பயிர் சுகாதார ஆலோசனை",
                  "நோய்", "பூச்சி", "வானிலை அபாயம்",
                  "பாதிக்கப்பட்ட தாவரப் பகுதிகளை அகற்றி, பயிரை தொடர்ந்து கண்காணிக்கவும்."),

        "हिन्दी": ("🌱 फसल स्वास्थ्य सलाह",
                   "रोग", "कीट", "मौसम जोखिम",
                   "प्रभावित पौधों के हिस्सों को हटाएं और फसल की नियमित निगरानी करें।"),

        "తెలుగు": ("🌱 పంట ఆరోగ్య సలహా",
                    "వ్యాధి", "పురుగు", "వాతావరణ ప్రమాదం",
                    "ప్రభావిత మొక్కల భాగాలను తొలగించి పంటను పర్యవేక్షించండి."),

        "ಕನ್ನಡ": ("🌱 ಬೆಳೆ ಆರೋಗ್ಯ ಸಲಹೆ",
                   "ರೋಗ", "ಕೀಟ", "ಹವಾಮಾನ ಅಪಾಯ",
                   "ಬಾಧಿತ ಸಸ್ಯ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ."),

        "മലയാളം": ("🌱 വിള ആരോഗ്യ ഉപദേശം",
                    "രോഗം", "കീടം", "കാലാവസ്ഥാ അപകടസാധ്യത",
                    "ബാധിച്ച സസ്യഭാഗങ്ങൾ നീക്കം ചെയ്ത് വിള നിരീക്ഷിക്കുക."),

        "বাংলা": ("🌱 ফসল স্বাস্থ্য পরামর্শ",
                   "রোগ", "পোকা", "আবহাওয়ার ঝুঁকি",
                   "আক্রান্ত উদ্ভিদের অংশ সরিয়ে ফেলুন এবং নিয়মিত ফসল পর্যবেক্ষণ করুন।"),

        "मराठी": ("🌱 पीक आरोग्य सल्ला",
                   "रोग", "कीड", "हवामानाचा धोका",
                   "बाधित झाडांचे भाग काढून टाका आणि पिकाचे नियमित निरीक्षण करा."),

        "ગુજરાતી": ("🌱 પાક આરોગ્ય સલાહ",
                     "રોગ", "જીવાત", "હવામાન જોખમ",
                     "અસરગ્રસ્ત છોડના ભાગોને દૂર કરો અને પાકનું નિયમિત નિરીક્ષણ કરો."),

        "ਪੰਜਾਬੀ": ("🌱 ਫਸਲ ਸਿਹਤ ਸਲਾਹ",
                    "ਬਿਮਾਰੀ", "ਕੀੜਾ", "ਮੌਸਮ ਦਾ ਖਤਰਾ",
                    "ਪ੍ਰਭਾਵਿਤ ਪੌਦਿਆਂ ਦੇ ਹਿੱਸੇ ਹਟਾਓ ਅਤੇ ਫਸਲ ਦੀ ਨਿਯਮਿਤ ਨਿਗਰਾਨੀ ਕਰੋ."),

        "ଓଡ଼ିଆ": ("🌱 ଫସଲ ସ୍ୱାସ୍ଥ୍ୟ ପରାମର୍ଶ",
                   "ରୋଗ", "କୀଟ", "ପାଣିପାଗ ଜୋଖିମ",
                   "ଆକ୍ରାନ୍ତ ଗଛର ଅଂଶଗୁଡ଼ିକୁ ବାହାର କରନ୍ତୁ ଏବଂ ଫସଲକୁ ନିୟମିତ ନିରୀକ୍ଷଣ କରନ୍ତୁ.")
    }

    if language in translations:
        title, disease_label, pest_label, weather_label, advice = translations[language]

        return f"""
{title}

🦠 {disease_label}: {disease}
🐛 {pest_label}: {pest}
🌦️ {weather_label}: {weather_risk}

💊 பரிந்துரை / Advice:
{advice}

🐛 Pest management:
Use integrated pest management and continue regular monitoring.

🔬 Expert validation:
Contact an agricultural expert if the risk is high.
"""

    return generate_advisory(disease, pest, weather_risk)


def complete_analysis(img, language, latitude, longitude):

    if img is None:
        return (
            "⚠️ Please upload a crop image.",
            "",
            "",
            "",
            "",
            "",
            "⚠️ Please upload an image."
        )

    # 🦠 Disease detection
    image_tensor = disease_transform(img).unsqueeze(0)

    with torch.no_grad():
        output = disease_model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    disease = classes[predicted.item()]
    disease_conf = confidence.item()

    # 🔬 Expert validation
    if disease_conf >= CONFIDENCE_THRESHOLD:
        validation = "✅ Prediction confidence acceptable."
    else:
        validation = (
            "⚠️ Uncertain prediction\n"
            "🔬 Expert validation recommended."
        )

    # 🐛 Pest detection
    pest_results = pest_model.predict(
        source=img,
        conf=0.25,
        verbose=False
    )

    pest_name = "No pest detected"
    pest_conf = 0

    if len(pest_results) > 0 and len(pest_results[0].boxes) > 0:
        box = pest_results[0].boxes[0]
        pest_conf = float(box.conf[0])
        pest_id = int(box.cls[0])
        pest_name = pest_results[0].names[pest_id]

    # 🌦️ Weather
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m,relative_humidity_2m,rain"
    )

    try:
        weather = requests.get(url, timeout=10).json()

        temperature = weather["current"]["temperature_2m"]
        humidity = weather["current"]["relative_humidity_2m"]
        rainfall = weather["current"]["rain"]

        weather_risk = get_weather_risk(
            temperature,
            humidity,
            rainfall
        )

    except Exception:
        temperature = "Unavailable"
        humidity = "Unavailable"
        rainfall = "Unavailable"
        weather_risk = "⚠️ Weather unavailable"

    # 📊 Overall risk
    risk_score, risk_level = calculate_overall_risk(
        disease_conf,
        pest_conf,
        weather_risk
    )

    # 🚨 Farmer alert
    farmer_alert = generate_farmer_alert(risk_level)

    # 💊 Advisory
    advisory = get_multilingual_advisory(
        disease,
        pest_name,
        weather_risk,
        language
    )

    # 🗺️ Farmer location map
    farmer_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=13
    )

    folium.Marker(
        [latitude, longitude],
        popup="📍 Farmer Field Location",
        tooltip="Farmer Location"
    ).add_to(farmer_map)

    analysis = f"""
🌱 CROP HEALTH ANALYSIS

🦠 Disease:
{disease}

📊 Disease Confidence:
{disease_conf * 100:.2f}%

🐛 Pest:
{pest_name}

📊 Pest Confidence:
{pest_conf * 100:.2f}%

🌦️ LIVE WEATHER

🌡️ Temperature: {temperature} °C
💧 Humidity: {humidity} %
🌧️ Rainfall: {rainfall} mm

⚠️ Weather Risk:
{weather_risk}

📊 OVERALL CROP RISK

Risk Score: {risk_score}/100
Risk Level: {risk_level}

📍 FARMER LOCATION

Latitude: {latitude}
Longitude: {longitude}
"""

    status = "✅ Analysis completed successfully."

    return (
        analysis,
        advisory,
        validation,
        farmer_alert,
        f"{risk_level}\nRisk Score: {risk_score}/100",
        farmer_map._repr_html_(),
        status
    )



import requests

def get_weather_forecast(latitude, longitude):

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,rain_sum"
        f"&forecast_days=3"
        f"&timezone=auto"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    daily = data["daily"]

    forecast = []

    for i in range(len(daily["time"])):
        forecast.append({
            "Date": daily["time"][i],
            "Max Temperature": daily["temperature_2m_max"][i],
            "Min Temperature": daily["temperature_2m_min"][i],
            "Rainfall": daily["rain_sum"][i]
        })

    return forecast


def get_location_weather_forecast(latitude, longitude):

    try:
        latitude = float(latitude)
        longitude = float(longitude)

        forecast = get_weather_forecast(
            latitude,
            longitude
        )

        output = "🌦️ 3-DAY WEATHER FORECAST\n\n"

        for day in forecast:
            output += (
                f"📅 {day['Date']}\n"
                f"🌡️ Max Temperature: {day['Max Temperature']} °C\n"
                f"🌡️ Min Temperature: {day['Min Temperature']} °C\n"
                f"🌧️ Expected Rainfall: {day['Rainfall']} mm\n\n"
            )

        return output

    except Exception as e:
        return f"⚠️ Weather forecast unavailable: {e}"


# Quick test using your automatically detected location
print(
    get_location_weather_forecast(
        11.3410,
        77.7172
    )
)

# ============================================
# 🌱 IPM + SAFE INPUT RECOMMENDATION MODULE
# ============================================

IPM_DATABASE = {

    "Early blight": {
        "ipm": [
            "Remove infected leaves and destroy them safely.",
            "Avoid overhead irrigation.",
            "Maintain proper spacing for good air circulation.",
            "Rotate crops where possible."
        ],
        "safe_input": [
            "Use approved fungicides only according to the product label.",
            "Prefer bio-control options such as Trichoderma where suitable.",
            "Follow the recommended dose and waiting period."
        ]
    },

    "Late blight": {
        "ipm": [
            "Remove severely infected plant parts.",
            "Avoid prolonged leaf wetness.",
            "Improve field ventilation.",
            "Monitor the crop closely during wet weather."
        ],
        "safe_input": [
            "Use a locally approved fungicide when required.",
            "Follow label dosage and pre-harvest interval.",
            "Avoid unnecessary repeated chemical applications."
        ]
    },

    "Powdery mildew": {
        "ipm": [
            "Remove heavily infected leaves.",
            "Improve air circulation.",
            "Avoid excessive nitrogen application.",
            "Monitor new growth regularly."
        ],
        "safe_input": [
            "Use approved sulfur-based or other registered fungicides when appropriate.",
            "Follow the product label and crop-specific instructions."
        ]
    },

    "Rice blast": {
        "ipm": [
            "Use healthy and disease-free seed.",
            "Avoid excessive nitrogen fertilizer.",
            "Maintain balanced irrigation.",
            "Remove severely affected plant material."
        ],
        "safe_input": [
            "Use locally approved rice-blast management products when necessary.",
            "Follow label dosage and pre-harvest interval."
        ]
    }
}


def get_ipm_recommendation(disease_name):

    disease_name = str(disease_name)

    # Find matching disease
    matched_key = None

    for key in IPM_DATABASE:

        if key.lower() in disease_name.lower():
            matched_key = key
            break

    if matched_key is None:
        return (
            "🌱 IPM RECOMMENDATION\n\n"
            "• Monitor the crop regularly.\n"
            "• Remove severely infected plant parts if appropriate.\n"
            "• Maintain field hygiene and proper spacing.\n"
            "• Avoid unnecessary pesticide application.\n\n"
            "🛡️ SAFE INPUT\n\n"
            "Use only locally registered agricultural products "
            "according to the label and expert recommendation."
        )

    data = IPM_DATABASE[matched_key]

    output = f"🌱 IPM RECOMMENDATION — {matched_key}\n\n"

    for action in data["ipm"]:
        output += f"• {action}\n"

    output += "\n🛡️ SAFE INPUT RECOMMENDATION\n\n"

    for action in data["safe_input"]:
        output += f"• {action}\n"

    return output




# ============================================
# 🌱 CONNECT IPM TO FARMER DASHBOARD
# ============================================

def extract_disease_for_ipm(analysis_text):
    """
    Extract the detected disease name from the AI analysis.
    """

    text = str(analysis_text)

    for disease in IPM_DATABASE:

        if disease.lower() in text.lower():
            return get_ipm_recommendation(disease)

    return get_ipm_recommendation("Unknown")


# Test with an example
test_analysis = "Disease detected: Early blight"



# ============================================
# 🧑‍🔬 EXPERT / LAB REFERRAL MODULE
# ============================================

def generate_referral(disease_confidence, overall_risk):

    try:
        confidence = float(disease_confidence)
    except:
        confidence = 0.0

    if confidence < 0.70:
        return """
🧑‍🔬 EXPERT / LAB REFERRAL

⚠️ AI diagnosis confidence is low.

Recommended action:
• Take a clear photo of the affected plant.
• Consult an agricultural extension officer.
• Submit a plant sample to an agricultural laboratory if required.
• Do not apply pesticides based only on the AI prediction.
"""

    if "HIGH" in str(overall_risk).upper():
        return """
🧑‍🔬 EXPERT / LAB REFERRAL

🔴 High crop risk detected.

Recommended action:
• Contact an agricultural expert for confirmation.
• Consider laboratory testing if the disease is unclear.
• Follow expert advice before applying chemical inputs.
• Continue monitoring the affected field.
"""

    return """
🧑‍🔬 EXPERT / LAB REFERRAL

✅ No immediate laboratory referral required.

Continue:
• Regular crop monitoring
• IPM practices
• Weather-risk monitoring
• Follow-up observation
"""


# ============================================
# CONNECT ANALYSIS → REFERRAL
# ============================================

def referral_from_analysis(analysis_text, risk_text):

    import re

    text = str(analysis_text)

    confidence = 0.0

    match = re.search(
        r'(\d+(?:\.\d+)?)\s*%',
        text
    )

    if match:
        confidence = float(match.group(1)) / 100

    return generate_referral(
        confidence,
        risk_text
    )


# ============================================
# TEST
# ============================================



import os
import re
import pandas as pd
import gradio as gr
import folium
import numpy as np
from sklearn.cluster import DBSCAN

REPORT_FILE = "sih_crop_health_reports.csv"


# =========================================================
# CONNECTED ANALYSIS
# =========================================================

def connected_analysis(img, language, latitude, longitude):

    # Run your existing AI analysis
    result = complete_analysis(
        img,
        language,
        latitude,
        longitude
    )

    analysis = result[0]
    advisory = result[1]
    validation = result[2]
    farmer_alert = result[3]
    risk_output = result[4]
    map_html = result[5]
    status = result[6]

    # Extract important values from AI result
    disease_match = re.search(
        r"🦠 Disease:\s*\n(.+)",
        analysis
    )

    disease_conf_match = re.search(
        r"📊 Disease Confidence:\s*\n([\d.]+)%",
        analysis
    )

    pest_match = re.search(
        r"🐛 Pest:\s*\n(.+)",
        analysis
    )

    pest_conf_match = re.search(
        r"📊 Pest Confidence:\s*\n([\d.]+)%",
        analysis
    )

    temperature_match = re.search(
        r"🌡️ Temperature:\s*([\d.]+)",
        analysis
    )

    humidity_match = re.search(
        r"💧 Humidity:\s*([\d.]+)",
        analysis
    )

    rainfall_match = re.search(
        r"🌧️ Rainfall:\s*([\d.]+)",
        analysis
    )

    weather_match = re.search(
        r"⚠️ Weather Risk:\s*\n(.+)",
        analysis
    )

    score_match = re.search(
        r"Risk Score:\s*(\d+)/100",
        analysis
    )

    risk_match = re.search(
        r"Risk Level:\s*(.+)",
        analysis
    )

    disease = (
        disease_match.group(1)
        if disease_match else "Unknown"
    )

    disease_conf = (
        float(disease_conf_match.group(1))
        if disease_conf_match else 0
    )

    pest = (
        pest_match.group(1)
        if pest_match else "No pest detected"
    )

    pest_conf = (
        float(pest_conf_match.group(1))
        if pest_conf_match else 0
    )

    temperature = (
        float(temperature_match.group(1))
        if temperature_match else None
    )

    humidity = (
        float(humidity_match.group(1))
        if humidity_match else None
    )

    rainfall = (
        float(rainfall_match.group(1))
        if rainfall_match else None
    )

    weather_risk = (
        weather_match.group(1)
        if weather_match else "Unknown"
    )

    risk_score = (
        int(score_match.group(1))
        if score_match else 0
    )

    risk_level = (
        risk_match.group(1)
        if risk_match else "Unknown"
    )

    # =====================================================
    # SAVE COMPLETE REPORT
    # =====================================================

    new_report = pd.DataFrame([{
        "Latitude": latitude,
        "Longitude": longitude,
        "Language": language,
        "Disease": disease,
        "Disease_Confidence": disease_conf,
        "Pest": pest,
        "Pest_Confidence": pest_conf,
        "Temperature_C": temperature,
        "Humidity_Percent": humidity,
        "Rainfall_mm": rainfall,
        "Weather_Risk": weather_risk,
        "Risk_Score": risk_score,
        "Risk_Level": risk_level,
        "Expert_Validation": validation,
        "Farmer_Alert": farmer_alert,
        "Advisory": advisory
    }])

    if os.path.exists(REPORT_FILE):
        old_reports = pd.read_csv(REPORT_FILE)
        all_reports = pd.concat(
            [old_reports, new_report],
            ignore_index=True
        )
    else:
        all_reports = new_report

    all_reports.to_csv(
        REPORT_FILE,
        index=False
    )

    return (
        analysis,
        advisory,
        validation,
        farmer_alert,
        risk_output,
        map_html,
        status
    )




import gradio as gr
import pandas as pd
from datetime import datetime
import os

REPORT_FILE = "sih_crop_health_reports.csv"
FOLLOWUP_FILE = "sih_crop_followup.csv"


def get_reports():

    if not os.path.exists(REPORT_FILE):
        return pd.DataFrame()

    return pd.read_csv(REPORT_FILE)


def load_previous_report(report_number):

    df = get_reports()

    if df.empty:
        return (
            "No report available",
            "",
            "",
            "",
            "",
            "",
            ""
        )

    try:
        index = int(report_number) - 1
        row = df.iloc[index]

        return (
            str(row["Disease"]),
            str(row["Disease_Confidence"]),
            str(row["Risk_Level"]),
            str(row["Risk_Score"]),
            str(row["Latitude"]),
            str(row["Longitude"]),
            str(row["Language"])
        )

    except Exception as e:

        return (
            f"Error: {e}",
            "",
            "",
            "",
            "",
            "",
            ""
        )


def save_followup(
    report_number,
    condition,
    observation
):

    df = get_reports()

    if df.empty:
        return "⚠️ No previous report found."

    try:

        index = int(report_number) - 1
        row = df.iloc[index]

        record = {
            "Followup_Date":
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "Report_Number":
                report_number,

            "Disease":
                row["Disease"],

            "Disease_Confidence":
                row["Disease_Confidence"],

            "Original_Risk_Level":
                row["Risk_Level"],

            "Original_Risk_Score":
                row["Risk_Score"],

            "Latitude":
                row["Latitude"],

            "Longitude":
                row["Longitude"],

            "Language":
                row["Language"],

            "Current_Crop_Condition":
                condition,

            "Farmer_Observation":
                observation
        }

        new_df = pd.DataFrame([record])

        if os.path.exists(FOLLOWUP_FILE):

            old_df = pd.read_csv(FOLLOWUP_FILE)

            new_df = pd.concat(
                [old_df, new_df],
                ignore_index=True
            )

        new_df.to_csv(
            FOLLOWUP_FILE,
            index=False
        )

        return "✅ Follow-up observation saved successfully."

    except Exception as e:

        return f"⚠️ Error: {e}"




def generate_followup_feedback(condition):

    if condition == "Improving":
        return """
🟢 FOLLOW-UP STATUS

Crop condition is improving.

Recommended:
• Continue the advised IPM practices.
• Continue weather-risk monitoring.
• Observe the crop regularly.
"""

    elif condition == "No Change":
        return """
🟡 FOLLOW-UP STATUS

No significant change has been reported.

Recommended:
• Continue monitoring the crop.
• Follow the existing advisory.
• Reassess the crop after further observation.
"""

    elif condition == "Worsening":
        return """
🔴 FOLLOW-UP STATUS

Crop condition is worsening.

Recommended:
• Contact an agricultural expert.
• Consider laboratory confirmation if required.
• Avoid applying additional pesticides without proper guidance.
• Upload a new crop image for further AI analysis if available.
"""

    elif condition == "Recovered":
        return """
✅ FOLLOW-UP STATUS

Crop recovery has been reported.

Recommended:
• Continue regular crop monitoring.
• Maintain field hygiene.
• Record the recovery for future AI improvement.
"""

    return "Select the current crop condition."



def save_followup_with_feedback(
    report_number,
    condition,
    observation
):

    df = get_reports()

    if df.empty:
        return "⚠️ No previous report found."

    try:
        index = int(report_number) - 1
        row = df.iloc[index]

        feedback = generate_followup_feedback(condition)

        record = {
            "Followup_Date":
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "Report_Number":
                report_number,

            "Disease":
                row["Disease"],

            "Disease_Confidence":
                row["Disease_Confidence"],

            "Original_Risk_Level":
                row["Risk_Level"],

            "Original_Risk_Score":
                row["Risk_Score"],

            "Latitude":
                row["Latitude"],

            "Longitude":
                row["Longitude"],

            "Language":
                row["Language"],

            "Current_Crop_Condition":
                condition,

            "Farmer_Observation":
                observation,

            "Followup_Feedback":
                feedback
        }

        new_df = pd.DataFrame([record])

        if os.path.exists(FOLLOWUP_FILE):
            old_df = pd.read_csv(FOLLOWUP_FILE)
            new_df = pd.concat(
                [old_df, new_df],
                ignore_index=True
            )

        new_df.to_csv(
            FOLLOWUP_FILE,
            index=False
        )

        return feedback

    except Exception as e:
        return f"⚠️ Error: {e}"

def load_followup_history():

    if not os.path.exists(FOLLOWUP_FILE):
        return pd.DataFrame(
            columns=[
                "Followup_Date",
                "Report_Number",
                "Disease",
                "Disease_Confidence",
                "Original_Risk_Level",
                "Original_Risk_Score",
                "Latitude",
                "Longitude",
                "Language",
                "Current_Crop_Condition",
                "Farmer_Observation",
                "Followup_Feedback"
            ]
        )

    return pd.read_csv(FOLLOWUP_FILE)




def record_followup_confirmation(
    report_number,
    field_confirmed,
    actual_disease="",
    farmer_note=""
):

    df = get_reports()

    if df.empty:
        return "⚠️ No previous report found."

    try:
        index = int(report_number) - 1
        row = df.iloc[index]

        predicted_disease = str(row["Disease"])

        save_ai_feedback(
            predicted_disease=predicted_disease,
            field_confirmed=field_confirmed,
            actual_disease=actual_disease,
            farmer_note=farmer_note
        )

        return "✅ Field confirmation saved for AI improvement."

    except Exception as e:
        return f"⚠️ Error: {e}"

import pandas as pd
from datetime import datetime

FEEDBACK_FILE = "sih_ai_feedback.csv"

def save_ai_feedback(
    predicted_disease,
    field_confirmed,
    actual_disease="",
    farmer_note=""
):

    record = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted Disease": predicted_disease,
        "Field Confirmed": field_confirmed,
        "Actual Disease": actual_disease,
        "Farmer Note": farmer_note
    }

    new_df = pd.DataFrame([record])

    try:
        old_df = pd.read_csv(FEEDBACK_FILE)
        new_df = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )
    except FileNotFoundError:
        pass

    new_df.to_csv(
        FEEDBACK_FILE,
        index=False
    )

    return "✅ Field confirmation saved for AI improvement."
# =========================================================
# INTEGRATED DEPLOYMENT UI
# =========================================================
# =========================================================
# INTEGRATED DEPLOYMENT UI
# =========================================================


# =========================================================
# OFFICER DASHBOARD
# =========================================================

def officer_dashboard():

    df = get_reports()

    if df.empty:
        return (
            "## 📊 Agriculture Officer Dashboard\n\n"
            "⚠️ No farmer reports available.",
            pd.DataFrame(),
            "<p>📍 No location data available.</p>"
        )

    total = len(df)

    high = len(
        df[
            df["Risk_Level"]
            .astype(str)
            .str.contains("HIGH", na=False)
        ]
    )

    medium = len(
        df[
            df["Risk_Level"]
            .astype(str)
            .str.contains("MEDIUM", na=False)
        ]
    )

    low = len(
        df[
            df["Risk_Level"]
            .astype(str)
            .str.contains("LOW", na=False)
        ]
    )

    disease_counts = df["Disease"].value_counts()

    most_disease = (
        disease_counts.index[0]
        if not disease_counts.empty
        else "None"
    )

    map_df = df.dropna(
        subset=["Latitude", "Longitude"]
    ).copy()

    hotspot_count = 0

    if len(map_df) >= 2:

        coords = map_df[
            ["Latitude", "Longitude"]
        ].values

        clustering = DBSCAN(
            eps=0.02,
            min_samples=2
        ).fit(coords)

        map_df["Hotspot_ID"] = clustering.labels_

        hotspot_count = len(
            set(clustering.labels_) - {-1}
        )

    else:
        map_df["Hotspot_ID"] = -1

    officer_map = folium.Map(
        location=[11.3410, 77.7172],
        zoom_start=7
    )

    for _, row in map_df.iterrows():

        folium.Marker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            popup=(
                f"Disease: {row['Disease']}<br>"
                f"Risk: {row['Risk_Level']}<br>"
                f"Pest: {row['Pest']}"
            )
        ).add_to(officer_map)

    summary = f"""
## 📊 Agriculture Officer Dashboard

**Total Farmer Reports:** {total}

🔴 **High Risk:** {high}

🟡 **Medium Risk:** {medium}

🟢 **Low Risk:** {low}

🦠 **Most Reported Disease:** {most_disease}

📍 **Detected Hotspots:** {hotspot_count}
"""

    return (
        summary,
        df,
        officer_map._repr_html_()
    )


# =========================================================
# FARMER LOCATION
# =========================================================

location_js = """
async () => {
    return await new Promise((resolve) => {

        if (!navigator.geolocation) {
            alert("Location is not supported by this browser.");
            resolve(["", ""]);
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {

                resolve([
                    position.coords.latitude.toFixed(6),
                    position.coords.longitude.toFixed(6)
                ]);

            },

            () => {
                alert("Location permission denied.");
                resolve(["", ""]);
            }

        );

    });
}
"""


# =========================================================
# AI FEEDBACK DISPLAY
# =========================================================

def load_ai_feedback():

    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame()

    return pd.read_csv(FEEDBACK_FILE)


# =========================================================
# MAIN GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    title="SIH Crop Health Management System"
) as app:

    gr.Markdown(
        """
# 🌾 AI-Powered Crop Disease & Pest Management System

### Early Detection • Pest Detection • Weather Risk • Farmer Advisory
"""
    )


    # =====================================================
    # FARMER DASHBOARD
    # =====================================================

    with gr.Tab("👨‍🌾 Farmer Dashboard"):

        gr.Markdown(
            """
## 🌱 Farmer Crop Health Analysis

Upload a crop image and allow location access.
The system will analyse disease, pest, weather risk
and provide an agricultural advisory.
"""
        )

        with gr.Row():

            with gr.Column():

                farmer_language = gr.Dropdown(
                    choices=language_options,
                    value="English",
                    label="🌐 Preferred Language"
                )

                crop_image = gr.Image(
                    type="pil",
                    label="📷 Upload Crop Image"
                )

            with gr.Column():

                latitude = gr.Textbox(
                    label="📍 Latitude"
                )

                longitude = gr.Textbox(
                    label="📍 Longitude"
                )

                location_button = gr.Button(
                    "📍 Allow My Location"
                )

                location_button.click(
                    fn=None,
                    inputs=[],
                    outputs=[
                        latitude,
                        longitude
                    ],
                    js=location_js
                )


        analyze_button = gr.Button(
            "🔍 Analyze Crop",
            variant="primary"
        )


        analysis_output = gr.Markdown(
            label="🦠 AI Analysis"
        )

        advisory_output = gr.Markdown(
            label="🌱 Farmer Advisory"
        )

        validation_output = gr.Markdown(
            label="🧑‍🔬 Expert Validation"
        )

        alert_output = gr.Markdown(
            label="🚨 Farmer Alert"
        )

        risk_output = gr.Markdown(
            label="⚠️ Overall Crop Risk"
        )

        map_output = gr.HTML(
            label="📍 Farmer Location"
        )

        status_output = gr.Markdown(
            label="📊 System Status"
        )


        analyze_button.click(
            fn=connected_analysis,
            inputs=[
                crop_image,
                farmer_language,
                latitude,
                longitude
            ],
            outputs=[
                analysis_output,
                advisory_output,
                validation_output,
                alert_output,
                risk_output,
                map_output,
                status_output
            ]
        )


        # =================================================
        # WEATHER FORECAST
        # =================================================

        gr.Markdown("---")

        gr.Markdown(
            "## 🌦️ 3-Day Weather Forecast"
        )

        forecast_button = gr.Button(
            "🌦️ Get 3-Day Forecast"
        )

        forecast_output = gr.Markdown()

        forecast_button.click(
            fn=get_location_weather_forecast,
            inputs=[
                latitude,
                longitude
            ],
            outputs=forecast_output
        )


        # =================================================
        # IPM
        # =================================================

        gr.Markdown("---")

        gr.Markdown(
            "## 🌱 IPM & Safe Input Recommendation"
        )

        ipm_button = gr.Button(
            "🌱 Get IPM Recommendation"
        )

        ipm_output = gr.Markdown()

        ipm_button.click(
            fn=extract_disease_for_ipm,
            inputs=analysis_output,
            outputs=ipm_output
        )


        # =================================================
        # EXPERT / LAB REFERRAL
        # =================================================

        gr.Markdown("---")

        gr.Markdown(
            "## 🧑‍🔬 Expert / Laboratory Referral"
        )

        referral_button = gr.Button(
            "🧑‍🔬 Check Referral Requirement"
        )

        referral_output = gr.Markdown()

        referral_button.click(
            fn=referral_from_analysis,
            inputs=[
                analysis_output,
                risk_output
            ],
            outputs=referral_output
        )


    # =====================================================
    # OFFICER DASHBOARD
    # =====================================================

    with gr.Tab("📊 Officer Dashboard"):

        gr.Markdown(
            """
## 📊 Agriculture Officer Dashboard

Monitor farmer reports, crop risks and geographical hotspots.
"""
        )

        officer_summary = gr.Markdown()

        officer_table = gr.Dataframe(
            label="📋 Farmer Reports",
            interactive=False
        )

        officer_map = gr.HTML(
            label="📍 Crop Risk Hotspot Map"
        )

        refresh_officer = gr.Button(
            "🔄 Refresh Officer Dashboard",
            variant="primary"
        )

        refresh_officer.click(
            fn=officer_dashboard,
            inputs=[],
            outputs=[
                officer_summary,
                officer_table,
                officer_map
            ]
        )


    # =====================================================
    # FOLLOW-UP MONITORING
    # =====================================================

    with gr.Tab("🔄 Follow-Up Monitoring"):

        gr.Markdown(
            """
## 🔄 Farmer Follow-Up Monitoring

Select a previous farmer report and record the current crop condition.
"""
        )

        followup_report_number = gr.Textbox(
            label="📄 Report Number",
            placeholder="Example: 1"
        )

        load_report_button = gr.Button(
            "🔍 Load Previous Report"
        )

        previous_disease = gr.Textbox(
            label="🦠 Previous Disease",
            interactive=False
        )

        previous_confidence = gr.Textbox(
            label="📊 Disease Confidence",
            interactive=False
        )

        previous_risk = gr.Textbox(
            label="⚠️ Original Risk Level",
            interactive=False
        )

        previous_score = gr.Textbox(
            label="📊 Original Risk Score",
            interactive=False
        )

        previous_latitude = gr.Textbox(
            label="📍 Latitude",
            interactive=False
        )

        previous_longitude = gr.Textbox(
            label="📍 Longitude",
            interactive=False
        )

        previous_language = gr.Textbox(
            label="🌐 Language",
            interactive=False
        )

        load_report_button.click(
            fn=load_previous_report,
            inputs=followup_report_number,
            outputs=[
                previous_disease,
                previous_confidence,
                previous_risk,
                previous_score,
                previous_latitude,
                previous_longitude,
                previous_language
            ]
        )


        followup_condition = gr.Dropdown(
            choices=[
                "Improving",
                "No Change",
                "Worsening",
                "Recovered"
            ],
            label="🌱 Current Crop Condition"
        )

        followup_observation = gr.Textbox(
            label="📝 Farmer Observation",
            placeholder="Describe the current crop condition..."
        )

        followup_button = gr.Button(
            "💾 Save Follow-Up",
            variant="primary"
        )

        followup_result = gr.Markdown()

        followup_button.click(
            fn=save_followup_with_feedback,
            inputs=[
                followup_report_number,
                followup_condition,
                followup_observation
            ],
            outputs=followup_result
        )


    # =====================================================
    # FIELD CONFIRMATION
    # =====================================================

    with gr.Tab("✅ Field Confirmation"):

        gr.Markdown(
            """
## ✅ Field Confirmation

Record whether the AI prediction was confirmed in the field.
"""
        )

        confirmation_report_number = gr.Textbox(
            label="📄 Report Number",
            placeholder="Example: 1"
        )

        field_confirmed = gr.Dropdown(
            choices=[
                "Yes",
                "No",
                "Uncertain"
            ],
            label="🔍 Was the AI Prediction Confirmed?"
        )

        actual_disease = gr.Textbox(
            label="🦠 Actual Disease",
            placeholder="Enter confirmed disease if known"
        )

        farmer_note = gr.Textbox(
            label="📝 Farmer / Expert Note"
        )

        confirmation_button = gr.Button(
            "💾 Save Field Confirmation",
            variant="primary"
        )

        confirmation_result = gr.Markdown()

        confirmation_button.click(
            fn=record_followup_confirmation,
            inputs=[
                confirmation_report_number,
                field_confirmed,
                actual_disease,
                farmer_note
            ],
            outputs=confirmation_result
        )


    # =====================================================
    # FOLLOW-UP HISTORY
    # =====================================================

    with gr.Tab("📚 Follow-Up History"):

        gr.Markdown(
            """
## 📚 Follow-Up History
"""
        )

        history_button = gr.Button(
            "🔄 Load Follow-Up History"
        )

        history_output = gr.Dataframe(
            interactive=False
        )

        history_button.click(
            fn=load_followup_history,
            inputs=[],
            outputs=history_output
        )


    # =====================================================
    # AI FEEDBACK
    # =====================================================

    with gr.Tab("🤖 AI Feedback"):

        gr.Markdown(
            """
## 🤖 AI Field Confirmation Records

These records are stored for future model improvement.
"""
        )

        feedback_button = gr.Button(
            "🔄 Load AI Feedback"
        )

        feedback_output = gr.Dataframe(
            interactive=False
        )

        feedback_button.click(
            fn=load_ai_feedback,
            inputs=[],
            outputs=feedback_output
        )


# =========================================================
# RENDER DEPLOYMENT
# =========================================================

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "7860"))
)
         
    
