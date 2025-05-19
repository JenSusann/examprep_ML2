# Multimodal AI – Week 10 Exercises (VLM, Object Detection, Text-Image-Audio)

## Setup

### 1. Repository klonen

```bash
git clone https://github.com/zhaw-iwi/MultimodalInteraction_ObjDet.git
cd MultimodalInteraction_ObjDet
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
pip install python-dotenv openai transformers torch torchvision matplotlib opencv-python pyttsx3
```

### 3. .env Datei erstellen

Erstelle eine `.env` Datei im Projektverzeichnis mit folgendem Inhalt (ersetze durch deine echten Keys):

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_gemini_key
```

Dann in Python laden:

```python
from dotenv import load_dotenv
import os
load_dotenv()
```

---

## Exercise 1 – VLM Basics

Notebook: `VLM_Basics.ipynb`

1. Öffne das Notebook in Google Colab oder lokal.
2. Führe jede Zelle aus, installiere ggf. fehlende Bibliotheken.
3. Verwende `.env` Datei zur API-Key-Nutzung.
4. Beobachte die Ausgaben und stelle sicher, dass du ein Bild erfolgreich analysieren kannst.

---

## Exercise 2 – Detect an Object on a Table

### Ziel: Finde die Tasse auf dem Bild `table_scene.jpg` mit YOLO, GPT-4 und Gemini.

#### 1. YOLOv5 verwenden

```python
import torch
from PIL import Image
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img_path = 'images/table_scene.jpg'
results = model(img_path)
results.print()
results.show()
```

#### 2. GPT-4 Vision API verwenden (OpenAI)

```python
import openai
import base64
import os
from dotenv import load_dotenv

load_dotenv()

with open("images/table_scene.jpg", "rb") as image_file:
    img_base64 = base64.b64encode(image_file.read()).decode()

response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Wo befindet sich die Tasse auf diesem Bild?"},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
        ]}
    ],
    max_tokens=300
)
print(response['choices'][0]['message']['content'])
```

#### 3. Gemini Vision verwenden

```python
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')

with open("images/table_scene.jpg", "rb") as f:
    img = f.read()

response = model.generate_content(
    ["Wo ist die Tasse auf diesem Bild?", img],
    stream=False
)
print(response.text)
```

#### 4. Vergleich der Methoden

| Methode  | Genauigkeit | Beschreibung                   |
|----------|-------------|--------------------------------|
| YOLO     | Hoch        | Liefert Bounding Box           |
| GPT-4    | Mittel      | Textuelle Beschreibung         |
| Gemini   | Mittel      | Textuell, manchmal ungenauer   |

---

## Exercise 3 – Text-Image-Audio Triage-Szene

### Ziel: Analysiere das Bild `accident_scene.jpg` und bestimme, wer Hilfe benötigt. Nutze danach ein TTS-System.

#### 1. GPT-4 Vision – Strukturierte Extraktion

```python
import openai
import base64
import os
from dotenv import load_dotenv

load_dotenv()

with open("images/accident_scene.jpg", "rb") as image_file:
    img_base64 = base64.b64encode(image_file.read()).decode()

response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Bitte analysiere diese Unfallszene. Wer benötigt sofortige medizinische Hilfe? Gib das Ergebnis strukturiert aus."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
        ]}
    ],
    max_tokens=500
)
print(response['choices'][0]['message']['content'])
```

#### 2. Strukturierte Ausgabe weitergeben an TTS-Modell

```python
import pyttsx3

triage_result = '''
Person A: bewusstlos, starke Blutung – braucht sofortige Hilfe.
Person B: wach, leichte Schnittwunde – stabile Situation.
'''

engine = pyttsx3.init()
engine.say(triage_result)
engine.runAndWait()
```

---
