"""
Prosty serwer Flask do klasyfikacji śmieci w czasie rzeczywistym.
Używa wytrenowanego modelu ResNet50 do predykcji typu odpadu z obrazu z kamery.
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import json
import os

app = Flask(__name__)

# --- Definicja modelu ResNet (identyczna jak w treningu) ---
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Używamy pretrenowanego ResNet50 z transfer learning
        self.network = models.resnet50(pretrained=True)
        # Zamieniamy ostatnią warstwę na naszą liczbę klas
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
    
    def forward(self, xb):
        return self.network(xb)

# --- Konfiguracja globalnych zmiennych ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Transformacje obrazu (identyczne jak podczas treningu)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_model():
    """Ładuje wytrenowany model ResNet50 z pliku."""
    global model
    model_path = 'pytorch_replica_model.pth'
    
    if os.path.exists(model_path):
        print("Ładowanie wytrenowanego modelu ResNet50...")
        model = ResNet(num_classes=6).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Tryb ewaluacji
        print("Model załadowany pomyślnie!")
    else:
        print(f"BŁĄD: Nie znaleziono modelu w ścieżce {model_path}")
        print("Upewnij się, że model został wytrenowany.")

def preprocess_image(image_data):
    """
    Przetwarza obraz z base64 do formatu wymaganego przez model.
    """
    try:
        # Dekodowanie base64
        image_data = image_data.split(',')[1]  # Usuń prefix data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Konwersja do PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Aplikacja transformacji
        image_tensor = transform(image).unsqueeze(0)  # Dodaj wymiar batch
        
        return image_tensor.to(device)
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        return None

@app.route('/')
def index():
    """Strona główna aplikacji."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint do predykcji typu odpadu na podstawie obrazu z kamery.
    """
    if model is None:
        return jsonify({'error': 'Model nie został załadowany'}), 500
    
    try:
        # Pobierz obraz z requesta
        data = request.get_json()
        image_data = data['image']
        
        # Przetwórz obraz
        image_tensor = preprocess_image(image_data)
        if image_tensor is None:
            return jsonify({'error': 'Błąd przetwarzania obrazu'}), 400
        
        # Predykcja
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Znajdź klasę z najwyższym prawdopodobieństwem
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
            
            predicted_class = class_names[predicted_class_idx]
            
            # Przygotuj wszystkie prawdopodobieństwa
            all_predictions = {}
            for i, class_name in enumerate(class_names):
                all_predictions[class_name] = round(probabilities[i].item() * 100, 1)
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 1),
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        print(f"Błąd podczas predykcji: {e}")
        return jsonify({'error': 'Błąd podczas predykcji'}), 500

if __name__ == '__main__':
    # Załaduj model przy starcie
    load_model()
    
    # Uruchom serwer
    print("Uruchamianie serwera Flask...")
    print("Aplikacja dostępna pod adresem: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
