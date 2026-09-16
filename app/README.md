# 🗂️ Web Application - AI Waste Classifier

A simple web application for real-time waste classification using the device camera and a trained ResNet50 model.

## 🚀 Features

- **Real-time camera access** - works on both phones and computers
- **Live image classification** - uses a ResNet50 model trained on 6 waste categories
- **Responsive interface** - optimized for mobile devices
- **Real-time results** - displays probabilities for all classes

## 🗃️ Classified Waste Types

1. **Cardboard** 📦
2. **Glass** 🥃
3. **Metal** ⚙️
4. **Paper** 📄
5. **Plastic** 🩴
6. **Other waste** 🗑️

## 📋 Requirements

- Python 3.8+
- Trained ResNet50 model (`pytorch_replica_model.pth`)
- Webcam or smartphone with a browser

## ⚙️ Installation

1. Navigate to the app directory:

```bash
cd app
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Make sure the trained model is located at:

```
../models/pytorch_resnet/pytorch_replica_model.pth
```

## 🎯 Run the Application

1. Start the Flask server:

```bash
python app.py
```

2. Open a browser and go to:

```
http://localhost:5000
```

3. Click "Enable Camera" and allow access
4. Show waste items to the camera and click "Classify Waste"

## 📱 Using on Mobile

1. Ensure your phone is on the same network as the server machine
2. Find your computer’s IP address (e.g., 192.168.1.100)
3. Open on your phone’s browser: `http://IP_ADDRESS:5000`
4. The app will automatically use the phone’s rear camera

## 🔧 Technical Stack

- **Backend**: Flask + PyTorch
- **Frontend**: HTML5 + JavaScript + CSS3
- **AI Model**: ResNet50 with transfer learning
- **Communication**: REST API (JSON)
- **Image format**: JPEG via base64

## 📊 Model Performance

ResNet50 achieved:

- **Validation accuracy**: 95.2%
- **Response time**: < 1 second
- **Supported resolution**: 256x256px (auto-scaling)

## 🛠️ Customization

To modify classes or the model, edit:

- `class_names` in `app.py` - class names
- `classTranslations` in `index.html` - translations
- Model path in the `load_model()` function

## 🔍 Troubleshooting

**Camera not working:**

- Check browser camera permissions
- Use HTTPS in production
- Ensure no other app is using the camera

**Model not loading:**

- Verify model file exists
- Check the file path
- Ensure PyTorch supports CUDA (if using GPU)

**Prediction errors:**

- Ensure the image is clear
- Make sure waste items are visible in the frame
- Try better lighting

