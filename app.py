import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import io

# Define the CustomCNN class (same as before)
class CustomCNN(torch.nn.Module):
    def __init__(self, num_classes=12):
        super(CustomCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 256)
        self.relu_fc1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model_path = 'best_model.pth'
model = CustomCNN(num_classes=12)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Class names
class_names = ['alan', 'arnur', 'arsen', 'asanali', 'bakytzhan', 'daniyal', 'nurtilek', 'said', 'yelarys', 'yerkebulan', 'zhan', 'zhangir']

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Futsal Player Classifier")
st.write("Upload an image to identify a futsal player.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]
            top3_probs, top3_indices = torch.topk(probs, 3)
            top3_probs = top3_probs.cpu().numpy() * 100
            top3_indices = top3_indices.cpu().numpy()

        # Display results
        st.subheader("Prediction Results")
        st.write("**Top Prediction**")
        st.write(f"Class: {class_names[top3_indices[0]]}, Confidence: {top3_probs[0]:.2f}%")
        st.write("**Top 3 Predictions**")
        for idx, prob in zip(top3_indices, top3_probs):
            st.write(f"Class: {class_names[idx]}, Confidence: {prob:.2f}%")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")