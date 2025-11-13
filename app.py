import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import io

# Define the CustomCNN class (same as before)
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # Increased starting filters
        self.bn1 = nn.BatchNorm2d(32)  # Add BN
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224->112

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112->56

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56->28

        # Add two more conv layers for depth
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28->14

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14->7

        # Fully connected layers (update input size: 256 channels * 7x7)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)  # Increased to 512
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))  # Add forward for new layers
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
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
    transforms.Resize((224, 224)),
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
