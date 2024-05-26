import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Placeholder model architecture
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1)
        self.fc1 = torch.nn.Linear(16*222*222, 3)  # Adjust according to your model architecture

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x

# Initialize the model
model = SimpleModel()

# Load pre-trained model weights
state_dict = torch.load('best.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Define the categories
categories = ['Plastic', 'Aluminum', 'Glass', 'Other']

# Define the transform to match the model's input requirements
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    img = Image.fromarray(frame)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Make prediction
    with torch.no_grad():
        outputs = model(preprocessed_frame)
    _, predicted = torch.max(outputs, 1)
    class_idx = predicted.item()
    class_name = categories[class_idx]

    # Display the result on the frame
    cv2.putText(frame, f'Material: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Recycling Classifier', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
