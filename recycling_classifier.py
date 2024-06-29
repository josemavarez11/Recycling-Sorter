import cv2 
import torch 
from torchvision import transforms 
from PIL import Image 

class SimpleModel(torch.nn.Module): 
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1)
        self.fc1 = torch.nn.Linear(16*222*222, 3)  

    def forward(self, x):
        x = torch.relu(self.conv1(x)) 
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = SimpleModel()

state_dict = torch.load('best.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False) 
model.eval() 

categories = ['Plastic', 'Aluminum', 'Glass', 'Other']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def preprocess_frame(frame):
    img = Image.fromarray(frame) 
    img = transform(img)
    img = img.unsqueeze(0)  
    return img

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    preprocessed_frame = preprocess_frame(frame)

    with torch.no_grad():
        outputs = model(preprocessed_frame)
    _, predicted = torch.max(outputs, 1)
    class_idx = predicted.item() 
    class_name = categories[class_idx]

    cv2.putText(frame, f'Material: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Recycling Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()