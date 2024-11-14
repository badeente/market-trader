import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class TradingModel(nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()
        
        # CNN for processing screenshots
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers for decision making
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Output layers
        self.decision_head = nn.Linear(32, 2)  # Red/Green decision
        self.minutes_head = nn.Linear(32, 1)   # Minutes prediction
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        decision = torch.softmax(self.decision_head(x), dim=1)
        minutes = torch.relu(self.minutes_head(x))  # Ensure positive minutes
        
        return decision, minutes

class TradingAgent:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradingModel().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.decision_criterion = nn.CrossEntropyLoss()
        self.minutes_criterion = nn.MSELoss()
        
    def preprocess_screenshot(self, screenshot_path):
        image = Image.open(screenshot_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)
    
    def predict(self, screenshot_path):
        self.model.eval()
        with torch.no_grad():
            image = self.preprocess_screenshot(screenshot_path)
            decision_probs, minutes = self.model(image)
            
            decision_idx = torch.argmax(decision_probs, dim=1).item()
            decision = "green" if decision_idx == 1 else "red"
            minutes = int(minutes.item())
            
            return decision, minutes
    
    def train_step(self, screenshot_path, reward):
        self.model.train()
        image = self.preprocess_screenshot(screenshot_path)
        
        # Forward pass
        decision_probs, minutes = self.model(image)
        
        # Calculate loss based on reward
        # This is a simple example - you might want to modify this based on your specific needs
        loss = -torch.log(decision_probs) * reward
        loss = loss.mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
