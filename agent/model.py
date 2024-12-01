import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import os
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.ln1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = nn.ReLU()(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(residual)
        out = nn.ReLU()(out)
        return out

class TradingModel(nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Convert BatchNorm to LayerNorm in ResNet
        def convert_bn_to_ln(module):
            if isinstance(module, nn.BatchNorm2d):
                num_features = module.num_features
                return nn.GroupNorm(1, num_features)
            for name, child in module.named_children():
                module._modules[name] = convert_bn_to_ln(child)
            return module
        
        # Convert BatchNorm layers to LayerNorm
        self.resnet = convert_bn_to_ln(self.resnet)
        
        # Freeze the early layers
        for param in list(self.resnet.parameters())[:-2]:
            param.requires_grad = False
            
        # Get number of features from ResNet
        num_features = self.resnet.fc.in_features
        
        # Enhanced decision making layers with residual connections
        self.decision_layers = nn.Sequential(
            ResidualBlock(num_features, 512),
            nn.Dropout(0.5),
            ResidualBlock(512, 256),
            nn.Dropout(0.3),
            ResidualBlock(256, 128),
            nn.Dropout(0.2),
            ResidualBlock(128, 64),
            nn.LayerNorm(64)
        )
        
        # Output layers
        self.action_head = nn.Linear(64, 2)  # 2-class prediction (0=long, 1=short)
        self.action_confidence = nn.Linear(64, 1)  # Confidence score for taking action
        
        # Stop-loss and take-profit heads with bias initialization
        self.long_sl_head = nn.Linear(64, 1)
        self.long_tp_head = nn.Linear(64, 1)
        self.short_sl_head = nn.Linear(64, 1)
        self.short_tp_head = nn.Linear(64, 1)
        
        # Initialize biases for more aggressive trading
        nn.init.constant_(self.action_confidence.bias, 0.5)  # Start with 0.5 confidence
        nn.init.constant_(self.long_sl_head.bias, -1.0)  # Smaller stop-loss
        nn.init.constant_(self.long_tp_head.bias, 1.0)   # Larger take-profit
        nn.init.constant_(self.short_sl_head.bias, -1.0)
        nn.init.constant_(self.short_tp_head.bias, 1.0)
        
    def forward(self, x):
        # ResNet feature extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Decision making
        features = self.decision_layers(x)
        
        # Output predictions
        action_probs = torch.softmax(self.action_head(features), dim=1)  # Long vs Short preference
        confidence = torch.sigmoid(self.action_confidence(features))  # Confidence in taking any position
        
        # Stop-loss and take-profit predictions with adjusted ranges
        long_sl = torch.sigmoid(self.long_sl_head(features)) * 0.05  # Max 5% stop-loss
        long_tp = torch.sigmoid(self.long_tp_head(features)) * 0.15  # Max 15% take-profit
        short_sl = torch.sigmoid(self.short_sl_head(features)) * 0.05
        short_tp = torch.sigmoid(self.short_tp_head(features)) * 0.15
        
        return action_probs, confidence, long_sl, long_tp, short_sl, short_tp

class TradingAgent:
    def __init__(self, model_path=None, learning_rate=0.001, max_risk_per_trade=0.02, confidence_threshold=0.3):  # Lowered threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = TradingModel().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Risk management parameters
        self.max_risk_per_trade = max_risk_per_trade
        self.confidence_threshold = confidence_threshold
        
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['model_state_dict'])
                    if 'max_risk_per_trade' in state_dict:
                        self.max_risk_per_trade = state_dict['max_risk_per_trade']
                    if 'confidence_threshold' in state_dict:
                        self.confidence_threshold = state_dict['confidence_threshold']
                else:
                    self.model.load_state_dict(state_dict)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Track best model
        self.best_loss = float('inf')
        self.last_lr = learning_rate
    
    def set_max_risk_per_trade(self, risk_percentage):
        """Set the maximum risk percentage per trade (between 0 and 1)"""
        if 0 < risk_percentage <= 1:
            self.max_risk_per_trade = risk_percentage
            logger.info(f"Max risk per trade set to {risk_percentage*100}%")
        else:
            raise ValueError("Risk percentage must be between 0 and 1")
    
    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for taking positions (between 0 and 1)"""
        if 0 < threshold <= 1:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold set to {threshold*100}%")
        else:
            raise ValueError("Confidence threshold must be between 0 and 1")
    
    def _log_lr_change(self):
        """Method to log learning rate changes"""
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr != self.last_lr:
            logger.info(f"Learning rate adjusted to: {current_lr}")
            self.last_lr = current_lr
    
    def get_current_lr(self):
        """Method to get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
        
    def preprocess_screenshot(self, screenshot_input):
        """Process either a file path or binary image data"""
        try:
            if isinstance(screenshot_input, (str, bytes)):
                if isinstance(screenshot_input, str):
                    image = Image.open(screenshot_input).convert('RGB')
                else:
                    image = Image.open(BytesIO(screenshot_input)).convert('RGB')
                
                image = self.transform(image)
                return image.unsqueeze(0).to(self.device)
            else:
                raise ValueError("Input must be either a file path string or bytes containing image data")
        except Exception as e:
            logger.error(f"Error preprocessing screenshot: {e}")
            raise
    
    def predict(self, screenshot_input):
        """Make a prediction with proper error handling"""
        try:
            self.model.eval()  # Ensure model is in evaluation mode
            with torch.no_grad():
                image = self.preprocess_screenshot(screenshot_input)
                action_probs, confidence, long_sl, long_tp, short_sl, short_tp = self.model(image)
                
                action = torch.argmax(action_probs, dim=1).item()  # 0=long, 1=short
                action_confidence = torch.max(action_probs).item()  # Confidence in long vs short
                position_confidence = confidence.item()  # Confidence in taking any position
                
                # Only suggest position if confidence is high enough
                should_take_position = position_confidence >= self.confidence_threshold
                
                # Ensure minimum stop-loss and take-profit values
                min_sl = 0.01  # 1% minimum stop-loss
                min_tp = 0.02  # 2% minimum take-profit
                
                trade_params = {
                    'should_trade': should_take_position,
                    'action': action if should_take_position else None,  # None means no position
                    'action_confidence': action_confidence,
                    'position_confidence': position_confidence,
                    'long_stop_loss': max(float(long_sl.item() * 100), min_sl * 100),
                    'long_take_profit': max(float(long_tp.item() * 100), min_tp * 100),
                    'short_stop_loss': max(float(short_sl.item() * 100), min_sl * 100),
                    'short_take_profit': max(float(short_tp.item() * 100), min_tp * 100)
                }
                
                logger.debug(f"Prediction made: {trade_params}")
                return trade_params
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def train_step(self, screenshot_input, target_action):
        """Training step with improved error handling and logging"""
        try:
            self.model.train()  # Ensure model is in training mode
            image = self.preprocess_screenshot(screenshot_input)
            
            # Forward pass
            action_probs, _, _, _, _, _ = self.model(image)  # Ignore confidence and stop-loss/take-profit during training
            
            # Calculate loss
            target = torch.tensor([target_action], device=self.device)
            loss = self.criterion(action_probs, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_value = loss.item()
            
            # Update learning rate scheduler
            self.scheduler.step(loss_value)
            self._log_lr_change()
            
            # Track best model
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                return loss_value, True  # True indicates new best model
            
            return loss_value, False
            
        except Exception as e:
            logger.error(f"Error during training step: {e}")
            raise
    
    def save_model(self, path, is_best=False):
        """Save model with proper error handling"""
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_loss': self.best_loss,
                'last_lr': self.last_lr,
                'max_risk_per_trade': self.max_risk_per_trade,
                'confidence_threshold': self.confidence_threshold
            }, path)
            
            if is_best:
                best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_loss': self.best_loss,
                    'last_lr': self.last_lr,
                    'max_risk_per_trade': self.max_risk_per_trade,
                    'confidence_threshold': self.confidence_threshold
                }, best_path)
                
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
