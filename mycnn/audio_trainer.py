import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, device=None, learning_rate=0.001):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=2, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            inputs = batch["input"].to(self.device)
            labels = batch["label"].to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        accuracy = accuracy_score(all_labels, all_preds)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc, preds, labels = self.validate_epoch()
            
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                print(f"Learning rate reduced from {prev_lr} to {current_lr}")
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        return preds, labels
