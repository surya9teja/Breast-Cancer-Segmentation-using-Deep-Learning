import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

class Model_Training():
  def __init__(self, model, train_loader, test_loader, optimizer, device, loss_fn):
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.model = model
    self.optimizer = optimizer
    self.device = device
    self.loss_fn = loss_fn
  
  def train(self):
    epoch_loss = 0.0
    self.model.train()
    for x, y in self.train_loader:
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32)

        self.optimizer.zero_grad()
        y_prediction = self.model(x)
        loss = self.loss_fn(y_prediction,y)
        loss.backward()
        self.optimizer.step()
        epoch_loss +=loss.item()
    epoch_loss = epoch_loss/len(self.train_loader)
    return epoch_loss
  def evaluate(self):
    epoch_loss = 0.0

    self.model.eval()
    with torch.no_grad():
        for x, y in self.test_loader:
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)

            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(self.test_loader)
    return epoch_loss