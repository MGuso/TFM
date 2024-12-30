import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms as transforms
import numpy as np
import copy
THRESHOLD = 0.4

def get_default_device():
    """Seleccionar GPU si está disponible, de lo contrario usar CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Mover tensor(es) al dispositivo seleccionado"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def F_score(output: list, label: list, threshold: float=THRESHOLD, beta: float=1.0) -> float:
    '''
        Retorna el F-score del modelo
        Entrada:
            output: array de salidas (outputs)
            label: array de etiquetas (labels)
            threshold: [opcional] float -> considerar la probabilidad de salida si está por encima del umbral
            beta: [opcional] float
        Salida:
            float -> F-score
    '''

    # Determinar las predicciones considerando el umbral
    prob = output > threshold
    label = label > threshold

    # Calcular verdaderos positivos (True Positives - TP)
    TP = (prob & label).sum(1).float()

    # Calcular verdaderos negativos (True Negatives - TN)
    TN = ((~prob) & (~label)).sum(1).float()

    # Calcular falsos positivos (False Positives - FP)
    FP = (prob & (~label)).sum(1).float()

    # Calcular falsos negativos (False Negatives - FN)
    FN = ((~prob) & label).sum(1).float()

    # Calcular precisión (precision)
    precision = torch.mean(TP / (TP + FP + 1e-12))

    # Calcular recall (sensibilidad)
    recall = torch.mean(TP / (TP + FN + 1e-12))

    # Calcular el F-score utilizando la fórmula con el parámetro beta
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)

    # Retornar el promedio del F-score
    return F2.mean(0)

class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch
        out = self(images)                          # Generar predicciones
        loss = F.binary_cross_entropy(out, targets) # Calcular pérdida (loss)
        score = F_score(out, targets)               # Calcular la métrica F-score
        return {'loss': loss, 'score': score.detach()}

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)                           # Generar predicciones
        loss = F.binary_cross_entropy(out, targets)  # Calcular pérdida (loss)
        score = F_score(out, targets)                # Calcular la métrica F-score
        return {'val_loss': loss.detach(), 'val_score': score.detach() }

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combinar las pérdidas (losses) de los lotes
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()  # Combinar las métricas F-score de los lotes
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def get_metrics_epoch_end(self, outputs, validation=True):
        if validation:
            loss_ = 'val_loss'
            score_ = 'val_score'
        else:
            loss_ = 'loss'
            score_ = 'score'

        # Combinar las pérdidas (losses) de los lotes
        batch_losses = [x[f'{loss_}'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()

        # Combinar las métricas F-score de los lotes
        batch_scores = [x[f'{score_}']
                        for x in outputs]
        epoch_scores = torch.stack(batch_scores).mean()

        return {f'{loss_}': epoch_loss.item(), f'{score_}': epoch_scores.item()}

    def epoch_end(self, epoch, result, epochs):
        # Mostrar el resumen al final de cada época
        print(f"Epoch: {epoch+1}/{epochs} -> last_lr: {result['lrs'][-1]:.4f}, train_loss: {result['loss']:.4f}, train_score: {result['score']:.4f}, val_loss: {result['val_loss']:.4f}, val_score: {result['val_score']:.4f}")

class GeneralModel(MultilabelImageClassificationBase):

    @staticmethod
    def get_sequential(num_ftrs):
        # Definir las capas secuenciales de la red neuronal
        linear_layers = nn.Sequential(
                nn.BatchNorm1d(num_features=num_ftrs),  # Normalización batch para las características de entrada
                nn.Linear(num_ftrs, 512),              # Capa totalmente conectada con 512 neuronas
                nn.ReLU(),                             # Activación ReLU
                nn.BatchNorm1d(512),                   # Normalización batch para las 512 características
                nn.Linear(512, 128),                   # Capa totalmente conectada con 128 neuronas
                nn.ReLU(),                             # Activación ReLU
                nn.BatchNorm1d(num_features=128),      # Normalización batch para las 128 características
                nn.Dropout(0.4),                       # Regularización dropout con una probabilidad de 0.4
                nn.Linear(128, 2),                     # Capa totalmente conectada con 2 salidas (etiquetas)
            )
        return linear_layers

    def __init__(self, model_name=None, model=None, input_size=None):
        super().__init__()

        # Usar un modelo preentrenado
        self.model_name = model_name
        self.model = copy.deepcopy(model)
        self.IS = input_size

        # Reemplazar la última capa del modelo preentrenado
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = GeneralModel.get_sequential(self.num_ftrs)

    def forward(self, xb):
        # Pasar los datos a través del modelo y aplicar una función sigmoide
        return torch.sigmoid(self.model(xb))

    def freeze(self):
        # Congelar (freeze) las capas residuales del modelo
        for param in self.model.parameters():
            param.require_grad = False

        # Asegurarse de que las capas completamente conectadas no estén congeladas
        for param in self.model._fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Descongelar (unfreeze) todas las capas del modelo
        for param in self.model.parameters():
            param.require_grad = True

    def __repr__(self):
        # Retornar una representación en cadena del modelo
        return f"{self.model}"

    def __str__(self):
        # Generar un resumen del modelo
        summary(self.model, (3, self.IS, self.IS))
        text_ = \
        f'''
            Model Name: {self.model_name}
            FC Layer input: {self.num_ftrs}
        '''
        return text_
