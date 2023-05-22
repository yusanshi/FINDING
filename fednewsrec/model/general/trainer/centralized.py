import torch
import torch.nn as nn

from fednewsrec.shared import args, device
from fednewsrec.test import evaluate


class CentralizedModel:

    def init_backprop(self):
        if args.loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=args.learning_rate)
        elif args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(),
                                             lr=args.learning_rate)
        else:
            raise NotImplementedError

    def backward(self, y_pred):
        if args.loss == 'BCE':
            y = torch.cat(
                (torch.ones(y_pred.size(0), 1),
                 torch.zeros(y_pred.size(0), args.negative_sampling_ratio)),
                dim=1).to(device)
        elif args.loss == 'CE':
            y = torch.zeros(y_pred.size(0)).long().to(device)
        else:
            raise NotImplementedError

        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, mode):
        return evaluate(self, mode)
