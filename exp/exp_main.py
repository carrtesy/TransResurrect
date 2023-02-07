from data_provider.data_factory import data_provider
from models import Transformer, TransformerLDec
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim

import os
import copy

import warnings
import matplotlib.pyplot as plt
import numpy as np

import wandb
warnings.filterwarnings('ignore')

import loss_landscapes
import loss_landscapes.metrics

class Exp_Main:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self._prepare_data()
        self._select_optimizer()
        self._select_criterion()
        wandb.watch(self.model, log="all", log_freq=100)


    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'TransformerLDec': TransformerLDec,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        self.logger.info(model)
        return model


    def _prepare_data(self):
        self.train_data, self.train_loader = data_provider(self.args, "train")
        self.val_data, self.val_loader = data_provider(self.args, "val")
        self.test_data, self.test_loader = data_provider(self.args, "test")


    def _select_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )


    def _select_criterion(self):
        self.criterion = nn.MSELoss()


    def train(self):
        best_val_loss = 1e9-1
        train_hist, val_hist, test_hist = [], [], []
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self.logger.info(f"epoch {epoch+1}, iters: {i+1}, loss: {loss.item():.7f} ")
                loss.backward()
                self.optimizer.step()

            train_loss = np.average(train_loss)
            val_loss = self.infer(self.val_loader)
            test_loss = self.infer(self.test_loader)
            train_hist.append(train_loss)
            val_hist.append(val_loss)
            test_hist.append(test_loss)

            self.logger.info(
                f"Epoch: {epoch + 1} | "
                f"Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}, Test Loss: {test_loss:.7f}"
            )

            torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint_path, f"Epoch{epoch + 1}.pth"))
            if val_loss < best_val_loss:
                self.logger.info(f"saving best model @epoch {epoch+1}")
                torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint_path, 'best.pth'))
                best_val_loss = val_loss

            wandb.log({
              "train_loss": train_loss,
              "vali_loss": val_loss,
              "test_loss": test_loss,
            })

        best_model_path = os.path.join(self.args.checkpoint_path, 'best.pth')
        self.logger.info(f"best model from: {best_model_path}")
        self.model.load_state_dict(torch.load(best_model_path))
        self.plot_training_graph(train_hist, val_hist, test_hist)
        self.visualize_loss_landscape()


    @torch.no_grad()
    def infer(self, dataloader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = self.criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    @torch.no_grad()
    def test(self):
        # load
        best_model_path = os.path.join(self.args.checkpoint_path, 'best.pth')
        self.logger.info(f"best model from: {best_model_path}")
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        # loss landscape
        self.visualize_loss_landscape()

        # test
        preds, trues = [], []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            outputs = self.model(batch_x)
            outputs = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()
            batch_y = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()
            preds.append(outputs)
            trues.append(batch_y)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        self.logger.info(f'test shape:{preds.shape} {trues.shape}')

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        wandb.log({
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "mspe": mspe,
        })

        # save
        np.save(os.path.join(self.args.output_path, 'pred.npy'), preds)
        np.save(os.path.join(self.args.output_path, 'true.npy'), trues)


    def plot_training_graph(self, train_hist=None, val_hist=None, test_hist=None):
        plt.figure(figsize=(20, 6))
        plt.title(f'Training History | {self.args.exp_id}')
        if train_hist:
            plt.plot(range(1, len(train_hist)+1), train_hist, label="train_loss")
        if val_hist:
            plt.plot(range(1, len(val_hist)+1), val_hist, label="val_loss")
        if test_hist:
            plt.plot(range(1, len(test_hist)+1), test_hist, label="test_loss")
        plt.legend()
        plt.savefig(os.path.join(self.args.plot_path, f"training_hist.png"))


    @torch.no_grad()
    def visualize_loss_landscape(self, STEPS=40):
        # loss landscape
        batch_x, batch_y, _, _ = iter(self.train_loader).__next__()
        criterion = self.criterion
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].float().to(self.device)
        ll_metric = loss_landscapes.metrics.Loss(criterion, batch_x, batch_y)
        model_final = copy.deepcopy(self.model)
        loss_data_fin = loss_landscapes.random_plane(
            model_final, ll_metric, 10, STEPS, normalization='filter', deepcopy_model=True)

        # 2D
        plt.figure(figsize=(20, 6))
        plt.contour(loss_data_fin, levels=50)
        plt.title(f'Loss Contours around Trained Model | {self.args.exp_id}')
        plt.savefig(os.path.join(self.args.plot_path, f"loss_landscape_2D.png"))

        # 3D
        fig = plt.figure(figsize=(20, 6))
        ax = plt.axes(projection='3d')
        X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
        Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
        ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title(f'Surface Plot of Loss Landscape | {self.args.exp_id}')
        fig.savefig(os.path.join(self.args.plot_path, f"loss_landscape_3D.png"))
