from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
from torch import optim
from torch.utils import data
from tqdm import tqdm_notebook as tqdm

from .utils import to_np


def get_train_test_dataset_and_loader(normalized_data, batch_size=64):
    """
    @returns
        dict with keys (train_loader, test_loader, train_dataset, test_dataset)
    """
    train_dataset = data.TensorDataset(
        torch.from_numpy(normalized_data["params_train_nk"]).float(),
        torch.from_numpy(normalized_data["vol_train_nf"]).float())
    test_dataset  = data.TensorDataset(
        torch.from_numpy(normalized_data["params_test_Nk"]).float(),
        torch.from_numpy(normalized_data["vol_test_Nf"]).float())

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = data.DataLoader(test_dataset,  batch_size=1,          shuffle=False, drop_last=False)

    return dict(train_loader=train_loader, test_loader=test_loader,
                train_dataset=train_dataset, test_dataset=test_dataset)

def get_torch_model(dim_input, dim_hidden = 30, dim_output = 88, num_layers=4,):
    """
    some simple model used in the original article. 
    
    @param dim_output
        equals to number of points in volatility surface    
    """
    model = nn.Sequential(
        *([nn.Linear(dim_input, dim_hidden), nn.ELU(),]
        + [nn.Linear(dim_hidden, dim_hidden), nn.ELU(),] * (num_layers - 2)
        + [nn.Linear(dim_hidden, dim_output)])
    ).float()

    return model

class Learner:
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-3)
        self.criterion = nn.MSELoss()
        self.LOG_EVERY_EPOCHS = 5

        self.train_loader = train_loader
        self.test_loader  = test_loader

        self.train_loss_list = []
        self.test_loss_list = []

    def train(self, num_epochs=21, **kwargs):
        for epoch in tqdm(range(num_epochs)):
            train_loss = 0.
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.train_loss_list.append(train_loss / len(self.train_loader))

            # testing
            test_loss = 0.
            with torch.no_grad():
                for data, target in self.test_loader:
                    output = self.model(data)
                    test_loss += F.mse_loss(output, target).item() # sum up batch loss
            
            self.test_loss_list.append(test_loss / len(self.test_loader))

            clear_output(True)
            self.show_loss(epoch, show_all=False,**kwargs)

        self.show_loss(num_epochs, show_all=True)

    def show_loss(self, epoch, show_all=True, filename=''):
        fig, axis = plt.subplots(1, 1)
        axis.set_title(epoch)
        num_epochs = len(self.train_loss_list)
        from_idx = 0 if show_all else num_epochs // 2
        
        axis.plot(range(from_idx, num_epochs), self.train_loss_list[from_idx:], label='train')
        axis.plot(range(from_idx, num_epochs), self.test_loss_list[from_idx:],  label='test')
        
        axis.set_yscale("log" if show_all else "linear")
        axis.grid()
        axis.legend()
        if filename!='':
            plt.savefig(filename)
        plt.show()

    def predict(self, dataset, vol_scaler):
        """
        @returns
            pred_surf_nms, target_surf_nms
        """
        pred_surf_nk = []
        target_surf_nk = []
        for params, surf in tqdm(dataset):
            pred_surf_nk.append(to_np(self.model(params)))
            target_surf_nk.append(to_np(surf))
            
        pred_surf_nk   = vol_scaler.inverse_transform(np.array(pred_surf_nk))
        target_surf_nk = vol_scaler.inverse_transform(np.array(target_surf_nk))


        pred_surf_nms   = pred_surf_nk.reshape(-1, 8, 11)
        target_surf_nms = target_surf_nk.reshape(-1, 8, 11)

        return pred_surf_nms, target_surf_nms




class ParamsOptimizer:
    def __init__(self, model, params_scaler, vol_scaler):
        self.model = deepcopy(model).double()
        for param in model.parameters():
            param.requires_grad = False

        self.params_scaler = params_scaler
        self.vol_scaler = vol_scaler

        self.criterion = nn.MSELoss()

        
    def _move_target_vlt_to_torch(self, target_vlt_surf, scale_vlt_surf):
        if scale_vlt_surf:
            target_vlt_surf = self.vol_scaler.transform(target_vlt_surf.reshape(1, -1)).ravel()           
        if isinstance(target_vlt_surf, np.ndarray):
            target_vlt_surf = torch.DoubleTensor(target_vlt_surf.ravel())
        assert target_vlt_surf.shape[0] == 88 and target_vlt_surf.ndim == 1 

        self.target_vlt_surf = target_vlt_surf

    def _get_real_target_vlt_numpy(self, scale_vlt_surf):
        target_vlt_surf = to_np(self.target_vlt_surf)
        target_vlt_surf_ms = self.vol_scaler.inverse_transform(target_vlt_surf).reshape(8, 11)
        return target_vlt_surf_ms

    def _get_pred_vlt_surf_ms(self, final_params):
        assert isinstance(final_params, np.ndarray)
        pred_vlt_surf = to_np(self.model(torch.DoubleTensor(final_params)))
        pred_vlt_surf_ms = self.vol_scaler.inverse_transform(pred_vlt_surf).reshape(8, 11)
        return pred_vlt_surf_ms

    def optimize(self, target_vlt_surf, initial_params=None, num_iterations=21, real_params=None, bounds=None,
                OptimCls=optim.LBFGS, lr=1., scale_vlt_surf=False,target_folder=''):
        self._move_target_vlt_to_torch(target_vlt_surf, scale_vlt_surf)

        if OptimCls is optim.LBFGS:
            assert num_iterations < 20, "for LBFGS you need at most 5 (most probably 1 or 2)"

        if initial_params is None:
            initial_params = torch.zeros(self.model[0].in_features, dtype=float)

        input_params = nn.Parameter(initial_params.double().clone(), requires_grad=True)

        try:
            optimizer = OptimCls([input_params], lr=lr)
        except TypeError:
            optimizer = OptimCls([input_params], bounds=bounds)


        self.surface_loss_list = []
        self.params_loss_list = []

        for epoch in tqdm(range(num_iterations)):        
            def closure():
                optimizer.zero_grad()
                pred_vlt_surf = self.model(input_params)
                loss = self.criterion(pred_vlt_surf, self.target_vlt_surf)
                loss.backward()
                return loss
            
            optimizer.step(closure)

            with torch.no_grad():
                pred_vlt_surf = self.model(input_params)
                loss = self.criterion(pred_vlt_surf, self.target_vlt_surf)
                self.surface_loss_list.append(loss.item())
                if real_params is not None:
                    self.params_loss_list.append(((input_params - real_params)**2).mean())
        
        self.show_loss(self.surface_loss_list, "surface mse loss by iteration")
        if real_params is not None:
            self.show_loss(self.params_loss_list, "params mse loss by iteration")
            
        final_params = to_np(input_params)
        pred_vlt_surf = self._get_pred_vlt_surf_ms(final_params)
        target_vlt_surf_ms = self._get_real_target_vlt_numpy(scale_vlt_surf)
        return self.params_scaler.inverse_transform(final_params), pred_vlt_surf, target_vlt_surf_ms


    def show_loss(self, loss_list, title=None,filename=''):
        fig, axis = plt.subplots(1, 1)
        axis.set_title(title)

        axis.plot(loss_list)

        axis.set_yscale("log")
        axis.grid()
        if filename!='':
            plt.savefig(filename)   
        plt.show()
