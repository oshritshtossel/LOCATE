import torch
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from LOCATE_model import GeneralModel
from torch.utils.data import DataLoader, TensorDataset


def LOCATE_training(X_train, Y_train, X_val, Y_val, representation_size=10, weight_decay_rep=0.02, weight_decay_dis=0.2,
                    lr_rep=0.001, lr_dis=0.01, rep_coef=1.0, dis_coef=0.0, activation_rep="elu", activation_dis="elu",
                    neurons=20, neurons2=10, dropout=0.08):
    """

    :param X_train: Log normalized and with column z-score dataframe of training microbiome features (dataframe)
    :param Y_train: Log normalized and with column z-score dataframe of training metabolites fearures (datframe)
    :param X_val: Log normalized and with column z-score dataframe of validation microbiome features (dataframe)
    :param Y_val: Log normalized and with column z-score dataframe of validation metabolites fearures (datframe) - for early stopping
    :param representation_size: Dimension of the intermediate representation (int)
    :param weight_decay_rep: L2 regularization coefficient of the representation network (float)
    :param weight_decay_dis: L2 regularization of the optional discriminator, is not used in the paper (float)
    :param lr_rep: Leaning rate of the representation network (float)
    :param lr_dis: Learning rate of the optional discriminator network, is not used in the paper (float)
    :param rep_coef: Weight of the loss upgrades of the representation network, is set to 1, when no discriminator is used (float)
    :param dis_coef: Weight of the loss upgrades of the discriminator network, is set to 0, when no discriminator is used (float)
    :param activation_rep: Activation function of the representation network, one of: {relu,elu,tanh}
    :param activation_dis: Activation function of the discriminator network, one of: {relu,elu,tanh}
    :param neurons: Number of neurons in the first layer of the representation network (int)
    :param neurons2: Number of neurons in the second layer of the representation network (int)
    :param dropout: Dropout parameter (float)
    :return: Trained model
    """
    type = np.ones(len(X_train.index) + len(X_val.index))
    indexes = list(X_train.index) + list(X_val.index)
    type = pd.DataFrame(data=type, index=indexes, columns=["t"])

    type_t = type.loc[X_train.index]
    type_v = type.loc[X_val.index]

    Y_train = torch.tensor(Y_train.to_numpy()).type(torch.float32)
    X_train = torch.tensor(X_train.to_numpy()).type(torch.float32)

    train_dataloader = DataLoader(TensorDataset(torch.tensor(type_t.to_numpy()).type(torch.float32)),
                                  batch_size=1000)
    valid_dataloader = DataLoader(TensorDataset(torch.tensor(X_val.to_numpy()).type(torch.float32),
                                                torch.tensor(Y_val.to_numpy()).type(torch.float32),
                                                torch.tensor(type_v.to_numpy()).type(torch.float32)),
                                  batch_size=X_train.shape[0])
    early_stop_callback = EarlyStopping(monitor='mse loss valid', patience=50, min_delta=0.001, mode="min")
    model = GeneralModel(Y_train, X_train, X_train.shape[1], representation_size, weight_decay_rep, weight_decay_dis,
                         lr_rep, lr_dis, rep_coef, dis_coef, activation_rep, activation_dis, neurons, neurons2, dropout)
    trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1000, callbacks=[early_stop_callback])
    trainer.fit(model, train_dataloader, valid_dataloader)
    return model


def LOCATE_predict(model, X_val, metab_names):
    """

    :param model: Trained model
    :param X_val: Log normalized and with column z-score dataframe of validation microbiome features (dataframe)
    :param metab_names: List of metabolites names
    :return: Z_val = intermediate representation, metabolites predictions dataframe
    """
    pred_metab, Z1 = model(torch.tensor(X_val.to_numpy()).type(torch.float32))
    Z_val = pd.DataFrame(Z1.detach().numpy(), index=X_val.index)
    n_pred = pd.DataFrame(pred_metab.detach().numpy(), columns=metab_names)
    return Z_val, n_pred
