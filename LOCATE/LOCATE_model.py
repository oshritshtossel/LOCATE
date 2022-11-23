import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F


class GeneralModel(pl.LightningModule):
    """
    LOCATE class
    """

    def __init__(self, train_metab, train_micro, input_size, representation_size, weight_decay_rep=0.999,
                 weight_decay_dis=0.999, lr_rep=0.001, lr_dis=0.001, rep_coef=1.0, dis_coef=0.0, activation_rep="relu",
                 activation_dis="relu", neurons=50, neurons2=50, dropout=0.0):
        """
        Initializes the model with the model hyperparameters.
        :param train_metab: Known metabolites of the training for data we have both microbiome and metabolites (Tensor)
        :param train_micro: Tensor of the microbial features (Tensor)
        :param input_size: train_micro.shape[1] (int)
        :param representation_size: Size of the intermediate representstion z, (int)
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
        """
        super().__init__()
        self.neurons = neurons
        self.neurons2 = neurons2
        if activation_rep == "relu":
            self.activation_rep = nn.ReLU
        elif activation_rep == "elu":
            self.activation_rep = nn.ELU
        elif activation_rep == "tanh":
            self.activation_rep = nn.Tanh

        if activation_dis == "relu":
            self.activation_dis = nn.ReLU
        elif activation_dis == "elu":
            self.activation_dis = nn.ELU
        elif activation_dis == "tanh":
            self.activation_dis = nn.Tanh
        self.linear_representation = nn.Sequential(nn.Linear(input_size, representation_size),
                                                   self.activation_rep(),
                                                   nn.Dropout(dropout),
                                                   nn.Linear(representation_size, self.neurons),
                                                   self.activation_rep(),
                                                   nn.Dropout(dropout),
                                                   nn.Linear(self.neurons, representation_size)
                                                   )
        self.discriminator = nn.Sequential(nn.Linear(representation_size, self.neurons2), self.activation_dis(),
                                           nn.Linear(self.neurons2, 1), nn.Sigmoid())
        self.train_metab = train_metab
        self.train_micro = train_micro
        self.weight_decay_rep = weight_decay_rep
        self.weight_decay_dis = weight_decay_dis
        self.lr_rep = lr_rep
        self.lr_dis = lr_dis
        self.find_transformer(self.linear_representation(train_micro))
        self.rep_coef = rep_coef
        self.dis_coef = dis_coef

    def find_transformer(self, Z):
        """
        Finds the approximated A* to relate the intermediate representation of the microbiome to the training metabolites
        :param Z: Intermediate representation of the microbiome
        :return: Approximated A*
        """
        X = torch.linalg.lstsq(Z, self.train_metab)
        a, b, c = torch.svd_lowrank(X.solution, q=min(6, self.train_metab.shape[1]))
        b = torch.diag(b)
        self.X = a @ b @ c.T

    def forward(self, micro, train=False):
        """
        LOCATE's forward function
        :param micro: microbial features (Tensor)
        :param train: Binary mode (True = training mode, False = test mode)
        :return: Predicted metaolites and the intermediate representation
        """
        Z = self.linear_representation(micro)
        if train:
            self.find_transformer(Z)
        metab = Z @ self.X
        return metab, Z

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.linear_representation.parameters(), lr=self.lr_rep,
                                       weight_decay=self.weight_decay_rep)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_dis,
                                       weight_decay=self.weight_decay_dis)
        return [optimizer_g, optimizer_d]

    def loss_g(self, metab, y, mode="valid"):
        loss = torch.tensor(0., requires_grad=True)
        loss = loss + F.mse_loss(metab, y)
        self.log(f"mse loss {mode}", loss, prog_bar=True)

        return loss

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        metab, Z = self.forward(self.train_micro, train=True)
        if optimizer_idx == 0:
            return self.loss_g(metab, self.train_metab, mode="train")

    def validation_step(self, train_batch, batch_idx):
        x, y, cond = train_batch
        metab, Z = self.forward(x)
        return self.loss_g(metab, y)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        loss.backward(retain_graph=True)
