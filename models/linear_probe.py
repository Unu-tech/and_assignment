"""
Linear probing LightningModule. Assumes binary classification task for now.
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torcheval.metrics.functional as mF


class LinearProbe(L.LightningModule):
    """
    LightningModule for attaching linear heads on intermediate states.
    This module defines how linear heads are trained, validated, and tested.
    """
    def __init__(self, pt_model):
        super().__init__()
        self.pt_model = pt_model
        self.pt_model.requires_grad_(False)  # freeze pt_model
        self.pt_model.eval()

        # Get model configuration
        self.linears = nn.ModuleList()

        self.embed_dim = pt_model.config.hidden_size
        self.depth = pt_model.config.num_hidden_layers
        self.out_dim = 1  # Binary classification

        # Create linear probes for each transformer layer
        for i in range(self.depth):
            # We will concat CLS and mean of all other tokens
            linear = nn.Linear(2 * self.embed_dim, self.out_dim)
            linear.weight.data.normal_(mean=0.0, std=0.01)
            linear.bias.data.zero_()
            self.linears.append(linear)

    def forward(self, inputs):
        if self.pt_model.training is True:
            raise RuntimeError("Attempted unfreezing pretrained model")

        with torch.no_grad():
            # Get hidden states from all layers
            outputs = self.pt_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                output_hidden_states=True,
                return_dict=True,
            )

            # Get hidden states from each layer
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer

        # Apply linear probes to each layer's representation
        results = []
        for state, linear in zip(hidden_states, self.linears):

            # Get intermediate of CLS
            cls_z = state[:, 0]
            mean_z = torch.mean(state[:, 1:], dim=1)
            results.append(linear(torch.cat((cls_z, mean_z), dim=-1)).squeeze())

        return results

    def training_step(self, train_batch):
        inputs, labels = train_batch, train_batch["label"]
        z_list = self.forward(inputs)

        loss = 0
        acc = 0

        for i, z_l in enumerate(z_list):
            z = F.sigmoid(z_l)
            # Binary classification
            layer_loss = F.binary_cross_entropy(z, labels.float())
            layer_acc = mF.binary_accuracy(z, labels.float())
            self.log(f"train_loss_l{i+1}", layer_loss, on_epoch=True, prog_bar=False)
            loss += layer_loss
            self.log(f"train_acc_l{i+1}", layer_acc, on_epoch=True, prog_bar=False)
            acc += layer_acc

        avg_loss = loss / len(z_list)
        avg_acc = acc / len(z_list)
        self.log("train_loss_avg", avg_loss, on_epoch=True, prog_bar=True)
        self.log("train_acc_avg", avg_acc, on_epoch=True, prog_bar=True)

        return avg_loss

    def validation_step(self, val_batch):
        inputs, labels = val_batch, val_batch["label"]
        z_list = self.forward(inputs)

        loss = 0
        acc = 0

        for i, z_l in enumerate(z_list):
            z = F.sigmoid(z_l)
            # Binary classification
            layer_loss = F.binary_cross_entropy(z, labels.float())
            layer_acc = mF.binary_accuracy(z, labels.float())
            self.log(f"val_loss_l{i+1}", layer_loss, on_epoch=True, prog_bar=False)
            loss += layer_loss
            self.log(f"val_acc_l{i+1}", layer_acc, on_epoch=True, prog_bar=False)
            acc += layer_acc

        avg_loss = loss / len(z_list)
        avg_acc = acc / len(z_list)
        self.log("val_loss_avg", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc_avg", avg_acc, on_epoch=True, prog_bar=True)

        return avg_loss

    def test_step(self, test_batch):
        inputs, labels = test_batch, test_batch["label"]
        z_list = self.forward(inputs)

        loss = 0
        acc = 0
        auc = 0

        for i, z_l in enumerate(z_list):
            # Binary classification
            z = F.sigmoid(z_l)
            layer_loss = F.binary_cross_entropy(z, labels.float())
            layer_acc = mF.binary_accuracy(z, labels.float())
            layer_auc = mF.binary_auroc(z, labels.float())

            self.log(f"test_loss_l{i+1}", layer_loss, on_epoch=True, prog_bar=False)
            self.log(f"test_acc_l{i+1}", layer_acc, on_epoch=True, prog_bar=False)
            self.log(f"test_auc_l{i+1}", layer_auc, on_epoch=True, prog_bar=False)

            loss += layer_loss
            acc += layer_acc
            auc += layer_auc

        loss = loss / len(z_list)
        acc = acc / len(z_list)
        auc = auc / len(z_list)

        self.log("test_loss_avg", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc_avg", acc, on_epoch=True, prog_bar=True)
        self.log("test_auc_avg", auc, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=1e-3, weight_decay=0.01
        )  # defaults
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1000, T_mult=1
        )  # around 1 restart per ep (1 ep is 25000/32=781.25 steps)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
