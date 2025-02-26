from src.data.imdb import IMDBDataModule
from src.models.linear_probe import LinearProbe
import lightning as L
import torch
import torch.nn as nn
import hydra
from transformers import AutoTokenizer, AutoModel

def train_bert_linear_probe(
        pt_name,
        batch_size: int,
        num_workers: int,
        max_epochs: int,
        seed: int,
        ):
    """
    Train a linear probe on top of a pretrained BERT/ERNIE model
    
    Args:
        pt_name: Name of pretrained model
        batch_size: Batch size for training
        num_workers: Number of workers for dataloader
        max_epochs: Maximum number of epochs to train for
        seed: Random seed for reproducibility
    """
    if pt_name == "ERNIE":
        pt_tag = "nghuyong/ernie-2.0-base-en"
    elif pt_name == "BERT":
        pt_tag = "google-bert/bert-base-uncased"
    else:
        raise ValueError("Not a supported pretrained model")

    L.seed_everything(seed)
    pt_model = AutoModel.from_pretrained(pt_tag)
    model = LinearProbe(pt_model)
    
    tokenizer = AutoTokenizer.from_pretrained(pt_tag)
    data_mod = IMDBDataModule(tokenizer=tokenizer, max_position_embeddings=pt_model.config.max_position_embeddings, batch_size=batch_size, num_workers=num_workers)

    # Setup trainer
    trainer = L.Trainer(
        default_root_dir=f"~/lprobe/{pt_name}/seed",
        devices="auto",
        accelerator="auto",
        benchmark=True,
        max_epochs=max_epochs,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_loss_avg",
                mode="min",
                save_last="link",
                save_on_train_epoch_end=False,
                ),
            L.pytorch.callbacks.DeviceStatsMonitor(),
        ],
    )

    # Train and test
    trainer.fit(model, data_mod)
    trainer.test(model=model, datamodule=data_mod)
    
    return model, trainer

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config):
#   print(config)
#   sys.exit()
    if config.get("pt_name") is None:
        raise ValueError(
                "Config input is incomplete. Set when calling. E.g.:"
                "\n\tpython train.py +pt_name=ERNIE"
                )

    train_bert_linear_probe(**config)

    pass


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

