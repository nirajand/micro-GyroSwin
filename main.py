import argparse
import lightning as L
import zarr
import torch
from torch.utils.data import DataLoader, Dataset
from model_factory import ScalableGyroNet

class PlasmaDataset(Dataset):
    def __init__(self, path):
        self.data = zarr.open(path, mode='r')
    def __len__(self): return self.data.dist_func.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.data.dist_func[i]).view(-1, 1), torch.from_numpy(self.data.target_flux[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accel", type=str, default="auto")
    args = parser.parse_args()

    dm = DataLoader(PlasmaDataset("data/plasma.zarr"), batch_size=1)
    model = ScalableGyroNet()

    # Scaling Logic: RTX 2060 (6GB) gets 16-bit mixed precision automatically
    trainer = L.Trainer(
        accelerator=args.accel,
        devices=1,
        max_epochs=5,
        precision="16-mixed" if args.accel != "cpu" else "32"
    )
    
    trainer.fit(model, dm)
    trainer.save_checkpoint("models/gyro_final.ckpt")
