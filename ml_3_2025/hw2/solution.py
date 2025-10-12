# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys
from pathlib import Path
from typing import Literal, Callable
from datetime import datetime

# List any extra packages you need here. Please, fix versions so reproduction of your results would be less painful.
PACKAGES_TO_INSTALL = ["gdown==4.4.0", "torch==2.8.0", "torchvision==0.23.0", "tqdm==4.67.1", "tensorboard==2.20.0"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch  # noqa: E402
from torch import nn, optim  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from torchvision.io import read_image  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.models import resnet18  # noqa: E402
from tqdm import tqdm  # noqa: E402


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        kind: Literal["train", "val", "test"],
        class2idx: dict | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.data_root = root
        self.kind = kind
        self.transform = transform
        self.target_transform = target_transform

        if kind == "test":
            self.imgs = list(map(lambda x: (str(x), None), (Path(root) / "images").iterdir()))
            return

        self.imgs = []
        classes = []

        for class_dir in Path(root).iterdir():
            class_name = class_dir.name
            img_dir = class_dir if kind == "val" else class_dir / "images"
            for img_path in img_dir.iterdir():
                self.imgs.append((str(img_path), class_name))
                classes.append(class_name)

        if class2idx is None:
            class_list = sorted(set(classes))
            self.class_to_idx = {name: idx for idx, name in enumerate(class_list)}
        else:
            self.class_to_idx = class2idx
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        image = read_image(img_path) / 255.0

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if self.transform is not None:
            image = self.transform(image)

        if self.kind == "test":
            return img_path, image

        label = self.class_to_idx[label]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    orig_dtype = G.dtype
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(orig_dtype)


class Muon(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 2e-2,
        betas: tuple[float, float] | None = None,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ):
        if betas is None:
            betas = (0.95, 0.999)
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, ns_steps=ns_steps)
        super(Muon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            betas = group.get("betas", (group.get("beta1", 0.95), 0.999))
            beta1 = betas[0]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.mul_(1 - lr * weight_decay)

                state = self.state[p]

                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)

                g = p.grad
                m = state["momentum"]

                if p.ndim == 2:
                    g = newtonschulz5(g, ns_steps)

                m.lerp_(g, 1 - beta1)
                g.lerp_(m, beta1)
                p.add_(g, alpha=-lr)


def get_dataloader(path: str, kind: Literal["train", "val", "test"]):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val' or 'test', the dataloader should be deterministic.
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train', 'val' or 'test'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    root = Path(path)

    if kind == "train":
        train_classes = sorted([p.name for p in (root / "train").iterdir()])
        class2idx = {name: idx for idx, name in enumerate(train_classes)}
    else:
        train_classes = sorted([p.name for p in (root / "train").iterdir()])
        class2idx = {name: idx for idx, name in enumerate(train_classes)}

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if kind == "train":
        transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomErasing(p=0.25),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

    dataset = TinyImageNetDataset(
        root / kind,
        kind,
        class2idx=class2idx if kind != "test" else None,
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=kind == "train",
    )

    return dataloader


def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = resnet18(num_classes=200)
    model.to(device)
    model = model

    return model


def get_optimizer(model: nn.Module):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    return Muon(model.parameters())


def predict(model: nn.Module, batch: torch.Tensor):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    batch = batch.to(next(model.parameters()).device)
    return model(batch)


@torch.no_grad()
def validate(dataloader: torch.utils.data.DataLoader, model: nn.Module):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    model.eval()
    device = next(model.parameters()).device

    correct_preds = 0
    num_samples = 0
    sum_loss = 0
    num_steps = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        num_samples += images.shape[0]
        sum_loss += F.cross_entropy(logits, labels).item()
        num_steps += 1

    return correct_preds / (num_samples + 1e-8), sum_loss / (num_steps + 1e-8)


def train(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    loader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
    epoch: int,
):
    model.train()
    device = next(model.parameters()).device
    correct_preds = 0
    num_samples = 0

    for i, (images, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        num_samples += images.shape[0]

        global_step = epoch * len(loader) + i
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    train_acc = correct_preds / (num_samples + 1e-8)
    return train_acc


def train_on_tinyimagenet(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    log_dir = f"logs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    num_epochs = 100
    best_val_acc = 0.0

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=optimizer.param_groups[0]["lr"],
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
    )

    for epoch in range(num_epochs):
        train_acc = train(model, loss_fn, optimizer, scheduler, train_dataloader, writer, epoch)
        val_acc, val_loss = validate(val_dataloader, model)

        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)

        print(f"Epoch {epoch}: Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f} Val loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model with val acc: {best_val_acc:.4f}")

    writer.close()


def load_weights(model: nn.Module, checkpoint_path: str):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)


def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here; md5_checksum = "abcd"
    # Your code here; google_drive_link = "https://drive.google.com/file/d/abcd/view?usp=sharing"
    md5_checksum = "c921af9801706eb7c8053255e888231d"
    google_drive_link = "https://drive.google.com/file/d/1HI44ZrLwB_oVY7da_foM6ujCqsjoTRl3/view?usp=sharing"

    return md5_checksum, google_drive_link
