import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from rsna19.configs.second_level import Config
from sklearn.metrics import log_loss


def main(config):
    train_x = torch.tensor(np.load(config.train_x), dtype=torch.float32)
    train_y = torch.tensor(np.load(config.train_y), dtype=torch.float32)
    val_x = torch.tensor(np.load(config.val_x), dtype=torch.float32)
    val_y = torch.tensor(np.load(config.val_y), dtype=torch.float32)

    # undo sigmoid
    train_pred = train_x[:, :config.predictions_in]
    val_pred = val_x[:, :config.predictions_in]

    train_pred[train_pred > 0] = torch.log(train_pred[train_pred > 0] / (1-train_pred[train_pred > 0]))
    val_pred[val_pred > 0] = torch.log(val_pred[val_pred > 0] / (1-val_pred[val_pred > 0]))

    # model = torch.nn.Linear(train_x.shape[1], features_out)

    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], config.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(config.hidden, config.features_out)
    )

    train_x = train_x.cuda()
    train_y = train_y.cuda()
    val_x = val_x.cuda()
    val_y = val_y.cuda()
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    val_log_loss = 0
    class_weights = torch.tensor(config.class_weights, dtype=torch.float32).cuda() * 6 / 7

    for i in tqdm(range(config.n_epochs)):
        optimizer.zero_grad()

        y_hat = model(train_x)
        loss = F.binary_cross_entropy_with_logits(y_hat, train_y, weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        val_y_hat = model(val_x)
        val_loss = F.binary_cross_entropy_with_logits(val_y_hat, val_y, weight=class_weights)

        if config.sklearn_loss:
            val_log_loss = log_loss(
                val_y.detach().cpu().numpy().flatten(),
                torch.sigmoid(val_y_hat).detach().cpu().numpy().flatten(),
                sample_weight=[1, 1, 1, 1, 1, 2] * val_y.shape[0],
                eps = 1e-7
            )

        model.train()

        if i % 50 == 0:
            print(f'{i:04d}: train: {loss.item():.04f}, val: {val_loss.item():.04f}, sklearn: {val_log_loss:.04f}')


if __name__ == "__main__":
    main(Config())
