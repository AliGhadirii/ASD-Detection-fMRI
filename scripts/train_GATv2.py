import argparse
import os
from tqdm import tqdm
import torch
import pandas as pd

from models.GATv2 import GATv2
from data.data_preparation import data_preparation
from utils.utils import get_metrics, count_parameters, plot_loss, plot_confusion_matrix


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\train_adjacency_tangent.npz",
        help="Path to the adjacancy matrix",
        required=True,
    )
    parser.add_argument(
        "--time_series_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\time_series.npz",
        help="Path to the time series matrix",
        required=True,
    )
    parser.add_argument(
        "--y_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\Y_target.npz",
        help="Path to the y target",
        required=True,
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        help="Path to the model weights file",
        required=False,
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path to the folder you want to save the model results",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Size of batch",
        required=False,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
        required=True,
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=1,
        help="Number of heads used in Attention Mechanism",
        required=False,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning Rate used for training the model",
        required=False,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Earlt Stopping patience",
        required=False,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="threshold use in data preprocessing section",
        required=False,
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="dropout rate used before linear classifier",
        required=False,
    )
    parser.add_argument(
        "--data_scaler_type",
        type=str,
        default="Standard",
        help="Method used for scaling the data. options: ['MinMax', 'Standard']",
        required=False,
    )
    args = parser.parse_args()
    return args


def train(model, device, batch, optimizer, loss_fn):

    model.train()
    batch = batch.to(device)
    optimizer.zero_grad()
    pred = model(batch)
    loss = loss_fn(pred.squeeze(), batch.y.float())

    loss.backward()
    optimizer.step()

    return loss.item()


def eval_batch(model, device, batch):

    model.eval()
    batch = batch.to(device)

    with torch.no_grad():
        y_pred = model(batch)

    y_pred = y_pred.detach().cpu()
    y_true = batch.y.view(y_pred.shape).detach().cpu()

    return y_pred, y_true


def eval(model, device, dataloader, loss_fn):

    model.eval()
    y_true = []
    y_pred = []

    for batch in dataloader:

        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            y_true.append(batch.y.view(pred.shape))
            y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).float().squeeze()
    y_pred = torch.cat(y_pred, dim=0).float().squeeze()

    val_loss = loss_fn(y_pred, y_true)
    y_true = y_true.detach().cpu().int()
    y_pred = y_pred.detach().cpu().int()

    return val_loss.item(), y_pred, y_true


def main(args):

    tag = "GATv2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Used Device is : {}".format(device))

    train_loader, val_loader, test_loader = data_preparation(
        adj_path=args.adj_path,
        time_series_path=args.time_series_path,
        y_path=args.y_path,
        batch_size=args.batch_size,
        threshold=args.threshold,
        scaler_type=args.data_scaler_type,
        CV=False,
    )

    model = GATv2(
        input_feat_dim=next(iter(train_loader)).x.shape[1],
        dim_shapes=[(128, 64), (64, 64), (64, 32)],
        heads=args.heads,
        num_classes=1,
        dropout_rate=args.dropout_rate,
    ).to(device)

    if args.weights_path is not None:
        model.load_weights(args.weights_path)

    count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    best_val_loss = 1000
    best_metrics = {}
    trigger_times = 0

    for epoch in range(1, 1 + args.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, batch in loop:

            loss = train(model, device, batch, optimizer, loss_fn)

            y_pred, y_true = eval_batch(model, device, batch)
            train_metrics_batch = get_metrics(y_pred, y_true)

            loop.set_description(f"Epoch: {epoch:02d}/{args.epochs:02d}")
            loop.set_postfix_str(
                f"batch {batch_idx}/{len(train_loader)}, "
                f"Loss: {loss:.4f}, "
                f"Accuracy: {100 * train_metrics_batch['acc']:.2f}%"
            )

        loss, train_y_pred, train_y_true = eval(model, device, train_loader, loss_fn)
        train_metrics = get_metrics(train_y_pred, train_y_true)

        val_loss, val_y_pred, val_y_true = eval(model, device, val_loader, loss_fn)
        val_metrics = get_metrics(val_y_pred, val_y_true)

        train_losses.append(loss)
        val_losses.append(val_loss)

        print(
            f"Loss: {loss:.4f}, "
            f"Accuracy: {100 * train_metrics['acc']:.2f}%, "
            f"Val_Loss: {val_loss:.4f}, "
            f"Val_Accuracy: {100 * val_metrics['acc']:.2f}%"
        )

        # Early stopping
        if val_loss >= best_val_loss:
            trigger_times += 1
            print(
                f"Val_Loss didn't improve from {best_val_loss}, trigger_times is {trigger_times}"
            )

            if trigger_times >= args.patience:
                print("Early stopping reached trigger_times limit")
                break

        else:
            print(f"Val_Loss improved from {best_val_loss} to {val_loss}")
            trigger_times = 0

        # Model Checkpoint
        if val_loss < best_val_loss:
            weights_path = os.path.join(args.results, f"model_weights_{tag}.pt")
            torch.save(model.state_dict(), weights_path)
            best_val_loss = val_loss
            best_metrics = val_metrics
            print(f"Model Checkpointed: model saved in {weights_path}")

    # Evaluating the best model
    print("\n\n\nBest Model results on validation set:")

    print(f"Best Validation Loss was : {best_val_loss:.4f}")
    print(f"Best Validation accuracy was : {100 * best_metrics['acc']:.2f}")

    plot_loss(train_losses, val_losses)
    label_names = ["0", "1"]
    plot_confusion_matrix(cm=best_metrics["cm"], classes=label_names)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
