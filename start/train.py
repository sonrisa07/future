import argparse
import copy
import datetime
import os
import time

import math
import numpy as np
import pandas as pd
import torch
from rich.progress import track
from torch import nn

from model.Real import Real
from model.Best import Best
from model.RTModel import Nut
from model.LSTM import LSTM
from metrics import RMSE_MAE_MAPE
from model.STGCN import STGCN
from start import plots
from utils import get_path

MODEL_MAP = {
    'LSTM': LSTM,
    'STGCN': STGCN,
    'CHESTNUT': Real,
    'BEST': Best
}

lr = 1e-2
decay = 5e-7
drop_prob = 0.1
eps = 1e-8
lr_decay_rate = 0.1
milestones = []
model_type = next(iter(MODEL_MAP))

scale = 1
epoch = 20
scope = 0.2
split = 0.8
device = 'cuda:0'


def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


@torch.no_grad()
def eval_model(model, val_loader, criterion, edge_index, pac):
    model.eval()
    batch_loss_list = []
    if isinstance(pac, STGCN):
        st_load, st_svc = pac.get_tsp_data()
        for tra, info, qos in track(val_loader, description='evaluating'):
            tra, info, qos = tra.to(device), info.to(device), qos.to(device)
            out_batch = model(tra, info, st_load, st_svc, edge_index)
            qos = qos.view(-1, 1)
            out_batch = out_batch.view(-1, 1)
            loss = criterion(out_batch, qos)
            batch_loss_list.append(loss.item())
    elif isinstance(pac, LSTM):
        for tra, info, load, svc, qos in track(val_loader, description='evaluating'):
            tra, info, load, svc, qos = tra.to(device), info.to(device), load.to(device), svc.to(device), qos.to(
                device)
            out_batch = model(tra, info, load, svc, edge_index)
            loss = criterion(out_batch, qos)
            batch_loss_list.append(loss.item())
    elif isinstance(pac, (Nut, Real)):
        tra, u_inv, srv, e_inv, inter = pac.get_tsp_data()
        tra, srv, inter = tra.to(device), srv.to(device), inter.to(device)
        for info, qos in track(val_loader):
            info, qos = info.to(device), qos.to(device)
            out_batch = model(tra, u_inv, srv, e_inv, inter, info, qos, edge_index)
            qos = qos.view(-1, 1)
            out_batch = out_batch.view(-1, 1)
            loss = criterion(out_batch, qos)
            batch_loss_list.append(loss.item())
    elif isinstance(pac, Best):
        for u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s, rate_b, s_c, s_s, s_b, tot_c, tot_s, tot_b, qos in track(
                train_loader):
            u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s, rate_b, s_c, s_s, s_b, tot_c, tot_s, tot_b, qos = u_lat.to(
                device), u_lon.to(device), u_speed.to(device), u_direction.to(device), e_lat.to(device), e_lon.to(
                device), e_radius.to(device), e_c.to(device), e_s.to(device), e_b.to(device), rate_c.to(
                device), rate_s.to(device), rate_b.to(device), s_c.to(device), s_s.to(device), s_b.to(device), tot_c.to(
                device), tot_s.to(device), tot_b.to(device), qos.to(device)
            out_batch = model(u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s,
                              rate_b, s_c, s_s, s_b, tot_c, tot_s, tot_b)
            qos = qos.view(-1, 1)
            out_batch = out_batch.view(-1, 1)
            loss = criterion(out_batch, qos)
            batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader, edge_index, pac):
    model.eval()
    y = []
    out = []
    if isinstance(pac, STGCN):
        st_load, st_svc = pac.get_tsp_data()
        for tra, info, qos in track(loader, description='predicting'):
            tra, info, qos = tra.to(device), info.to(device), qos.to(device)
            out_batch = model(tra, info, st_load, st_svc, edge_index)
            out_batch = out_batch.cpu().numpy()
            qos = qos.cpu().numpy()
            out.append(out_batch)
            y.append(qos)
    elif isinstance(pac, LSTM):
        for tra, info, load, svc, qos in track(loader, description='predicting'):
            tra, info, load, svc, qos = tra.to(device), info.to(device), load.to(device), svc.to(device), qos.to(
                device)
            out_batch = model(tra, info, load, svc, edge_index)
            out_batch = out_batch.cpu().numpy()
            qos = qos.cpu().numpy()
            out.append(out_batch)
            y.append(qos)
    elif isinstance(pac, (Nut, Real)):

        tra, u_inv, srv, e_inv, inter = pac.get_tsp_data()
        tra, srv, inter = tra.to(device), srv.to(device), inter.to(device)
        for info, qos in track(loader):
            info, qos = info.to(device), qos.to(device)
            out_batch = model(tra, u_inv, srv, e_inv, inter, info, qos, edge_index)
            out_batch = out_batch.cpu().numpy()
            qos = qos.cpu().numpy()
            out.append(out_batch)
            y.append(qos)
    elif isinstance(pac, Best):
        for u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s, rate_b, s_c, s_s, s_b, tot_c, tot_s, tot_b, qos in track(
                train_loader):
            u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s, rate_b, s_c, s_s, s_b, tot_c, tot_s, tot_b, qos = u_lat.to(
                device), u_lon.to(device), u_speed.to(device), u_direction.to(device), e_lat.to(device), e_lon.to(
                device), e_radius.to(device), e_c.to(device), e_s.to(device), e_b.to(device), rate_c.to(
                device), rate_s.to(device), rate_b.to(device), s_c.to(device), s_s.to(device), s_b.to(device), tot_c.to(
                device), tot_s.to(device), tot_b.to(device), qos.to(device)
            out_batch = model(u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s,
                              rate_b, s_c, s_s, s_b, tot_c, tot_s, tot_b)
            out_batch = out_batch.cpu().numpy()
            qos = qos.cpu().numpy()
            out.append(out_batch)
            y.append(qos)

    out = np.vstack(out).squeeze()
    y = np.vstack(y).squeeze()

    return y, out


def train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        edge_index,
        pac,
        png_path,
        max_epochs=200,
        early_stop=100,
        verbose=1,
        plot=True,
        log=None,
        save=None,
):
    model = model.to(device)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        print(f'{epoch:02} start.')
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, edge_index, pac)
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, val_loader, criterion, edge_index, pac)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, train_loader, edge_index, pac))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, val_loader, edge_index, pac))

    out_str = f"Early stopping at epoch: {epoch + 1}\n"
    out_str += f"Best at epoch {best_epoch + 1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += ''
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plots.plot_line(train_loss_list, val_loss_list, png_path)

    if save:
        torch.save(best_state_dict, save)
    return model


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, edge_index, pac):
    model.train()
    batch_loss_list = []
    if isinstance(pac, STGCN):
        st_load, st_svc = pac.get_tsp_data()
        for tra, info, qos in track(train_loader, description='training'):
            tra, info, qos = tra.to(device), info.to(device), qos.to(device)
            out_batch = model(tra, info, st_load, st_svc, edge_index)
            qos = qos.view(-1, 1)
            out_batch = out_batch.view(-1, 1)
            loss = criterion(out_batch, qos)
            batch_loss_list.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    elif isinstance(pac, LSTM):
        for tra, info, load, svc, qos in track(train_loader, description='training'):
            tra, info, load, svc, qos = tra.to(device), info.to(device), load.to(device), svc.to(device), qos.to(
                device)
            out_batch = model(tra, info, load, svc, edge_index)
            loss = criterion(out_batch, qos)
            batch_loss_list.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    elif isinstance(pac, (Nut, Real)):

        tra, u_inv, srv, e_inv, inter = pac.get_tsp_data()
        tra, srv, inter = tra.to(device), srv.to(device), inter.to(device)
        for info, qos in track(train_loader):
            info, qos = info.to(device), qos.to(device)
            preds = model(tra, u_inv, srv, e_inv, inter, info, qos, edge_index)
            qos = qos.view(-1, 1)
            preds = preds.view(-1, 1)
            loss = criterion(preds, qos)
            batch_loss_list.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    elif isinstance(pac, Best):
        for u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s, rate_b, s_c, s_s, s_b, tot_c, tot_s, tot_b, qos in track(
                train_loader):
            u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s, rate_b, s_c, s_s, s_b, tot_c, tot_s, tot_b, qos = u_lat.to(
                device), u_lon.to(device), u_speed.to(device), u_direction.to(device), e_lat.to(device), e_lon.to(
                device), e_radius.to(device), e_c.to(device), e_s.to(device), e_b.to(device), rate_c.to(
                device), rate_s.to(device), rate_b.to(device), s_c.to(device), s_s.to(device), s_b.to(device), tot_c.to(
                device), tot_s.to(device), tot_b.to(device), qos.to(device)
            preds = model(u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s,
                          rate_b, s_c, s_s, s_b, tot_c, tot_s, tot_b)
            qos = qos.view(-1, 1)
            preds = preds.view(-1, 1)
            loss = criterion(preds, qos)
            batch_loss_list.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    else:
        pass
    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


@torch.no_grad()
def test_model(model, test_loader, edge_index, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, test_loader, edge_index, pac)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

    parser.add_argument(
        "-model", "--model_type",
        type=str, choices=MODEL_MAP.keys(), default="LSTM",
        help="Model type to train: 'LSTM', 'STGCN' (default: LSTM)."
    )

    parser.add_argument(
        "-e", "--epoch",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)."
    )
    parser.add_argument(
        "-l", "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for the optimizer (default: 0.01)."
    )
    parser.add_argument(
        "-m", "--milestones",
        type=int,
        nargs='+',
        default=[10, 20],
        help="Epochs at which the learning rate decays (default: [])."
    )
    parser.add_argument(
        "-s", "--scope",
        type=float,
        nargs='+',
        default=0.2,
        help="The dataset scope (default: 0.2)."
    )
    parser.add_argument(
        "-dp", "--dropout",
        type=float,
        default=0.1,
        help="The dropout (default: 0.1)."
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default='cuda:1',
        help="Device to use for training, e.g., 'cuda:0', 'cuda:1', or 'cpu' (default: 'cuda:1')."
    )

    args = parser.parse_args()

    epoch = args.epoch
    lr = args.lr
    milestones = args.milestones
    device = args.device
    model_type = args.model_type
    drop_prob = args.dropout
    scope = args.scope

    # ------------------------------- load dataset ------------------------------- #

    user_df = pd.read_csv(get_path('user.csv'))
    load_df = pd.read_csv(get_path('load.csv'))
    server_df = pd.read_csv(get_path('server.csv'))
    service_df = pd.read_csv(get_path('service.csv'))
    inv_df = pd.read_csv(get_path('invocation.csv'))

    max_time = math.floor(load_df['timestamp'].max() * scale)

    user_df = user_df[user_df['timestamp'] <= max_time]
    load_df = load_df[load_df['timestamp'] <= max_time]
    inv_df = inv_df[inv_df['timestamp'] <= max_time]

    pac = MODEL_MAP[model_type](user_df, server_df, load_df, service_df, inv_df, 9)

    train_loader, val_loader, test_loader = pac.get_dataloaders(scope, split)

    # -------------------------------- load model -------------------------------- #

    model = pac.net
    model_name = pac.__class__.__name__

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=decay,
        eps=eps,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=lr_decay_rate
    )

    # --------------------------- train and test model --------------------------- #

    loss_path = f"../losses/"

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)
    edge_index = torch.LongTensor(pac.edge_index).to(device)
    model = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        edge_index,
        pac,
        os.path.join(loss_path, f"{model_name}-{now}.png"),
        epoch,
        15,
        1,
        True,
        log,
        save
    )

    print_log(f"Saved Model: {save}", log=log)

    test_model(model, test_loader, edge_index, log=log)

    log.close()
