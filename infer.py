import argparse
import os
import random
import time
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from dataloader import DataGenerator
import logging
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch_wavelets import DWT1DForward, DWT1DInverse

import csv

parser = argparse.ArgumentParser()
parser.add_argument('--minibatch_len', default=10, type=int)
parser.add_argument('--pre_len', default=1, type=int)
parser.add_argument('--interval', default=1, type=int)
parser.add_argument('--batch_size', default=2048, type=int)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--logdir', default='./log', type=str)
parser.add_argument('--datadir', default='dataaaaaa', type=str)
parser.add_argument('--netdir', default=None, type=str)


class Test:
    def __init__(self, opt, net=None):
        self.opt = opt
        self.iscuda = torch.cuda.is_available()
        self.device = f'cuda:{torch.cuda.current_device()}' if self.iscuda and not opt.cpu else 'cpu'
        self.data_set = DataGenerator(data_path=self.opt.datadir,
                                      minibatch_len=opt.minibatch_len, interval=opt.interval,
                                      use_preset_data_ranges=False, train=False, dev=False, test_shuffle=True)
        self.net = net
        self.model_path = None
        self.MSE = torch.nn.MSELoss(reduction='mean')
        self.MAE = torch.nn.L1Loss(reduction='mean')
        if net is not None:
            assert next(self.net.parameters()).device == self.device

    def load_model(self, model_path):
        self.model_path = model_path
        self.net = torch.load(model_path, map_location=self.device)

    def test(self):
        print_str = f'model details:\n{self.net.args}'
        print(print_str)
        self.log_path = self.opt.logdir + f'/{datetime.datetime.now().strftime("%y-%m-%d")}'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        logging.basicConfig(filename=os.path.join(self.log_path, f'test_{self.net.args["train_opt"].comments}.log'),
                            filemode='w', format='%(asctime)s   %(message)s', level=logging.DEBUG)
        logging.debug(print_str)
        logging.debug(self.model_path)

        # print(dataset=self.data_set.test_set)

        test_data = DataLoader(dataset=self.data_set.test_set, batch_size=self.opt.batch_size, shuffle=False,
                               collate_fn=self.data_set.collate)
        
        idwt = DWT1DInverse(wave=self.net.args['train_opt'].wavelet, mode=self.net.args['train_opt'].wt_mode).to(
            self.device)
        self.net.eval()
        tgt_set = []
        pre_set = []

        with torch.no_grad():
            his_batch_set = []
            all_score_set = []
            for i, batch in enumerate(test_data):
                batch = torch.FloatTensor(batch).to(self.device)
                print(batch.shape)
                n_batch, _, n_attr = batch.shape
                inp_batch = batch[:, :self.opt.minibatch_len-self.opt.pre_len, :]  # shape: batch * his_len * n_attr
                his_batch_set.append(inp_batch)
                tgt_set.append(batch[:, -self.opt.pre_len:, :])  # shape: batch * pre_len * n_attr
                pre_batch_set = []
                for j in range(self.opt.pre_len):
                    if j > 0:
                        new_batch = pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :].unsqueeze(1)
                        inp_batch = torch.cat((inp_batch[:, 1:, :], new_batch), dim=1)  # shape: batch * his_len * n_attr
                    if self.net.__class__.__name__ == 'WTFTP':
                        print(f"net_input {inp_batch.shape}")
                        wt_pre_batch, score_set = self.net(inp_batch)
                        print(f"net_output {wt_pre_batch[0].shape}")

                    else:
                        print(f"net_input {inp_batch.shape}")
                        wt_pre_batch = self.net(inp_batch)
                        print(f"net_output {wt_pre_batch[0].shape}")
                    pre_batch = idwt((wt_pre_batch[-1].transpose(1, 2).contiguous(),
                                      [comp.transpose(1, 2).contiguous() for comp in
                                       wt_pre_batch[:-1]])).contiguous()
                    pre_batch = pre_batch.transpose(1, 2)  # shape: batch * n_sequence * n_attr
                    pre_batch_set.append(pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :])
                if self.net.__class__.__name__ == 'WTFTP' and j == 0:
                    all_score_set.append(score_set)
                pre_batch_set = torch.stack(pre_batch_set, dim=1)  # shape: batch * pre_len * n_attr
                pre_set.append(pre_batch_set)

            tgt_set = torch.cat(tgt_set, dim=0)
            pre_set = torch.cat(pre_set, dim=0)
            print(f"tgt_set {tgt_set.shape}")
            print(f"pre_set {pre_set.shape}")

            print(tgt_set)


            # try:
            #     all_score_set = torch.cat(all_score_set, dim=0)
            # except:
            #     all_score_set = ['no scores']
            # his_batch_set = torch.cat(his_batch_set, dim=0)
            # torch.save([his_batch_set, tgt_set, pre_set, all_score_set],
            #            f'{self.net.args["train_opt"].comments}_his_tgt_pre_score.pt', _use_new_zipfile_serialization=False)

            for i in range(self.opt.pre_len):
                avemse = float(self.MSE(tgt_set[:, :i + 1, :], pre_set[:, :i + 1, :]).cpu())
                avemae = float(self.MAE(tgt_set[:, :i + 1, :], pre_set[:, :i + 1, :]).cpu())
                rmse = {}
                mae = {}
                mre = {}
                for j, name in enumerate(self.data_set.attr_names):
                    rmse[name] = float(self.MSE(self.data_set.unscale(tgt_set[:, :i + 1, j], name),
                                                self.data_set.unscale(pre_set[:, :i + 1, j], name)).sqrt().cpu())
                    mae[name] = float(self.MAE(self.data_set.unscale(tgt_set[:, :i + 1, j], name),
                                               self.data_set.unscale(pre_set[:, :i + 1, j], name)).cpu())
                    logit = self.data_set.unscale(tgt_set[:, :i + 1, j], name) != 0
                    mre[name] = float(torch.mean(torch.abs(self.data_set.unscale(tgt_set[:, :i + 1, j], name)-
                                                self.data_set.unscale(pre_set[:, :i + 1, j], name))[logit]/
                                      self.data_set.unscale(tgt_set[:, :i + 1, j], name)[logit]).cpu()) * 100 if name in 'lonlatalt' \
                                else "N/A"
                X = self.data_set.unscale(pre_set[:, :i + 1, 0], 'lon').cpu().numpy()
                Y = self.data_set.unscale(pre_set[:, :i + 1, 1], 'lat').cpu().numpy()
                Z = self.data_set.unscale(pre_set[:, :i + 1, 2], 'alt').cpu().numpy()

                X_t = self.data_set.unscale(tgt_set[:, :i + 1, 0], 'lon').cpu().numpy()
                Y_t = self.data_set.unscale(tgt_set[:, :i + 1, 1], 'lat').cpu().numpy()
                Z_t = self.data_set.unscale(tgt_set[:, :i + 1, 2], 'alt').cpu().numpy()

                MDE = np.mean(np.sqrt((X - X_t) ** 2 + (Y - Y_t) ** 2 + (Z - Z_t) ** 2))
                print_str = f'\nStep {i + 1}: \naveMSE(scaled): {avemse:.8f}, in each attr(RMSE, unscaled): {rmse}\n' \
                            f'aveMAE(scaled): {avemae:.8f}, in each attr(MAE, unscaled): {mae}\n' \
                            f'In each attr(MRE, %): {mre}\n' \
                            f'MDE(unscaled): {MDE:.8f}\n'
                print(print_str)
                logging.debug(print_str)

    def calculate_and_plot_erros(self,target,predicted):

        target = np.array(target)
        predicted = np.array(predicted)
        absolute_errors = np.abs(target - predicted)
        deviation_errors = target - predicted

        pose_errors = np.linalg.norm(target - predicted, axis=1)
        absolute_pose_errors = np.abs(pose_errors)
        deviation_pose_errors = pose_errors - pose_errors  # This would be zero, showing no deviation for same error

        # Plotting
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        file_path = "/home/vs/wtftp-model/log/traj_analyse/"
        file_name = "spline.csv"
        lable_plot = " Wavelet attention, level=3"

        # Plot Euclidean pose errors
        axs[0].plot(pose_errors, label='Pose Error (Euclidean Distance)', marker='o')
        axs[0].set_title('Pose Errors (Euclidean Distance) between Target and Predicted. MODEL:' + lable_plot, fontsize=16)
        axs[0].set_xlabel('Index', fontsize=14)
        axs[0].set_ylabel('Pose Error (Distance)', fontsize=14)
        axs[0].legend(fontsize=12)
        axs[0].grid(True)

        # Plot absolute errors
        axs[1].plot(absolute_errors, label=['X', 'Y', 'Z'])
        axs[1].set_title('Absolute Errors. MODEL:' + lable_plot, fontsize=16)
        axs[1].set_xlabel('Index', fontsize=14)
        axs[1].set_ylabel('Error', fontsize=14)
        axs[1].grid(True)  # Show grid
        axs[1].legend(fontsize=12)

        # Plot deviation errors
        axs[2].plot(deviation_errors, label=['X', 'Y', 'Z'])
        axs[2].set_title('Deviation Errors. MODEL:' + lable_plot, fontsize=16)
        axs[2].set_xlabel('Index', fontsize=14)
        axs[2].set_ylabel('Error', fontsize=14)
        axs[2].legend(fontsize=12)
        axs[2].grid(True)  # Show grid
        
        plt.tight_layout()
        plt.show()

        # data = zip(pose_errors)
        # with open(file_path+file_name, mode='w', newline='') as file:

        #     writer = csv.writer(file)
        #     # Write header
        #     writer.writerow(['pose_errors'])
        #     # Write data rows
        #     writer.writerows(data)



    def plot_3d_trajectories(self, target, pred):
        target = np.array(target)
        pred = np.array(pred)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(target[:, 0], target[:, 1], target[:, 2], color='red', s=2, label='Target')
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], color='blue', s=3, label='Prediction')

        # Get the range for each axis
        x = target[:, 0]
        y = target[:, 1]
        z = target[:, 2]

        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        z_range = np.max(z) - np.min(z)

        # Find the max range to make the axes equal
        max_range = max(x_range, y_range, z_range)

        # Set the limits for each axis to ensure they are equal
        mid_x = (np.max(x) + np.min(x)) / 2
        mid_y = (np.max(y) + np.min(y)) / 2
        mid_z = (np.max(z) + np.min(z)) / 2

        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        # Add labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectories')
        ax.legend()

        plt.show()


    def draw_demo(self, items=None, realtime=False, all_items=False):
        plt.rcParams['font.family'] = 'arial'
        plt.rcParams['font.size'] = 7
        total_num = len(self.data_set.test_set)
        test_data = DataLoader(dataset=self.data_set.test_set, batch_size=self.opt.batch_size, shuffle=False,
                               collate_fn=self.data_set.collate)
        idwt = DWT1DInverse(wave=self.net.args['train_opt'].wavelet, mode=self.net.args['train_opt'].wt_mode).to(
            self.device)
        if items is None:
            items = [random.randint(0, total_num)]
        elif type(items) is int:
            items = [items % total_num]
        elif type(items) is list and len(items) > 0 and type(items[0]) is int:
            pass
        else:
            TypeError(type(items))
        if realtime:
            items = [int(input("item: "))]

        if all_items:
            k = 0
            target = []
            pred = []
            while k < total_num:
                item = k

                n_batch = item // self.opt.batch_size
                n_minibatch = item % self.opt.batch_size
                sel_batch = None
                for i, batch in enumerate(test_data):
                    if i == n_batch:
                        sel_batch = batch
                        break
                traj = sel_batch[n_minibatch:n_minibatch + 1, ...]
                with torch.no_grad():
                    self.net.eval()
                    self.net.to(self.device)
                    inp_batch = torch.FloatTensor(traj[:, :self.opt.minibatch_len-self.opt.pre_len, :]).to(
                        self.device)  # shape: 1 * his_len * n_attr
                    pre_batch_set = []
                    full_pre_set = []
                    for j in range(self.opt.pre_len):
                        if j > 0:
                            new_batch = pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :].unsqueeze(1)
                            inp_batch = torch.cat((inp_batch[:, 1:, :], new_batch), dim=1)  # shape: batch * his_len * n_attr
                        if self.net.__class__.__name__ == 'WTFTP':
                            wt_pre_batch, score_set = self.net(inp_batch)
                        else:
                            wt_pre_batch = self.net(inp_batch)
                        if j == 0:
                            first_wt_pre = wt_pre_batch
                        pre_batch = idwt((wt_pre_batch[-1].transpose(1, 2).contiguous(),
                                        [comp.transpose(1, 2).contiguous() for comp in
                                        wt_pre_batch[:-1]])).contiguous()
                        pre_batch = pre_batch.transpose(1, 2)  # shape: 1 * n_sequence * n_attr
                        pre_batch_set.append(pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :])
                        full_pre_set.append(pre_batch[:, :self.opt.minibatch_len-self.opt.pre_len + 1, :].clone())
                    pre_batch_set = torch.stack(pre_batch_set, dim=1)  # shape: 1 * pre_len * n_attr

                lla_his = np.array(traj[0, :self.opt.minibatch_len-self.opt.pre_len, 0:3])  # shape: his_len * n_attr
                lla_trg = np.array(traj[0, -self.opt.pre_len:, 0:3])  # shape: pre_len * n_attr
                lla_pre = np.array(pre_batch_set[0, :, 0:3].cpu().numpy())  # shape: pre_len * n_attr
                for i, name in enumerate(self.data_set.attr_names):
                    if i > 2:
                        break
                    lla_his[:, i] = self.data_set.unscale(lla_his[:, i], name)
                    lla_trg[:, i] = self.data_set.unscale(lla_trg[:, i], name)
                    lla_pre[:, i] = self.data_set.unscale(lla_pre[:, i], name)
                
                target.append([float(lla_trg[:, 0]), float(lla_trg[:, 1]), float(lla_trg[:, 2])])
                pred.append([float(lla_pre[:, 0]), float(lla_pre[:, 1]), float(lla_pre[:, 2])])
                
                k+=1

            self.calculate_and_plot_erros(target,pred)
            self.plot_3d_trajectories(target, pred)

        if not all_items:
            while len(items) > 0 and items[0] > 0:
                item = items[0]
                del items[0]
                n_batch = item // self.opt.batch_size
                n_minibatch = item % self.opt.batch_size
                sel_batch = None
                for i, batch in enumerate(test_data):
                    if i == n_batch:
                        sel_batch = batch
                        break
                traj = sel_batch[n_minibatch:n_minibatch + 1, ...]
                with torch.no_grad():
                    self.net.eval()
                    self.net.to(self.device)
                    inp_batch = torch.FloatTensor(traj[:, :self.opt.minibatch_len-self.opt.pre_len, :]).to(
                        self.device)  # shape: 1 * his_len * n_attr
                    pre_batch_set = []
                    full_pre_set = []
                    for j in range(self.opt.pre_len):
                        if j > 0:
                            new_batch = pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :].unsqueeze(1)
                            inp_batch = torch.cat((inp_batch[:, 1:, :], new_batch), dim=1)  # shape: batch * his_len * n_attr
                        if self.net.__class__.__name__ == 'WTFTP':
                            wt_pre_batch, score_set = self.net(inp_batch)
                        else:
                            wt_pre_batch = self.net(inp_batch)
                        if j == 0:
                            first_wt_pre = wt_pre_batch
                        pre_batch = idwt((wt_pre_batch[-1].transpose(1, 2).contiguous(),
                                        [comp.transpose(1, 2).contiguous() for comp in
                                        wt_pre_batch[:-1]])).contiguous()
                        pre_batch = pre_batch.transpose(1, 2)  # shape: 1 * n_sequence * n_attr
                        pre_batch_set.append(pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :])
                        full_pre_set.append(pre_batch[:, :self.opt.minibatch_len-self.opt.pre_len + 1, :].clone())
                    pre_batch_set = torch.stack(pre_batch_set, dim=1)  # shape: 1 * pre_len * n_attr

                lla_his = np.array(traj[0, :self.opt.minibatch_len-self.opt.pre_len, 0:3])  # shape: his_len * n_attr
                lla_trg = np.array(traj[0, -self.opt.pre_len:, 0:3])  # shape: pre_len * n_attr
                lla_pre = np.array(pre_batch_set[0, :, 0:3].cpu().numpy())  # shape: pre_len * n_attr
                for i, name in enumerate(self.data_set.attr_names):
                    if i > 2:
                        break
                    lla_his[:, i] = self.data_set.unscale(lla_his[:, i], name)
                    lla_trg[:, i] = self.data_set.unscale(lla_trg[:, i], name)
                    lla_pre[:, i] = self.data_set.unscale(lla_pre[:, i], name)

                fig = plt.figure(figsize=(10,10))

                ax = fig.add_subplot(111, projection='3d')

                # ax.axis('equal')
                ax.plot3D(lla_his[:, 0], lla_his[:, 1], lla_his[:, 2], marker='o', markeredgecolor='dodgerblue',
                            label='his')
                ax.plot3D(lla_trg[:, 0], lla_trg[:, 1], lla_trg[:, 2], marker='*', markeredgecolor='blueviolet',
                            label='tgt')
                ax.plot3D(lla_pre[:, 0], lla_pre[:, 1], lla_pre[:, 2], marker='p', markeredgecolor='orangered',
                            label='pre')
                ax.set_xlabel('x, m')
                ax.set_ylabel('y, m')
                ax.set_zlabel('z, m')

                ax.set_xlim([-1.0, 1.0])
                ax.set_ylim([-1.0, 1.0])
                ax.set_zlim([0, 2.0])

                plt.suptitle(f'item_{item}')
                ax.legend()
                plt.tight_layout()
                plt.show()

                if realtime:
                    items.append(int(input("item: ")))

if __name__ == '__main__':
    opt = parser.parse_args()
    test = Test(opt)
    test.load_model(opt.netdir)
    test.test()
    test.draw_demo(all_items=False)
