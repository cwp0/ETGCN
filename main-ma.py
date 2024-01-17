import json
import socket

import numpy as np
import torch.nn as nn
import torch.optim
import os

import models
import torch.nn.functional as F
import utils.metrics
import utils.data

num_nodes = 12
num_features = 2
batch_size = 32
seq_len = 10
pre_len = 1
epoches = 10

class server_module:
    def __init__(self, rec_ip, rec_port, deep_web):
        self.rec_ip = rec_ip
        self.rec_port = rec_port # 8082
        self.deep_web = deep_web
        # self.policy_name = "MADDPG"  # 六个字符
        self.policy_name = "MATD3 "  # 六个字符

    def println(self, message):
        print("[ETGCN][✅]" + message)
    
    def printerr(self, message):
        print("[ETGCN][❌]" + message)

    def getAction(self, pred):
        sk = socket.socket()
        sk.connect(("127.0.0.1", 9000))
        msg = json.dumps({"messagetype": "getaction", "workload": pred[0,4:7].tolist()})
        self.println("send message to " + str(self.policy_name) + ":       " + str(msg.encode()))
        sk.sendall(msg.encode())
        return sk.recv(1024)

    def listen(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            4096
        )
        server.bind((self.rec_ip, self.rec_port))
        server.listen(10) # 最大连接数

        buffer = torch.zeros(1, seq_len, num_nodes, num_features, dtype=torch.float32)
        self.println("开始监听客户端连接：")
        while True:
            conn, addr = server.accept()
            self.println("✨✨✨✨✨ 客户端连接成功 ✨✨✨✨✨")
            self.println("socket对象：                   " + str(conn))

            reply = {"from": "ETGCN"}
            try:
                data = conn.recv(1024)
                self.println("receive message from FogSim:  " + str(data) + " " + str(len(data)))
                datadecoded = data.decode(encoding="UTF-8")  # "UTF-8"
                obs = json.loads(datadecoded, strict=False)
                if(obs["type"] == "init"):
                    buffer = torch.zeros(1, seq_len, num_nodes, num_features)
                    reply["msg"] = "[ETGCN] buffer初始化成功"
                #     存储workload同时获取动作
                else:
                    buffer = buffer[:, 1:, :, :]
                    x = torch.from_numpy(np.matrix(obs["graph"], dtype=np.float32))
                    x = x.unsqueeze(0).unsqueeze(0)
                    buffer = torch.cat((buffer, x), 1)
                    pred = self.deep_web.action(buffer)
                    action_msg = self.getAction(pred)
                    self.println(str(self.policy_name) + " return:                " + str(action_msg))
                    conn.send(action_msg)
                    self.println("send action_msg to FogSim:    " + str(action_msg))
                    continue # 跳出本次循环
                    # reply["action"] = 0.1
                    # reply["workload0"] = 5
                msg = json.dumps(reply, ensure_ascii=False)
                msg = msg.encode(encoding="UTF-8", errors="strict")
                self.println("send init message to FogSim:  " + str(msg))
                conn.send(msg)
            except UnicodeDecodeError:
                self.printerr("receive_obs UnicodeDecodeError")
                pass


class ETGCN(nn.Module):
    def __init__(self, input_dim: int, num_features, hidden_dim, regressor="linear",
               feat_max_val: float = 70.0):
        super(ETGCN, self).__init__()
        # self.adj = utils.data.functions.load_adjacency_matrix("D:\mx\T-GCN\T-GCN\T-GCN-PyTorch\data\eua_adj_1.csv")
        self.model = models.GRU(input_dim, num_features, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)  ## 1: pre_len
        self.feat_max_val = feat_max_val

    def forward(self, x):
        # (batch_size, seq_len, num_nodes, num_features)
        batch_size, _, num_nodes, _ = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions


class ETGCNNet:
    def __init__(self):
        self.model = ETGCN(num_nodes, num_features, 64)
        self.train_loader = None
        self.test_loader = None
        self.loss = F.mse_loss
        self.acc = utils.metrics.accuracy
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1.5e-3,
        )
        self.init_dataloader()
        self.loadModel("./model.pth")

    def init_dataloader(self):
        (
            self.train_dataset,
            self.val_dataset,
        ) = utils.data.functions.generate_torch_datasets(
            utils.data.functions.load_features("data/eua_loads_1.txt"),
            seq_len,
            pre_len,
            split_ratio=0.8,
            normalize=True,
        )
        from torch.utils.data import DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    def step(self, batch):
        x, y = batch
        num_nodes = x.size(2)
        predictions = self.model(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y[:, :, :, -1].reshape((-1, y.size(2)))
        return predictions, y

    # 针对fogSim一条请求的处理
    def action(self, x):
        num_nodes = x.size(2)
        predictions = self.model(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        return predictions

    # 加载或者训练成熟模型
    def loadModel(self, path):
        if(os.path.exists(path)):
            self.model = torch.load(path)
        else:
            for i in range(epoches):
                # print("第{}轮训练".format(i + 1))
                train_step = 0
                for data in self.train_loader:
                    predictions, y = self.step(data)
                    result_loss = self.loss(predictions, y)
                    self.optimizer.zero_grad()
                    result_loss.backward()
                    self.optimizer.step()

                    train_step += 1
                    if (train_step % 100 == 0):
                        print("第{}轮的第{}次训练的loss:{}".format((i + 1), train_step, result_loss.item()))

                self.model.eval()
                count = 0
                accuracy = 0.0
                with torch.no_grad():
                    for test_data in self.test_loader:
                        count += 1
                        predictions, y = self.step(test_data)
                        predictions = predictions * 70.0
                        y = y * 70.0
                        accuracy += utils.metrics.accuracy(predictions, y)
                accuracy /= count
                print("第{}轮训练在测试集上的准确率为{}".format((i+1), accuracy))
                if(accuracy > 0.82):
                    print("训练完成，保存模型退出该环节开始监听fogSim请求")
                    torch.save(self.model, path)
                    return


if __name__ == "__main__":

    net = ETGCNNet()
    server = server_module("127.0.0.1", 8082, net)
    server.listen()