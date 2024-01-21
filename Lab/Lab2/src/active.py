import numpy as np
from sklearn.metrics import log_loss
from tqdm import trange

output_file = open("./tmp/output.csv", "w")
output_file.write("epoch,loss,acc\n")

class LinearActive:
    def __init__(self, cryptosystem, messenger, *, epochs=100, batch_size=100, learning_rate=0.1):
        self.cryptosystem = cryptosystem
        self.messenger = messenger
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_lambda = 0.01

        self.RESIDUE_PRECISION = 3 # 浮点数舍入精度
        self.activation = lambda x: 1.0 / (1.0 + np.exp(-x)) # 激活函数sigmoid

    def _sync_pubkey(self):
        signal = self.messenger.recv() # receive start signal
        if signal == "START_SIGNAL": 
            print("Training protocol started.")
            print("[ACTIVE] Sending public key to passive party...")
            self.messenger.send(self.cryptosystem.pub_key) # send public key
        else:
            raise ValueError("Invalid signal, exit.")
        print("[ACTIVE] Sending public key done!")

    def _gradient(self, residue, batch_idxes):
        data_grad = self._data_grad_naive(residue, batch_idxes)
        reg_grad = self._reg_grad()
        return data_grad + reg_grad

    def _data_grad_naive(self, residue, batch_idxes):
        data_grad = -1 * (residue[:, np.newaxis] * self.x_train[batch_idxes]).mean(axis=0)
        return data_grad

    def train(self, trainset):
        self.x_train = trainset.features
        self.y_train = trainset.labels
        self.y_val = trainset.labels

        # initialize model parameters
        self._init_weights(trainset.n_features)

        # transmit public key to passive party
        self._sync_pubkey()

        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1

        # Main Training Loop Here
        tbar = trange(self.epochs)
        for epoch in tbar:
            all_idxes = np.arange(n_samples)
            np.random.seed(epoch)
            np.random.shuffle(all_idxes)

            total_loss = 0.0
            total_acc = 0.0

            for batch in range(n_batches):
                # Choose batch indexes
                start = batch * bs
                end = len(all_idxes) if batch == n_batches - 1 else (batch + 1) * bs
                batch_idxes = all_idxes[start:end]

                # Q1. Active party calculates y_hat
                # -----------------------------------------------------------------
                # self.params: (n_features, )
                # self.x_train[batch_idxes]: (batch_size, n_features)
                active_wx = np.dot(self.x_train[batch_idxes], self.params)  # （填空）计算A的wx
                passive_wx = self.messenger.recv()
                full_wx = active_wx + passive_wx  # （填空）综合A、B的wx（纵向联邦综合结果的关键一步）
                y_hat = self.activation(full_wx)
                # -----------------------------------------------------------------

                loss = self._loss(self.y_train[batch_idxes], y_hat)
                acc = self._acc(self.y_train[batch_idxes], y_hat)
                tbar.set_description(f"[loss={loss:.4f}, acc={acc:.4f}]")
                total_acc += acc * len(batch_idxes)
                total_loss += loss * len(batch_idxes)

                residue = self.y_train[batch_idxes] - y_hat
                # residue = residue * y_hat * (1 - y_hat) # sigmoid derivative
                residue = np.array([round(res, self.RESIDUE_PRECISION) for res in residue])

                # Q2. Active party helps passive party to calculate gradient
                # -----------------------------------------------------------------
                enc_residue = self.cryptosystem.encrypt_vector(residue)  # （填空）对误差进行加密
                enc_residue = np.array(enc_residue)
                self.messenger.send(enc_residue)

                enc_passive_grad = self.messenger.recv()
                passive_grad = self.cryptosystem.decrypt_vector(enc_passive_grad)  # （填空）解密得到B的梯度与梯度之和
                self.messenger.send(passive_grad)
                # -----------------------------------------------------------------

                # Active party calculates its own gradient and update model
                active_grad = self._gradient(residue, batch_idxes)
                self._gradient_descent(self.params, active_grad)
            
            total_acc /= n_samples
            total_loss /= n_samples
            output_file.write(f"{epoch},{total_loss},{total_acc}\n")
            output_file.flush()
        print("Finish model training.")
        output_file.close()

    def _init_weights(self, size):
        np.random.seed(0)
        self.params = np.random.normal(0, 1.0, size) # 正态分布初始化参数

    def _reg_grad(self):
        params = self.params
        reg_grad = self.reg_lambda * params
        return reg_grad

    def _gradient_descent(self, params, grad):
        params -= self.learning_rate * grad

    def _loss(self, y_true, y_hat):
        # Logistic regression uses log-loss as loss function
        data_loss = self._logloss(y_true, y_hat)
        reg_loss = self._reg_loss()
        total_loss = data_loss + reg_loss

        return total_loss

    def _acc(self, y_true, y_hat):
        # Q3. Compute accuracy
        # -----------------------------------------------------------------
        acc = np.mean(y_true == (y_hat >= 0.5)) # （填空）计算accuracy
        # 布尔数组与0,1数组在绝大多数情况下等价，布尔数组与0,1数组的比较、布尔数组的求平均值等操作中布尔数组的行为与0,1数组一致
        # -----------------------------------------------------------------
        return acc

    @staticmethod
    def _logloss(y_true, y_hat):
        origin_size = len(y_true)
        if len(np.unique(y_true)) == 1: # 如果y_true中只有一种类别
            if y_true[0] == 0:
                y_true = np.append(y_true, 1)
                y_hat = np.append(y_hat, 1.0)
            else:
                y_true = np.append(y_true, 0)
                y_hat = np.append(y_hat, 0.0)

        return log_loss(y_true=y_true, y_pred=y_hat, normalize=False) / origin_size

    def _reg_loss(self):
        params = self.params
        reg_loss = 1.0 / 2 * self.reg_lambda * (params**2).sum() # L2正则化
        return reg_loss
