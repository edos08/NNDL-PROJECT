import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm.contrib.telegram import tqdm


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.encoder_k = base_encoder

        hidden_dim = 512

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))

        # enc_q = self.encoder_q.cuda()
        # enc_k = self.encoder_k.cuda()
        # summary(enc_q, input_size=(3, 224, 224), batch_size=256, device='cuda')
        # summary(enc_k, input_size=(3, 224, 224), batch_size=256, device='cuda')

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("index_queue", torch.zeros(K).long())
        self.index_queue -= 1

        self.register_buffer("label_labeled_queue", torch.zeros(137726).long())
        self.label_labeled_queue -= 1

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _init_label_information(self, index, label):
        """
        Update index_labeled_queue and label_labeled_queue
        """
        self.label_labeled_queue[index] = label
    
    @torch.no_grad()
    def _show_label_information(self):
        # print(self.label_labeled_queue)
        print(torch.where(self.label_labeled_queue >= 0)[0].shape)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, indexes):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.index_queue[ptr:ptr + batch_size] = indexes
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, index):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        labels = self.label_labeled_queue[index]  # correspondence target  B
        queue_l = self.label_labeled_queue[self.index_queue]
        # compute logits
        features = torch.cat((q, k, self.queue.clone().detach().t()), dim=0).requires_grad_(True)
        target = torch.cat((labels, labels, queue_l.clone().detach()), dim=0).requires_grad_(False)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, index)

        return features, target
    

def train(train_loader: DataLoader, model: nn.Module, epoch: int, criterion: nn.modules.loss, optimizer: torch.optim,
          device, pbar: tqdm = None) -> float:
    """
    Train a model on the provided training dataset

    Args:
         train_loader (DataLoader): training dataset
         model (nn.Module): model to train
         epoch (int): the current epoch
         criterion (nn.modules.loss): loss function
         optimizer (torch.optim): optimizer for the model
         device (torch.device): the device to load the model
         pbar (tqdm): tqdm progress bar
    """

    model.train()
    start = time.time()

    epoch_loss = np.array([])

    for images, _, index in train_loader:
        # GPU casting
        images[0] = images[0].to(device)
        images[1] = images[1].to(device)
        index = index.to(device)

        if images[0].shape[0] != 256:
            print("Batch size is not 256, skipping batch: ", images[0].shape[0])
            continue

        # Forward step
        features, targets = model(im_q=images[0], im_k=images[1], index=index)
        loss = criterion(features, targets)

        epoch_loss = np.append(epoch_loss, loss.cpu().data)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pbar is not None:
            pbar.update(1)

    epoch_loss_mean = epoch_loss.mean()
    epoch_loss_std = epoch_loss.std()

    end = time.time()

    print("\n-- TRAINING --")
    print("Epoch: ", epoch + 1, "\n"
          " - Loss: ", epoch_loss_mean, " +- ", epoch_loss_std, "\n "
          " - Time: ", end - start, "s")

    return epoch_loss_mean
