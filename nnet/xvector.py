# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2019 Yevhenii Prokopalo, Anthony Larcher


The authors would like to thank the BUT Speech@FIT group (http://speech.fit.vutbr.cz) and Lukas BURGET
for sharing the source code that strongly inspired this module. Thank you for your valuable contribution.
"""

import h5py
import logging
import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from collections import OrderedDict
from sidekit.nnet.xsets import XvectorMultiDataset, XvectorDataset, StatDataset
from sidekit.bosaris import IdMap
from sidekit.statserver import StatServer


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reS'





def split_file_list(batch_files, num_processes):
    # Cut the list of files into args.num_processes lists of files
    batch_sub_lists = [[]] * num_processes
    x = [ii for ii in range(len(batch_files))]
    for ii in range(num_processes):
        batch_sub_lists[ii - 1] = [batch_files[z + ii] for z in x[::num_processes] if (z + ii) < len(batch_files)]
    return batch_sub_lists


class Xtractor(torch.nn.Module):
    def __init__(self, spk_number):
        super(Xtractor, self).__init__()
        self.frame_conv0 = torch.nn.Conv1d(20, 512, 5)
        self.frame_conv1 = torch.nn.Conv1d(512, 512, 3, dilation=2)
        self.frame_conv2 = torch.nn.Conv1d(512, 512, 3, dilation=3)
        self.frame_conv3 = torch.nn.Conv1d(512, 512, 1)
        self.frame_conv4 = torch.nn.Conv1d(512, 1500, 1)
        self.seg_lin0 = torch.nn.Linear(3000, 512)
        self.seg_lin1 = torch.nn.Linear(512, 512)
        self.seg_lin2 = torch.nn.Linear(512, spk_number)
        #
        self.norm0 = torch.nn.BatchNorm1d(512)
        self.norm1 = torch.nn.BatchNorm1d(512)
        self.norm2 = torch.nn.BatchNorm1d(512)
        self.norm3 = torch.nn.BatchNorm1d(512)
        self.norm4 = torch.nn.BatchNorm1d(1500)
        self.norm6 = torch.nn.BatchNorm1d(512)
        #
        self.activation = torch.nn.Softplus()

    def forward(self, x):
        frame_emb_0 = self.norm0(self.activation(self.frame_conv0(x)))
        frame_emb_1 = self.norm1(self.activation(self.frame_conv1(frame_emb_0)))
        frame_emb_2 = self.norm2(self.activation(self.frame_conv2(frame_emb_1)))
        frame_emb_3 = self.norm3(self.activation(self.frame_conv3(frame_emb_2)))
        frame_emb_4 = self.norm4(self.activation(self.frame_conv4(frame_emb_3)))
        # Pooling Layer that computes mean and standard devition of frame level embeddings
        # The output of the pooling layer is the first segment-level representation
        mean = torch.mean(frame_emb_4, dim=2)
        std = torch.std(frame_emb_4, dim=2)
        seg_emb_0 = torch.cat([mean, std], dim=1)
        # No batch-normalisation after this layer
        seg_emb_1 = self.activation(self.seg_lin0(seg_emb_0))
        # new layer with batch Normalization
        seg_emb_2 = self.norm6(self.activation(self.seg_lin1(seg_emb_1)))
        # No batch-normalisation after this layer
        # seg_emb_3 = self.activation(self.seg_lin2(seg_emb_2))
        seg_emb_3 = self.seg_lin2(seg_emb_2)
        return seg_emb_3

    def LossFN(self, x, lable):
        loss = - torch.trace(torch.mm(torch.log10(x), torch.t(lable)))
        return loss

    def init_weights(self):
        """
        """
        torch.nn.init.normal_(self.frame_conv0.weight, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.frame_conv1.weight, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.frame_conv2.weight, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.frame_conv3.weight, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.frame_conv4.weight, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.seg_lin0.weight, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.seg_lin1.weight, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.seg_lin2.weight, mean=-0.5, std=1.)

        torch.nn.init.normal_(self.frame_conv0.bias, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.frame_conv1.bias, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.frame_conv2.bias, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.frame_conv3.bias, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.frame_conv4.bias, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.seg_lin0.bias, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.seg_lin1.bias, mean=-0.5, std=1.)
        torch.nn.init.normal_(self.seg_lin2.bias, mean=-0.5, std=1.)

    def extract(self, x):
        frame_emb_0 = self.norm0(self.activation(self.frame_conv0(x)))
        frame_emb_1 = self.norm1(self.activation(self.frame_conv1(frame_emb_0)))
        frame_emb_2 = self.norm2(self.activation(self.frame_conv2(frame_emb_1)))
        frame_emb_3 = self.norm3(self.activation(self.frame_conv3(frame_emb_2)))
        frame_emb_4 = self.norm4(self.activation(self.frame_conv4(frame_emb_3)))
        # Pooling Layer that computes mean and standard devition of frame level embeddings
        # The output of the pooling layer is the first segment-level representation
        mean = torch.mean(frame_emb_4, dim=2)
        std = torch.std(frame_emb_4, dim=2)
        seg_emb = torch.cat([mean, std], dim=1)
        # No batch-normalisation after this layer
        # seg_emb_1 = self.activation(self.seg_lin0(seg_emb_0))

        seg_emb_A = self.seg_lin0(seg_emb)
        seg_emb_B = self.seg_lin1(self.activation(seg_emb_A))

        # return torch.nn.functional.softmax(seg_emb_3,dim=1)
        return seg_emb_A, seg_emb_B


def xtrain(args):
    # Initialize a first model and save to disk
    model = Xtractor(args.class_number)
    current_model_file_name = "initial_model"
    torch.save(model.state_dict(), current_model_file_name)

    for epoch in range(1, args.epochs + 1):
        current_model_file_name = train_epoch(epoch, args, current_model_file_name)

        # Add the cross validation here
        accuracy = cross_validation(args, current_model_file_name)
        print("*** Cross validation accuracy = {.02f} %".format(accuracy))


def train_epoch(epoch, args, initial_model_file_name):
    # Compute the megabatch number
    with open(args.batch_training_list, 'r') as fh:
        batch_file_list = [l.rstrip() for l in fh]

    # Shorten the batch_file_list to be a multiple of

    megabatch_number = len(batch_file_list) // (args.averaging_step * args.num_processes)
    megabatch_size = args.averaging_step * args.num_processes
    print("Epoch {}, number of megabatches = {}".format(epoch, megabatch_number))

    current_model = initial_model_file_name

    # For each sublist: run an asynchronous training and averaging of the model
    for ii in range(megabatch_number):
        print('Process megabatch [{} / {}]'.format(ii + 1, megabatch_number))
        current_model = train_asynchronous(epoch,
                                           args,
                                           current_model,
                                           batch_file_list[megabatch_size * ii: megabatch_size * (ii + 1)],
                                           ii,
                                           megabatch_number)  # function that split train, fuse and write the new model to disk
    return current_model


def train_worker(rank, epoch, args, initial_model_file_name, batch_list, output_queue):
    model = Xtractor(args.class_number)
    model.load_state_dict(torch.load(initial_model_file_name))
    model.train()

    torch.manual_seed(args.seed + rank)
    train_loader = XvectorMultiDataset(batch_list, args.batch_path)

    device = torch.device("cuda:{}".format(rank))
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr = args.lr)
    optimizer = optim.Adam([{'params': model.frame_conv0.parameters(), 'weight_decay': args.l2_frame},
                            {'params': model.frame_conv1.parameters(), 'weight_decay': args.l2_frame},
                            {'params': model.frame_conv2.parameters(), 'weight_decay': args.l2_frame},
                            {'params': model.frame_conv3.parameters(), 'weight_decay': args.l2_frame},
                            {'params': model.frame_conv4.parameters(), 'weight_decay': args.l2_frame},
                            {'params': model.seg_lin0.parameters(), 'weight_decay': args.l2_seg},
                            {'params': model.seg_lin1.parameters(), 'weight_decay': args.l2_seg},
                            {'params': model.seg_lin2.parameters(), 'weight_decay': args.l2_seg}
                            ])


    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    # criterion = torch.nn.CrossEntropyLoss()

    accuracy = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()

        accuracy += (torch.argmax(output.data, 1) == target.to(device)).sum()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(
                epoch, batch_idx + 1, train_loader.__len__(),
                       100. * batch_idx / train_loader.__len__(), loss.item(),
                       100.0 * accuracy.item() / ((batch_idx + 1) * args.batch_size)))

    model_param = OrderedDict()
    params = model.state_dict()

    for k in list(params.keys()):
        model_param[k] = params[k].cpu().detach().numpy()
    output_queue.put(model_param)


def train_asynchronous(epoch, args, initial_model_file_name, batch_file_list, megabatch_idx, megabatch_number):
    # Split the list of files for each process
    sub_lists = split_file_list(batch_file_list, args.num_processes)

    #
    output_queue = mp.Queue()
    # output_queue = multiprocessing.Queue()

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train_worker,
                       args=(rank, epoch, args, initial_model_file_name, sub_lists[rank], output_queue)
                       )
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)

    # Average the models and write the new one to disk
    asynchronous_model = []
    for ii in range(args.num_processes):
        asynchronous_model.append(dict(output_queue.get()))

    for p in processes:
        p.join()

    av_model = Xtractor(args.class_number)
    tmp = av_model.state_dict()

    average_param = dict()
    for k in list(asynchronous_model[0].keys()):
        average_param[k] = asynchronous_model[0][k]

        for mod in asynchronous_model[1:]:
            average_param[k] += mod[k]

        if 'num_batches_tracked' not in k:
            tmp[k] = torch.FloatTensor(average_param[k] / len(asynchronous_model))

    # return the file name of the new model
    current_model_file_name = "{}/model_{}_epoch_{}_batch_{}".format(args.model_path, args.expe_id, epoch,
                                                                     megabatch_idx)
    torch.save(tmp, current_model_file_name)
    if megabatch_idx == megabatch_number:
        torch.save(tmp, "{}/model_{}_epoch_{}".format(args.model_path, args.expe_id, epoch))

    return current_model_file_name

def cross_validation(args, current_model_file_name):
    """

    :param args:
    :param current_model_file_name:
    :return:
    """
    with open(args.cross_validation_list, 'r') as fh:
        cross_validation_list = [l.rstrip() for l in fh
        sub_lists = split_file_list(cross_validation_list, args.num_processes)

    #
    output_queue = mp.Queue()

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=cv_worker,
                       args=(rank, args, current_model_file_name, sub_lists[rank], output_queue)
                       )
        # We first evaluate the model across `num_processes` processes
        p.start()
        processes.append(p)

    # Average the models and write the new one to disk
    result = []
    for ii in range(args.num_processes):
        result.append(output_queue.get())

    for p in processes:
        p.join()

    # Compute the global accuracy
    accuracy = 0.0
    total_batch_number = 0
    for batch_number, acc in result:
        accuracy += acc
        total_batch_number += batch_number

    return 100. * accuracy / (total_batch_number * args.batch_size)


def cv_worker(rank, args, current_model_file_name, batch_list, output_queue):
    model = Xtractor(args.class_number)
    model.load_state_dict(torch.load(current_model_file_name))
    model.eval()

    cv_loader = XvectorMultiDataset(batch_list, args.batch_path)

    device = torch.device("cuda:{}".format(rank))
    model.to(device)

    accuracy = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data.to(device))
        accuracy += (torch.argmax(output.data, 1) == target.to(device)).sum()

    output_queue.put((train_loader.__len__(), accuracy))

def extract_idmap(args, device_ID, segment_indices, fs_params, idmap_name, output_queue):
    """
    Function that takes a model and an idmap and extract all x-vectors based on this model
    and return a StatServer containing the x-vectors
    """
    device = torch.device("cuda:{}".format(device_ID))

    # Create the dataset
    tmp_idmap = IdMap(idmap_name)
    idmap = IdMap()
    idmap.leftids = tmp_idmap.leftids[segment_indices]
    idmap.rightids = tmp_idmap.rightids[segment_indices]
    idmap.start = tmp_idmap.start[segment_indices]
    idmap.stop = tmp_idmap.stop[segment_indices]

    segment_loader = StatDataset(idmap, fs_params)

    # Load the model
    model_file_name = '/'.join([args.model_path, args.model_name])
    model = Xtractor(args.class_number)
    model.load_state_dict(torch.load(model_file_name))
    model.eval()

    # Get the size of embeddings
    emb_a_size = model.seg_lin0.weight.data.shape[0]
    emb_b_size = model.seg_lin1.weight.data.shape[0]

    # Create a Tensor to store all x-vectors on the GPU
    emb_A = numpy.zeros((idmap.leftids.shape[0], emb_a_size)).astype(numpy.float32)
    emb_B = numpy.zeros((idmap.leftids.shape[0], emb_b_size)).astype(numpy.float32)

    # Send on selected device
    model.to(device)

    # Loop to extract all x-vectors
    for idx, (model_id, segment_id, data) in enumerate(segment_loader):
        print('Extract X-vector for {}\t[{} / {}]'.format(segment_id, idx, segment_loader.__len__()))
        print("shape of data = {}".format(list(data.shape)))
        print("shape[2] = {}".format(list(data.shape)[2]))
        if list(data.shape)[2] < 20:
            pass
        else:
            A, B = model.extract(data.to(device))
            emb_A[idx, :] = A.detach().cpu()
            emb_B[idx, :] = B.detach().cpu()

    output_queue.put((segment_indices, emb_A, emb_B))


def extract_parallel(args, fs_params, dataset):
    emb_a_size = 512
    emb_b_size = 512

    if dataset == 'enroll':
        idmap_name = args.enroll_idmap
    elif dataset == 'test':
        idmap_name = args.test_idmap
    elif dataset == 'back':
        idmap_name = args.back_idmap

    idmap = IdMap(idmap_name)
    x_server_A = StatServer(idmap, 1, emb_a_size)
    x_server_B = StatServer(idmap, 1, emb_b_size)
    x_server_A.stat0 = numpy.ones(x_server_A.stat0.shape)
    x_server_B.stat0 = numpy.ones(x_server_B.stat0.shape)

    # Split the indices
    mega_batch_size = idmap.leftids.shape[0] // args.num_processes
    segment_idx = []
    for ii in range(args.num_processes):
        segment_idx.append(
            numpy.arange(ii * mega_batch_size, numpy.min([(ii + 1) * mega_batch_size, idmap.leftids.shape[0]])))

    # Extract x-vectors in parallel
    output_queue = mp.Queue()

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=extract_idmap,
                       args=(args, rank, segment_idx[rank], fs_params, idmap_name, output_queue)
                       )
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)

    # Get the x-vectors and fill the StatServer
    for ii in range(args.num_processes):
        indices, A, B = output_queue.get()
        x_server_A.stat1[indices, :] = A
        x_server_B.stat1[indices, :] = B

    for p in processes:
        p.join()

    return x_server_A, x_server_B
