from shutil import copyfile

import math
import torch
import torch.nn as nn
import test

from torch.autograd import Variable
import logging
import sklearn.metrics.pairwise as smp
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import time

logger = logging.getLogger("logger")
import os
import json
import numpy as np
import config
import copy
import utils.csv_record
import main


def get_krum_scores(grads, groupsize):
    krum_scores = np.zeros(len(grads))
    cur_time = time.time()
    num_clients = len(grads)
    #grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
    #grads = np.zeros((num_clients, grad_len))

    #for i in range(len(client_grads)):
        #grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len)) if len(client_grads) !=0 else 0

    #for i in range (len(X)):
        #for name, data in X[i].items():
            #print (name)
            #if name == 'fc.weight':
                #detached_data= data.cpu().detach().numpy()
            # print(detached_data.shape)
            #detached_data=detached_data.tolist()
            # print(detached_data)
                #X_list.append(detached_data)
            #X_list = np.array(X_list)
    #X_list1 =np.array(X_list)
    squared_grads = [x**2 for x in grads]
    #distances = torch.sum(torch.pow(X_list, 2)) + torch.sum(torch.pow(X_list,2)) -2 * np.dot(X_list1, X_list1.T)
    distances = np.sum(squared_grads, axis =1)+ np.sum(squared_grads, axis = 1)- 2 * np.dot(grads, grads.T)
    #print(distances)

    #return math.sqrt(squared_sum)
        #for name, layer in X[i].items():
            #print(name)
            #data = data.numpy()
            #X_list1.append([x for x in layer.data])
            #X_list2.append([x**2 for x in layer.data])
    #X_list1= np.array(X_list1)
    #X_list2 = np.array(X_list2)#for name, data in updates.items():
        #client_grads.append(data[1])  # gradient
        #alphas.append(data[0])  # num_samples
        #names.append(name)X = [x**2 for x in X]X = [x**2 for x in X]
    #distances = np.sum(X_list2, axis =0)[:,None]+ np.sum(X_list2, axis = 0)[None] - (2 * np.dot(X_list,X_list.T))
    for i in range(len(grads)):
        krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])
    return krum_scores


class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'./saved_models/mnist_pretrain/model_{self.name}_{current_time}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.fg= FoolsGold(use_memory=self.params['fg_use_memory'])

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum

    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var= sum_var.to(config.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = Variable(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(
            self.params['scale_weights'] * (model_vec - target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        logger.info("los")
        logger.info(cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1 - cs_sim

        return 1e3 * loss

    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            model_update = 100 * (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
                name].view(-1)

            cs = F.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))
        logger.info(model_id)
        logger.info((sum(cs_list) / len(cs_list)).data[0])
        return 1e3 * sum(cos_los_submit)

    def accum_similarity(self, last_acc, new_acc):

        cs_list = list()

        cs_loss = torch.nn.CosineSimilarity(dim=0)
        # logger.info('new run')
        for name, layer in last_acc.items():
            cs = cs_loss(Variable(last_acc[name], requires_grad=False).view(-1),
                         Variable(new_acc[name], requires_grad=False).view(-1))
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))
        # logger.info("AAAAAAAA")
        # logger.info((sum(cs_list)/len(cs_list)).data[0])
        return sum(cos_los_submit)

    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def accumulate_weight(self, weight_accumulator, epochs_submit_update_dict, state_keys,num_samples_dict):
        """
         return Args:
             updates: dict of (num_samples, update), where num_samples is the
                 number of training samples corresponding to the update, and update
                 is a list of variable weights
         """
        if self.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_gradients = epochs_submit_update_dict[state_keys[i]][0] # agg 1 interval
                num_samples = num_samples_dict[state_keys[i]]
                updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
            return None, updates
        elif self.params['aggregation_methods'] == config.AGGR_KRUM:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_gradients = epochs_submit_update_dict[state_keys[i]][0] # agg 1 interval
                num_samples = num_samples_dict[state_keys[i]]
                updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
            return None, updates

        else:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                update= dict()
                num_samples=num_samples_dict[state_keys[i]]

                for name, data in local_model_update_list[0].items():
                    update[name] = torch.zeros_like(data)

                for j in range(0, len(local_model_update_list)):
                    local_model_update_dict= local_model_update_list[j]
                    for name, data in local_model_update_dict.items():
                        weight_accumulator[name].add_(local_model_update_dict[name])
                        update[name].add_(local_model_update_dict[name])
                        detached_data= data.cpu().detach().numpy()
                        # print(detached_data.shape)
                        detached_data=detached_data.tolist()
                        # print(detached_data)
                        local_model_update_dict[name]=detached_data # from gpu to cpu

                updates[state_keys[i]]=(num_samples,update)

            return weight_accumulator,updates

    def init_weight_accumulator(self, target_model):
        weight_accumulator = dict()
        for name, data in target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        return weight_accumulator


    def average_shrink_models(self, weight_accumulator, target_model, epoch_interval):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["no_models"])
            # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])

            # update_per_layer = update_per_layer * 1.0 / epoch_interval
            if self.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))

            data.add_(update_per_layer)
        return True


    def krum_update (self,target_model,updates,clip):
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[krum agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[krum agg] considering poison per batch poison_fraction: {poison_fraction}')

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        n = len(client_grads)
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
        grads = np.zeros((num_clients, grad_len))

        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len)) if len(client_grads) !=0 else 0
        scores = get_krum_scores(grads,n - clip)
        good_idx = np.argpartition(scores, n - clip)[:(n - clip)]
        print('good_idx', good_idx)
        agg_grads = []
        for i in range(len(client_grads[0])):
            #assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c in good_idx:
                    temp += client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)

        #print(np.shape(T))
        #T_t = torch.as_tensor(T)
        #client_grads_mean = np.mean(agg_grads, axis = 0)
        for i, (name, params) in enumerate(target_model.named_parameters()):
            #if (i in good_idx):
            agg_grads[i] = agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()
        #wv=wv.tolist()
        #utils.csv_record.add_weight_result(names, wv, alpha)
        #client_grads=torch.from_numpy(client_grads).float()
        return True, names,client_grads


    def foolsgold_update(self,target_model,updates):
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[foolsgold agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[foolsgold agg] considering poison per batch poison_fraction: {poison_fraction}')

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        agg_grads, wv,alpha = self.fg.aggregate_gradients(client_grads,names)
        for i, (name, params) in enumerate(target_model.named_parameters()):
            agg_grads[i]=agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()
        wv=wv.tolist()
        utils.csv_record.add_weight_result(names, wv, alpha)
        return True, names, wv, alpha, client_grads

    def contra_update(self,target_model,updates, reputation_dict):
        client_grads = []
        epsilon=1E-5
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        #grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[contra agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[contra agg] considering poison per batch poison_fraction: {poison_fraction}')

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()

        agg_grads, wv,alpha, avg_cs, cs_sorted = self.fg.aggregate_gradients(client_grads,names)
        for i, (name, params) in enumerate(target_model.named_parameters()):
                agg_grads[i]=agg_grads[i] * self.params["eta"]
                if params.requires_grad:
                    params.grad = agg_grads[i].to(config.device)
        #target_model2 = target_model
        #losses = []
        #for i in range(len(client_grads)):
            #agg_grads=self.fg.exc_one(i,wv,client_grads)


            #for j, (name, params) in enumerate(target_model2.named_parameters()):
                #agg_grads[j]=agg_grads[j] * self.params["eta"]
                #if params.requires_grad:
                    #params.grad = agg_grads[j].to(config.device)
            #epoch_loss1 = self.lossfunc(target_model)
            #epoch_loss2 = self.lossfunc(target_model2)
            #loss = (epoch_loss1 - epoch_loss2)
            #losses.append(loss)
        #bad_idx_loss = np.argpartition(losses,-5)[-5:]
        for i in range(len(client_grads)):
            _name = names[i]
            if avg_cs[i] >=0.9: #and i in bad_idx_loss:
                #wv[i] = 0
                reputation_dict[_name] = reputation_dict[_name] - 0.5
            else:
                reputation_dict[_name] = reputation_dict[_name] + 0.5

        wv = (1 - (np.mean(cs_sorted, axis = 1))) + (np.mean(cs_sorted, axis = 1) - alpha)
        optimizer.step()
        wv=wv.tolist()
        utils.csv_record.add_weight_result(names, wv, alpha)
        return True, names, wv, alpha, client_grads, reputation_dict


    def geometric_median_update(self, target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm= None):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
               """
        points = []
        alphas = []
        names = []
        for name, data in updates.items():
            points.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)

        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        alphas = torch.from_numpy(alphas).float()

        # alphas.float().to(config.device)
        median = Helper.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = Helper.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            logger.info('Starting Weiszfeld algorithm')
            logger.info(log_entry)
        logger.info(f'[rfa agg] init. name: {names}, weight: {alphas}')
        # start
        wv=None
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, Helper.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = Helper.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = Helper.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         Helper.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                logger.info(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
            logger.info(f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: { abs(prev_obj_val - obj_val)}')
            logger.info(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv=copy.deepcopy(weights)
        alphas = [Helper.l2dist(median, p) for p in points]

        update_norm = 0
        for name, data in median.items():
            update_norm += torch.sum(torch.pow(data, 2))
        update_norm= math.sqrt(update_norm)

        if max_update_norm is None or update_norm < max_update_norm:
            for name, data in target_model.state_dict().items():
                update_per_layer = median[name] * (self.params["eta"])
                if self.params['diff_privacy']:
                    update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
                data.add_(update_per_layer)
            is_updated = True
        else:
            logger.info('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            is_updated = False

        utils.csv_record.add_weight_result(names, wv.cpu().numpy().tolist(), alphas)

        return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(),alphas

    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        squared_sum = 0
        for name, data in p1.items():
            squared_sum += torch.sum(torch.pow(p1[name]- p2[name], 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """Compute geometric median objective."""
        temp_sum= 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * Helper.l2dist(median, p)
        return temp_sum

        # return sum([alpha * Helper.l2dist(median, p) for alpha, p in zip(alphas, points)])

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)

        weighted_updates= dict()

        for name, data in points[0].items():
            weighted_updates[name]=  torch.zeros_like(data)
        for w, p in zip(weights, points): # 对每一个agent
            for name, data in weighted_updates.items():
                temp = (w / tot_weights).float().to(config.device)
                temp= temp* (p[name].float())
                # temp = w / tot_weights * p[name]
                if temp.dtype!=data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates

    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def update_epoch_submit_dict(self,epochs_submit_update_dict,global_epochs_submit_dict, epoch,state_keys):

        epoch_len= len(epochs_submit_update_dict[state_keys[0]])
        for j in range(0, epoch_len):
            per_epoch_dict = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                local_model_update_dict = local_model_update_list[j]
                per_epoch_dict[state_keys[i]]= local_model_update_dict

            global_epochs_submit_dict[epoch+j]= per_epoch_dict

        return global_epochs_submit_dict


    def save_epoch_submit_dict(self, global_epochs_submit_dict):
        with open(f'{self.folder_path}/epoch_submit_update.json', 'w') as outfile:
            json.dump(global_epochs_submit_dict, outfile, ensure_ascii=False, indent=1)

    def estimate_fisher(self, model, criterion,
                        data_loader, sample_size, batch_size=64):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        if self.params['type'] == 'text':
            data_iterator = range(0, data_loader.size(0) - 1, self.params['bptt'])
            hidden = model.init_hidden(self.params['batch_size'])
        else:
            data_iterator = data_loader

        for batch_id, batch in enumerate(data_iterator):
            data, targets = self.get_batch(data_loader, batch,
                                           evaluation=False)
            if self.params['type'] == 'text':
                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, self.n_tokens), targets)
            else:
                output = model(data)
                loss = log_softmax(output, dim=1)[range(targets.shape[0]), targets.data]
                # loss = criterion(output.view(-1, ntokens
            # output, hidden = model(data, hidden)
            loglikelihoods.append(loss)
            # loglikelihoods.append(
            #     log_softmax(output.view(-1, self.n_tokens))[range(self.params['batch_size']), targets.data]
            # )

            # if len(loglikelihoods) >= sample_size // batch_size:
            #     break
        logger.info(loglikelihoods[0].shape)
        # estimate the fisher information of the parameters.
        loglikelihood = torch.cat(loglikelihoods).mean(0)
        logger.info(loglikelihood.shape)
        loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, model, fisher):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            model.register_buffer('{}_estimated_fisher'
                                  .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, lamda, cuda=False):
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_estimated_mean'.format(n))
                fisher = getattr(model, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    def lossfunc(self,model):
        epsilon=1E-5
        model.eval()
        total_loss = 0
        correct = 0
        correct2=0
        dataset_size = 0
        if self.params['type'] == config.TYPE_LOAN:
            for i in range(0, len(self.allStateHelperList)):
                state_helper = self.allStateHelperList[i]
                data_iterator = state_helper.get_testloader()
                for batch_id, batch in enumerate(data_iterator):
                    data, targets = state_helper.get_batch(data_iterator, batch, evaluation=True)
                    dataset_size += len(data)
                    output = model(data)
                    total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        elif self.params['type'] == config.TYPE_CIFAR \
                or self.params['type'] == config.TYPE_MNIST \
                or self.params['type'] == config.TYPE_TINYIMAGENET:
            data_iterator = self.test_data
            for batch_id, batch in enumerate(data_iterator):
                data, targets = self.get_batch(data_iterator, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                targetIdx = np.where(targets.data.view_as(pred).cpu().sum().item == 4)
                correct1 = np.where(pred[targetIdx] == 4)
                attacked1 = np.where(pred[targetIdx] == 9)
                attacked2 = len(pred[targetIdx] == 9)/(len(pred[targetIdx]==4) + epsilon)
                correct2 += pred.eq(targets.data.view_as(pred)).cpu().sum().item()==9

        acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
        attacked3 = 100.0 * (float(correct2) / float(dataset_size))  if dataset_size!=0 else 0
        total_l = total_loss / dataset_size if dataset_size!=0 else 0

        main.logger.info('Average loss: {:.4f}'.format(total_l))
        model.train()
        return (total_l)

class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict=dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self,client_grads,names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # if self.memory is None:
        #     self.memory = np.zeros((num_clients, grad_len))
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len)) if len(client_grads) !=0 else 0
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]]+=grads[i]
            else:
                self.memory_dict[names[i]]=copy.deepcopy(grads[i])
            self.memory[i]=self.memory_dict[names[i]]
        # self.memory += grads

        if self.use_memory:
            wv, alpha,avg_cs, cs_sorted = self.foolsgold_new(self.memory)  # Use FG
        else:
            wv, alpha, avg_cs,cs_sorted = self.foolsgold_new(grads)  # Use FG
        logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        agg_grads = []

        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)


        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha, avg_cs, cs_sorted

    def foolsgold(self,grads):
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)
        avg_cs = []

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        k = 5
        losses =[]
        epsilon = 1E-5
        for i in range(n_clients):
            temp=0
            good_idx =[[]]
            good_idx = np.argpartition(cs[i], (n_clients - k))[(n_clients - k):n_clients]
            for j in range(5):
                good_idx_data = cs[i][good_idx[j]]
                temp += cs[i][good_idx[j]]
            temp = temp /5
            avg_cs.append(temp)
        avg_cs = np.array(avg_cs)
        maxcs = 1 - (np.max(cs, axis = 1))
        wv = 1 - (np.max(cs, axis=1))

        cs_sorted = np.sort(cs[:,-5:], axis = 0)
        cs_sorted_mean = (np.mean(cs_sorted, axis = 1))
        #target_model = helper.target_model
        #epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_lossfunc(helper,model=target_model, is_poison=False,visualize=True, agent_name_key="global")
        #csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

        #target_model2 = target_model
        #for i in range(len(grads)):
            #agg_grads=self.exc_one(i,wv,grads)

            #for j, (name, params) in enumerate(target_model2.named_parameters()):
                #agg_grads[j]=agg_grads[j] * self.params["eta"]
                #if params.requires_grad:
                    #params.grad = agg_grads[j].to(config.device)
            #epoch_loss2,_,_,_ = test.Mytest_lossfunc(helper=helper,model=target_model2,is_poison=False,visualize=True,agent_name_key="global")
            #loss = (epoch_loss - epoch_loss2)
            #losses.append(loss)
        #bad_idx_loss = np.argpartition(losses,-5)[-5:]
        #for i in range(n_clients):
            #if avg_cs[i] >=0.9 and i in bad_idx_loss:
                #wv[i] = epsilon
            #else:
        #wv = (1 - (np.mean(cs_sorted, axis = 1))) + (np.mean(cs_sorted, axis = 1) - np.max(cs , axis = 1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv,alpha

    def exc_one(self,ex_cl,wv,client_grads):
        agg_grads = []

        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i]
            temp = temp * 0.0
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == ex_cl:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        return agg_grads


    def foolsgold_new(self,grads):
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)
        avg_cs = []
        epsilon = 1E-5

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / (maxcs[j] + epsilon)

        k = 5
        losses =[]
        epsilon = 1E-5
        for i in range(n_clients):
            temp=0
            good_idx =[[]]
            good_idx = np.argpartition(cs[i], (n_clients - k))[(n_clients - k):n_clients]
            for j in range(5):
                good_idx_data = cs[i][good_idx[j]]
                temp += cs[i][good_idx[j]]
            temp = temp /5
            avg_cs.append(temp)
        avg_cs = np.array(avg_cs)
        maxcs = 1 - (np.max(cs, axis = 1))
        wv = 1 - (np.max(cs, axis=1))

        cs_sorted = np.sort(cs[:,-5:], axis = 0)
        cs_sorted_mean = (np.mean(cs_sorted, axis = 1))
        #for i in range(len(grads)):
            #agg_grads=self.exc_one(i,wv,grads)

            #for j, (name, params) in enumerate(target_model2.named_parameters()):
                #agg_grads[j]=agg_grads[j] * self.params["eta"]
                #if params.requires_grad:
                    #params.grad = agg_grads[j].to(config.device)
            #epoch_loss2,_,_,_ = test.Mytest_lossfunc(helper=helper,model=target_model2,is_poison=False,visualize=True,agent_name_key="global")
            #loss = (epoch_loss - epoch_loss2)
            #losses.append(loss)
        bad_idx_loss = np.argpartition(losses,-5)[-5:]
        for i in range(n_clients):
            if avg_cs[i] >=0.9 and i in bad_idx_loss:
                wv[i] = epsilon
            else:
                wv = (1 - (np.mean(cs_sorted, axis = 1))) + (np.mean(cs_sorted, axis = 1) - np.max(cs , axis = 1))
        wv[wv > (1 + epsilon)] = 1
        wv[wv < (0 + epsilon)] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / (np.max(wv) + epsilon)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > (1 + epsilon))] = 1
        wv[(wv < (0 + epsilon))] = 0

        # wv is the weight
        return wv,alpha, avg_cs, cs_sorted
