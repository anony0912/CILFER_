
import os
import numpy as np
import torch
import os.path as osp
import subprocess
try:
    import cPickle as pickle
except:
    import pickle

from sklearn.metrics import confusion_matrix


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def forgetting(accuracies):
    if len(accuracies) == 1:
        return 0.

    last_accuracies = accuracies[-1]
    usable_tasks = last_accuracies.keys()
    max_id = len(usable_tasks)
    forgetting = 0.
    for id, task in enumerate(usable_tasks):
        max_task = 0.
        for task_accuracies in accuracies[:-1]:
            if task in task_accuracies:
                max_task = max(max_task, task_accuracies[task])

        if id != max_id-1:
            forgetting += max_task - last_accuracies[task]
        

    return round(forgetting / (len(usable_tasks)-1),2)

def accuracy(y_pred, y_true, nb_old, nb_base,increment=10):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)

    cm = confusion_matrix(y_true,y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return all_acc

def accuracy_per_task(ypreds, ytrue, task_size=10):
    if isinstance(task_size,list):
        all_acc_per_task = dict()
        sum = 0
        test_list = []
        for i in range(len(task_size)):
            test_list.append(sum)
            sum += task_size[i]
        test_list.append(sum)
        for i_class_id in range(1,len(test_list)):
            if test_list[i_class_id] > np.max(ytrue)+1:
                break
            idxes = np.where(np.logical_and(ytrue >= test_list[i_class_id-1], ytrue < test_list[i_class_id]))[0]
            label = "{}-{}".format(str(test_list[i_class_id-1]).rjust(2, "0"),str(test_list[i_class_id]-1).rjust(2, "0"))
            all_acc_per_task[label] = round((ypreds[idxes]==ytrue[idxes]).sum() / len(ypreds[idxes])*100,2)
        return all_acc_per_task
    else:
        raise ValueError('please input list of incremental phase.')


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def savepickle(data, file_path):
    mkdir_p(osp.dirname(file_path), delete=False)
    print('pickle into', file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def mkdir_p(path, delete=False, print_info=True):
    if path == '': return

    if delete:
        subprocess.call(('rm -r ' + path).split())
    if not osp.exists(path):
        if print_info:
            print('mkdir -p  ' + path)
        subprocess.call(('mkdir -p ' + path).split())
        
        
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count