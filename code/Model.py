### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
from tqdm import tqdm
import sys

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)
        self.model_setup()

    def model_setup(self):
        # send it to GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.network.to(self.device)

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):

        # load checkpoint to continue from prev training point
        # checkpointfile = os.path.join('../saved_models/wrn_v2/', 'model-%d.ckpt'%(50))
        # self.load(checkpointfile)

        # define loss_fn, optimizer, batch_size and epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=configs["learning_rate"],
            weight_decay=configs["weight_decay"],
            momentum=configs["momentum"],
            nesterov=True)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[80,140,200],gamma=0.2)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[20,40],gamma=0.2)
        max_epoch = configs['max_epoch']

        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // configs['batch_size']

        print('### Training... ###')
        qbar = tqdm(range(1,max_epoch+1),position=0, file=sys.stdout)
        for epoch in qbar:
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # keep track of validation accuracy and loss for plotting
            val_acc_history = np.array([],dtype=np.float32)

            qbar1 = tqdm(range(num_batches),position=1,leave=False ,file=sys.stdout)
            for i in qbar1:
                # Construct the current batch.
                X_batch = curr_x_train[i*configs['batch_size']:min((i+1)*configs['batch_size'],curr_x_train.shape[0])]
                y_batch = curr_y_train[i*configs['batch_size']:min((i+1)*configs['batch_size'],curr_y_train.shape[0])]
                X_batch = np.array(list(map(lambda x: parse_record(x,True),X_batch))) # gotta check whether a lambda works as a map function over numpy array
                X_batch = torch.tensor(X_batch,device=self.device, dtype=torch.float)
                pred = self.network(X_batch,True)
                y_batch = torch.tensor(y_batch,device=self.device)
                loss = self.loss_fn(pred,y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                qbar1.set_description('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss))
            
            self.scheduler.step()
            duration = time.time() - start_time
            qbar.set_description('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % configs['save_interval'] == 0:
                self.save(epoch)
                if x_valid is not None and y_valid is not None:
                    val_acc = self.evaluate(x_valid,y_valid,validation=True)
                    row = np.array([val_acc], dtype=np.float32)
                    val_acc_history = np.hstack((val_acc_history, row))
                    highest_acc_epoch=np.argmax(val_acc_history)
                    print('Maximum accuracy on val set at {:d} epoch is {:.4f}'.format(highest_acc_epoch+1,val_acc_history[highest_acc_epoch]))

                
                
                

    def evaluate(self, x, y,validation=True,checkpoint_list=[]):
        if validation:
            print('### Validation ###')
            preds = []
            with torch.no_grad():
                for i in tqdm(range(x.shape[0])):
                    ### YOUR CODE HERE
                    x_processed = np.array(list(map(lambda x: parse_record(x,False),x[i:i+1])))
                    x_processed = torch.tensor(x_processed,device=self.device,dtype=torch.float)
                    preds.append(torch.argmax(self.network(x_processed,False),axis=1))
                    ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            acc = torch.sum(preds==y)/y.shape[0]
            print('Test accuracy: {:.4f}'.format(acc))
            return acc
        else:
            print('### Test ###')
            for checkpoint in checkpoint_list:
                checkpointfile = os.path.join(self.configs['save_dir'], 'model-%d.ckpt'%(checkpoint))
                self.load(checkpointfile)
                preds = []
                with torch.no_grad():
                    for i in tqdm(range(x.shape[0])):
                        x_processed = np.array(list(map(lambda x: parse_record(x,False),x[i:i+1])))
                        x_processed = torch.tensor(x_processed,device=self.device,dtype=torch.float)
                        preds.append(torch.argmax(self.network(x_processed,False),axis=1))

                    y = torch.tensor(y)
                    preds = torch.tensor(preds)
                    acc = torch.sum(preds==y)/y.shape[0]
                    print('Checkpoint {:d} Test accuracy: {:.4f}'.format(checkpoint,acc))


    def predict_prob(self, x):
        checkpointfile = os.path.join(self.configs['save_dir'], 'model-%d.ckpt'%(50))
        self.load(checkpointfile)
        preds = []
        with torch.no_grad():
            for i in tqdm(range(x.shape[0])):
                x_processed = np.array(list(map(lambda x: parse_record(x,False),x[i:i+1])))
                x_processed = torch.tensor(x_processed,device=self.device,dtype=torch.float)
                preds.append(self.network(x_processed,False).cpu().detach().numpy())
        preds = np.array(preds)
        preds = np.squeeze(preds,axis=1)
        assert preds.shape==(x.shape[0],self.configs['num_classes'])
        return preds
                
            



    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs['save_dir'], 'model-%d.ckpt'%(epoch))
        os.makedirs(self.configs['save_dir'], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))


### END CODE HERE