from torch.utils.data import IterableDataset
import os, numpy as np
import os.path as osp
import h5py
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
import torch
import math
import time

class BLEVEBase():

    def __init__(self, max_epochs=1, graph='r', k=7, radius=0.8, files=None):

        self.open_tra_num = 5    # This influnces memory usage 
        self.file_handle = files
        self.shuffle_file()

        self.data_keys =  ('grid', 'pressure')

        self.tra_index = 0
        self.epcho_num = 1
        self.tra_readed_index = -1

        # dataset attr
        self.tra_len = 25    #!!! the number of timestep, be cautious when dataset changed
        self.time_iterval = 0.001

        self.opened_tra = []
        self.opened_tra_readed_index = {}
        self.opened_tra_readed_random_index = {}
        self.tra_data = {}
        self.max_epochs = max_epochs #not in use
        
        # graph connection
        self.graph = graph
        self.k = k
        self.radius = radius
    
    def open_tra(self):
        while(len(self.opened_tra) < self.open_tra_num):

            tra_index = self.datasets[self.tra_index]

            if tra_index not in self.opened_tra:
                self.opened_tra.append(tra_index)
                self.opened_tra_readed_index[tra_index] = -1
                self.opened_tra_readed_random_index[tra_index] = np.random.permutation(self.tra_len - 1)   # The last 2 timesteps are used for gt only

            self.tra_index += 1
            if self.check_if_epcho_end():
                self.epcho_end()
    
    def check_and_close_tra(self):
        to_del = []
        for tra in self.opened_tra:
            if self.opened_tra_readed_index[tra] >= (self.tra_len - 2):
                to_del.append(tra)
        for tra in to_del:
            self.opened_tra.remove(tra)
            try:
                del self.opened_tra_readed_index[tra]
                del self.opened_tra_readed_random_index[tra]
                del self.tra_data[tra]
            except Exception as e:
                print(e)
                

    def shuffle_file(self):
        datasets = list(self.file_handle.keys())
        np.random.shuffle(datasets)
        self.datasets = datasets

    def epcho_end(self):
        self.tra_index = 0
        self.shuffle_file()
        self.epcho_num = self.epcho_num + 1

    def check_if_epcho_end(self):
        if self.tra_index >= len(self.file_handle):
            return True
        return False

    @staticmethod
    def datas_to_graph(datas, graph='r', k=6, radius=0.8):
        #datas is a list of ["grid", "pressure", "time"]
        grid, current_p, next_p, next2_p, next3_p, time = datas[0], datas[1][0], datas[1][1], datas[1][2], datas[1][3], datas[2]
        num_nodes = current_p.shape[0]
        time_vector = np.ones((num_nodes, 1)) * time
        
        # Node attribute contains node_type, current pressure, time
        node_attr = np.hstack((current_p[:, None], time_vector))   
        node_attr = torch.as_tensor(node_attr, dtype=torch.float32)
        
        # grid coordinates
        grid = torch.as_tensor(grid, dtype=torch.float32)
        
        # target is the pressure of next time step     
        next_p = torch.as_tensor(next_p, dtype=torch.float32)
        next2_p = torch.as_tensor(next2_p, dtype=torch.float32)
        next3_p = torch.as_tensor(next3_p, dtype=torch.float32)
        
        # Graph connection by knn or radius
        if graph == 'r':
            edge_index = radius_graph(x=grid, r=radius, loop=True, num_workers=4)
        elif graph == 'knn':
            edge_index = knn_graph(x=grid, k=k, loop=True, num_workers=4)
        else:
            raise ValueError("graph type not implemented.")          
        
        g = Data(x=node_attr, y=next_p, edge_index=edge_index, pos=grid, next_y=next2_p, next2_y=next3_p)

        return g


    def __next__(self):
   
        self.check_and_close_tra()
        self.open_tra()
        
        # Stop dataloader by max number of epochs
        # if self.epcho_num > self.max_epochs:
        #     raise StopIteration

        selected_tra = np.random.choice(self.opened_tra)

        data = self.tra_data.get(selected_tra, None)
        if data is None:
            data = self.file_handle[selected_tra]
            self.tra_data[selected_tra] = data

        selected_tra_readed_index = self.opened_tra_readed_index[selected_tra]
        selected_frame = self.opened_tra_readed_random_index[selected_tra][selected_tra_readed_index+1]
        self.opened_tra_readed_index[selected_tra] += 1

        datas = []
        for k in self.data_keys:
            if k in [ 'grid']:
                r = np.array((data[k]), dtype=np.float32)
            elif k in ['pressure']:
                r = np.array((data[k][selected_frame], data[k][selected_frame+1], data[k][selected_frame+1], data[k][selected_frame+1]), dtype=np.float32)
            datas.append(r)
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))
        #("node_type", "grid", "pressure", "time"
        g = self.datas_to_graph(datas, graph=self.graph, k=self.k, radius=self.radius)
  
        return g

    def __iter__(self):
        return self


class BLEVE(IterableDataset):
    def __init__(self,dataset_dir, max_epochs=0, graph='r', k=6, radius=0.8, split='train') -> None:

        super().__init__()

        dataset_dir = osp.join(dataset_dir, split+'.h5')
        self.max_epochs = max_epochs
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        self.graph = graph
        self.k = k
        self.radius = radius
        print('Dataset '+  self.dataset_dir + ' Initilized')

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_handle)
        else:
            per_worker = int(math.ceil(len(self.file_handle)/float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_handle))

        keys = list(self.file_handle.keys())
        keys = keys[iter_start:iter_end]
        files = {k: self.file_handle[k] for k in keys}
        return BLEVEBase(max_epochs=self.max_epochs, files=files, graph=self.graph, k=self.k, radius=self.radius)


class BLEVE_ROLLOUT(IterableDataset):
    def __init__(self, dataset_dir, split='test', name='bleve rollout', graph='r', k=6, radius=0.8):

        dataset_dir = osp.join(dataset_dir, split+'.h5')
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        self.data_keys =  ("grid", "pressure")
        self.time_iterval = 0.001
        self.load_dataset()
        self.graph = graph
        self.k = k
        self.radius = radius

    def load_dataset(self):
        datasets = list(self.file_handle.keys())   # a list of file name, e.g. 100045
        self.datasets = datasets

    def change_file(self, file_index):
        
        self.file_index = self.datasets[file_index]
        self.cur_tra = self.file_handle[self.file_index]
        self.cur_targecity_length = self.cur_tra['pressure'].shape[0]
        self.cur_tragecity_index = 0
        self.edge_index = None

    def __next__(self):
        if self.cur_tragecity_index==(self.cur_targecity_length - 1):
            raise StopIteration

        datas = []
        data = self.cur_tra
        selected_frame = self.cur_tragecity_index

        datas = []
        for k in self.data_keys:
            if k in ['grid']:
                r = np.array((data[k]), dtype=np.float32)
            elif k in ['pressure']:
                r = np.array((data[k][selected_frame], data[k][selected_frame+1], data[k][selected_frame+1], data[k][selected_frame+1]), dtype=np.float32)
            datas.append(r)
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))

        self.cur_tragecity_index += 1
        ## For benchmark runtime
        # if self.cur_tragecity_index==(self.cur_targecity_length - 1):
        #     self.cur_tragecity_index = (self.cur_targecity_length - 2)
        
        g = BLEVEBase.datas_to_graph(datas, graph=self.graph, k=self.k, radius=self.radius)
        return g


    def __iter__(self):
        return self

