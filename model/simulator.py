from .model import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os



class Simulator(nn.Module):

    def __init__(self, message_passing_num, device, node_input_size=4, edge_input_size=3, hidden_size=64, output_size=1, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, 
                                             node_input_size=node_input_size, 
                                             edge_input_size=edge_input_size, 
                                             output_size=output_size,
                                             hidden_size=hidden_size).to(device)
        self._output_normalizer = normalization.Normalizer(size=output_size, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)
        # self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        print('Simulator model initialized')

    def update_node_attr(self, frames, graph:Data):
        node_feature = []

        node_feature.append(frames) # pressure
        node_feature.append(graph.pos) # xyz
        node_feature.append(graph.x[:, 1:2]) # time
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, accumulate=self.training)

        return attr

    def velocity_to_accelation(self, noised_frames, next_velocity):

        acc_next = next_velocity - noised_frames
        return acc_next


    def forward(self, graph:Data, velocity_sequence_noise):
        
        if self.training:
            
            frames = graph.x[:, :1]
            target = graph.y
            noised_frames = frames + velocity_sequence_noise
            node_attr = self.update_node_attr(noised_frames, graph)
            graph.x = node_attr
            predicted = self.model(graph)
            
            # target_acceration = self.velocity_to_accelation(noised_frames, target)
            target_normalized = self._output_normalizer(target, self.training)

            return predicted.squeeze(), target_normalized.squeeze()

        else:
    
            frames = graph.x[:, :1]
            node_attr = self.update_node_attr(frames, graph)
            graph.x = node_attr
            predicted = self.model(graph)

            predicted_unnormalized = self._output_normalizer.inverse(predicted)
            #predicted_velocity = frames + velocity_update # only when prediction is accleration

            return predicted_unnormalized.squeeze()

    def load_checkpoint(self, ckpdir=None):
        
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir)
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.'+k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s"%ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir = self.model_dir

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer  = self._node_normalizer.get_variable()
        # _edge_normalizer = self._edge_normalizer.get_variable()

        to_save = {'model':model, '_output_normalizer':_output_normalizer, '_node_normalizer':_node_normalizer}

        torch.save(to_save, savedir)
        print('Simulator model saved at %s'%savedir)