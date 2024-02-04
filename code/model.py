from asyncio import FastChildWatcher
from cProfile import label
import datetime
from collections import defaultdict
import pickle
from random import sample
from turtle import forward
import torch
from torch import Tensor, dropout, sigmoid, unsqueeze
import torch.nn as nn 
from torch.nn import Parameter 
from torch_geometric.utils import degree
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing,GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool,global_max_pool
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Linear
import numpy as np
import time
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
import pandas as pd
import pickle

class MYGNN(MessagePassing):
    def __init__(self,dim,user_num,item_num,add_self_loops=False):
        super(MYGNN,self).__init__()
        self.dim=dim
        self.info=nn.Linear(self.dim,self.dim)
        self.add_self_loops=add_self_loops
        self.user_num=user_num
        self.item_num=item_num
       
        
    def message(self, x, edge_index):
        row,col=edge_index
        
        mess=self.info(x[row]*x[col])
        emb=scatter(mess,row,dim=0,dim_size=self.user_num+self.item_num,reduce='mean')
        return emb
    
    def aggregate(self,emb):
        return emb 
    
    def update(self, emb):
        return emb       
    
    def propagate(self,x, edge_index):
        x=self.message(x,edge_index)
        x=self.aggregate(x)
        x=self.update(x)
        return x
        
    def forward(self,x,edge_index):
        return self.propagate(x,edge_index)
    
class AName(nn.Module):
    def __init__(self,args,feature_num,user_num,item_num,device,uu_matrix,ua_matrix):
        super(AName,self).__init__()
        
        self.uu_matrix=uu_matrix
        self.ua_matrix=ua_matrix
        
        self.G=self.GCN().to(device)
        self.Gbya=self.generateGfromH().to(device)
        
        self.feature_num=feature_num
        self.device=device
        self.args=args
        self.dim=args.dim
        
        self.all_user_num=user_num
        self.all_item_num=item_num
      
        self.num_user_feature=args.num_user_feature
        self.mask=nn.Embedding(1,self.dim)
        self.feature_embedding = nn.Embedding(self.feature_num + 1, self.dim)
        # torch.nn.init.xavier_normal_(self.feature_embedding.weight)
        
        self.info=nn.Linear(self.dim,self.dim)
        self.binfo=nn.Linear(self.dim,self.dim)
        self.dinfo=nn.Linear(self.dim,self.dim)
        
        self.SSL=nn.Sequential(
            nn.Linear(self.dim,2*self.dim),
            nn.ReLU(),
            nn.Linear(self.dim*2,self.dim),
        )
       
        
        self.node_weight = nn.Embedding(self.feature_num + 1, 1)
        self.node_weight.weight.data.normal_(0.0,0.01)
        
       

        self.lin1=nn.Linear(self.dim,1)
        
        
        self.layernorm=torch.nn.LayerNorm([self.dim])
        
      
        self.gcn1=MYGNN(self.dim,self.all_user_num,self.all_item_num)
        self.gcn2=MYGNN(self.dim,self.all_user_num,self.all_item_num)
        
        self.user_bias=nn.Embedding(user_num+1,1)
        self.item_bias=nn.Embedding(item_num+1,1)
        self.bn=nn.BatchNorm1d(self.dim)
        self.lin=nn.Linear(self.dim,self.dim)
        self.build()
        self.begin()

        
        
        
        self.readout=nn.Linear(self.dim,1,bias=False)
 
        
    def begin(self):
        userD = pickle.load(open('../data/{}/graph/user.pickle'.format(self.args.dataset), 'rb'))
        itemD = pickle.load(open('../data/{}/graph/items.pickle'.format(self.args.dataset), 'rb'))
        self.u_begin=userD[1]['id']
        self.i_begin=itemD[1]['id']
        
    def build(self):
        train_DF=pd.read_csv('../data/{}/graph/train_df.csv'.format(self.args.dataset),header=None,sep=':')
        index=train_DF[2].astype(bool)
        user=train_DF[0][index]-1
        item=train_DF[1][index]+self.all_user_num-1
 
        graph=torch.cat((torch.tensor(np.array(user)).reshape(1,-1),torch.tensor(np.array(item)).reshape(1,-1)),dim=0).long()
        graph=to_undirected(graph)
        self.graph=graph.to(self.device)
        
            
    
    def compute2(self):
        user_emb=self.feature_embedding(torch.arange(self.u_begin,self.u_begin+self.all_user_num).to(self.device))
        
        g1=self.G@user_emb
        g2=self.G@g1
        
        return g2
    
    def compute3(self):
        user_emb=self.feature_embedding(torch.arange(self.u_begin,self.u_begin+self.all_user_num).to(self.device))
        g1=self.Gbya@user_emb
        g2=self.Gbya@g1
       
        return g2
        
    def GCN(self,):
        uu_matrix=self.uu_matrix.toarray()
        H = np.array(uu_matrix)
       
        H=torch.from_numpy(H).float()
       
        DV = torch.sum(H, dim=1) + 1e-5
       
        DE = torch.sum(H, dim=0) + 1e-5
     
        invDE = torch.diag(torch.pow(DE, -0.5))
    
        DV2 = torch.diag(torch.pow(DV, -0.5))
    
        G = DV2@H@invDE
        
        return G
    
    
    def generateGfromH(self,):
        ua_matrix=self.ua_matrix.toarray()
        H = np.array(ua_matrix)
        H=torch.from_numpy(H).float()
        DV = torch.sum(H, dim=1) + 1e-5
        DE = torch.sum(H, dim=0) + 1e-5
        invDE = torch.diag(torch.pow(DE, -1))
        DV2 = torch.diag(torch.pow(DV, -1))
        HT = H.T
        
        
        G = DV2@H@invDE@HT
        return G
    
    def forward(self,data,userB,itemB):
    
        predictAui,sslloss=self.forwardone(data,userB,itemB)
        predictBui=self.forwardtow(userB,itemB,data.y)
        return predictAui,predictBui,sslloss
    
    def forwardtow(self,userB,itemB,datay):
        
        x2=self.compute2()
        userembx2=x2[userB-1]
        
        x3=self.compute3()
        userembx3=x3[userB-1]
        
        useremb=userembx2+userembx3
        
        allitem_emb=self.feature_embedding(torch.arange(self.i_begin,self.i_begin+self.all_item_num).to(self.device))
        itemiemb=allitem_emb[itemB-1]
        
        user_bias=self.user_bias(userB)
        item_i_bias=self.item_bias(itemB)
        
        input=useremb*itemiemb
        
        predictui=self.readout(input)
        
        predictui=predictui.reshape(-1)+user_bias.reshape(-1)+item_i_bias.reshape(-1)

        return   predictui
        
    def forwardone(self,data,userB,itemB,train=True):
        node_id=data.x
        batch=data.batch
        datay=data.y.reshape(-1,3)[:,:2].reshape(-1)
        true_label=data.y.reshape(-1,3)[:,2].reshape(-1)
        y=datay.reshape(-1,2)

        node_w = torch.squeeze(self.node_weight(node_id)).reshape(-1,1)
        ui_sum_weight = global_add_pool(node_w, data.batch)
        node_emb = self.feature_embedding(node_id)
        
        node_emb = node_emb.squeeze()
        
        in_edge_index = data.edge_index
        out_edge_index = torch.transpose(data.edge_attr, 0, 1)
        out_edge_index = self.offset(out_edge_index,batch,self.num_user_feature,y.reshape(-1,2))
        
        left=node_emb[in_edge_index[0,:]]
        right=node_emb[in_edge_index[1,:]]
        
        n_info=self.info(left*right)
        
        y=datay.reshape(-1,2)
        
        in_node=global_mean_pool(n_info,torch.repeat_interleave(torch.arange(datay.sum()).to(self.device),torch.repeat_interleave(datay.to(self.device),datay.to(self.device))))
        
        left=node_emb[out_edge_index[0,:]]
        right=in_node[out_edge_index[1,:]]
        
        n_info_b=self.binfo(left*right)
        
        in_node_b=global_mean_pool(n_info_b,torch.repeat_interleave(torch.arange(torch.sum(datay)).to(self.device),torch.repeat_interleave(torch.cat((y[:,1].reshape(-1,1),y[:,0].reshape(-1,1)),dim=1).reshape(-1).to(self.device),datay.to(self.device))))
     
        left=in_node[out_edge_index[0,:]]
        right=node_emb[out_edge_index[1,:]]
        n_info_d=self.dinfo(left*right)
        
        
        in_node_d=global_mean_pool(n_info_d,torch.repeat_interleave(torch.arange(torch.sum(datay)).to(self.device),torch.repeat_interleave(torch.cat((y[:,1].reshape(-1,1),y[:,0].reshape(-1,1)),dim=1).reshape(-1).to(self.device),datay.to(self.device))))

        ui_gru_output=node_emb+in_node+in_node_b+in_node_d

        ui_gru_output=ui_gru_output
        y=datay.reshape(-1,2)
        
        ui_gru_output=ui_gru_output

        sample=torch.rand(len(datay)//2,self.args.sample_num).to(self.device)
        sample_index=sample*torch.sum(y,dim=1,keepdim=True)
        sample_index=torch.floor(sample_index)
        offset=torch.cat((torch.tensor([0]).to(self.device),torch.cumsum(torch.sum(y,dim=1),dim=0)[:-1]),dim=0).reshape(-1,1)
        sample_index=sample_index+offset
        sample_index=sample_index.long()

        graph_emb=global_mean_pool(ui_gru_output,torch.repeat_interleave(torch.arange(len(datay)//2).to(self.device),torch.sum(y,dim=1).reshape(-1).to(self.device)))
        #[batch_size,sample_num,dim]
        target=ui_gru_output[sample_index]
        target=target/torch.sum(y,dim=1).reshape(-1,1,1)
        input=graph_emb.unsqueeze(1)-target
        
        input=input.reshape(-1,self.dim)
        target=target.reshape(-1,self.dim)

        input=self.SSL(input)
        outout=self.readout(input).reshape(-1)
        label=self.readout(target).reshape(-1)
        
        loss=torch.nn.MSELoss()

        sslloss=loss(outout,label)
        
        predict=self.readout(graph_emb)+ui_sum_weight
 
        predict=predict.reshape(-1)
        
        
        
        return predict,sslloss
        
   
    
    def sample_index(self,indices,sample_num,sample_all_num):

        graph_num=len(indices)
        indices=indices.reshape(-1)
        offset=torch.cat((torch.zeros(1),torch.ones(graph_num-1)*sample_all_num)).to(self.device).long()
        offset=torch.cumsum(offset,dim=0)
        
        offset=torch.repeat_interleave(offset,(torch.ones(graph_num,dtype=torch.long)*sample_num).to(self.device))
        
        return indices+offset
      
        
    def offset(self,out_edge_index,batch,user_fn,y):

        y=y.reshape(-1,2)
        ones = torch.ones_like(batch).to(self.device)
        node_num_graph=global_add_pool(ones.reshape(-1,1), batch).reshape(-1).to(self.device)
        edge_num_graph=2*(y[:,0]*y[:,1]) #size[graph_num]
        cum_num = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(node_num_graph, dim=0)[:-1]))
        offset_list = torch.repeat_interleave(cum_num, edge_num_graph, dim=0).repeat(2, 1)
        outer_edge_index_offset = out_edge_index + offset_list
        return outer_edge_index_offset
    
    def buildGATinput(self,ui_emb,y,edge_index):
        u_emb=ui_emb[torch.repeat_interleave(torch.tensor([True,False]).to(self.device).repeat(1,len(y)//2),y)]
        i_emb=ui_emb[torch.repeat_interleave(torch.tensor([False,True]).to(self.device).repeat(1,len(y)//2),y)]
        
        return u_emb,i_emb
    
    
    
    
    
    
