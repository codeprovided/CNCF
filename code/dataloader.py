from calendar import day_abbr
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import numpy as np
import pickle
import pandas as pd
import os.path as osp
import itertools
import os
from torch.utils.data import Dataset
import scipy.sparse as sp

class ADataset(InMemoryDataset):
    r'''
        数据集
    '''
    def __init__(self, root, dataset, transform=None, pre_transform=None):
        self.path = root 
        self.dataset = dataset 
        super(ADataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.info = torch.load(self.processed_paths[1])
        self.data_num = self.info['data_num']
        self.feature_num = self.info['feature_num']

    @property
    def raw_dir(self):
        return osp.join(osp.join(self.root,self.dataset), 'raw')
    
    @property
    def raw_file_names(self):
        return ['{}/{}/raw/user.pickle'.format(self.root,self.dataset),
                '{}/{}/raw/items.pickle'.format(self.root,self.dataset),
                '{}/{}/raw/feature.pickle'.format(self.root,self.dataset),
                '{}/{}/raw/train_df.csv'.format(self.root,self.dataset),
                '{}/{}/raw/valid_df.csv'.format(self.root,self.dataset),
                '{}/{}/raw/test_df.csv'.format(self.root,self.dataset)]
       

    @property
    def processed_dir(self):
        return osp.join(osp.join(self.root,self.dataset), 'processed')
    
    @property
    def processed_file_names(self):
        return ['{}.dataset'.format(self.dataset),
                '{}.info'.format(self.dataset)]
        
    
    def download(self):
        pass
        
    def process(self):
        print('process')
        self.userfile  = self.raw_file_names[0]
        self.itemfile  = self.raw_file_names[1]
        self.featurefile = self.raw_file_names[2]
        self.trainfile  = self.raw_file_names[3]
        self.validfile = self.raw_file_names[4]
        self.testfile = self.raw_file_names[5]
        graphs, info = self.getData()
        if not os.path.exists(f"{self.path}/{self.dataset}/processed/"):
            os.mkdir(f"{self.path}/{self.dataset}/processed/")

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(info, self.processed_paths[1])
        
    def featureNum(self):
        return self.feature_num

    def dataNum(self):
        return self.data_num
    
    def getData(self):
        self.userD = pickle.load(open(self.userfile, 'rb'))
        print('userd:',len(self.userD))
        self.itemD = pickle.load(open(self.itemfile, 'rb'))
        print('itemd:',len(self.itemD))
        feature_dict = pickle.load(open(self.featurefile, 'rb'))
       
        train_df = pd.read_csv(self.trainfile, sep=":", header=None)
        print('traindf:',len(train_df))
        valid_df = pd.read_csv(self.validfile, sep=":",header=None)
        print('vailddf:',len(valid_df))
        test_df = pd.read_csv(self.testfile,sep=":",header=None)
        print('testdf',len(test_df))
        self.alllen=len(train_df)+len(valid_df)+len(test_df)
        train_graph=self.bprGraphsFromData(train_df)
        print('train graph finish')
        valid_graph=self.bprGraphsFromData(valid_df)
        print('vaild graph finish')
        test_graph=self.bprGraphsFromData(test_df)
        print('test graph finish')
        graphs = train_graph + valid_graph + test_graph 
        info={}
        info['data_num']=len(graphs)
        info['feature_num']=len(feature_dict)
        info['user_num']=len(self.userD)
        info['item_num']=len(self.itemD)
        info['split_index']=[len(train_graph),len(valid_graph),len(test_graph)]
        return graphs,info
        
    def bprGraphsFromData(self,bpr_df,type='train'):
        graphs=[]
        all_num=len(bpr_df)
        print("all num:",all_num)
        num_graphs = bpr_df.shape[0]
        for i in range(len(bpr_df)):
            if i%(all_num//20)==0:
                print('finish:',5*i/(all_num//20),'%')
            row=bpr_df.iloc[i]
            user_index=row[0]
            item_index=row[1]
           
            ratings=row[2]
            if user_index not in self.userD:
                print('user_index error',user_index)
            user_id=self.userD[user_index]['id']
            if item_index not in self.itemD:
                print("item_index error")
            item_id=self.itemD[item_index]['id']

            user_att=self.userD[user_index]['att']
            item_att=self.itemD[item_index]['att']
  
            userlist=[user_id]+user_att
            itemlist=[item_id]+item_att
      
            graph=self.buildBprGraph(userlist,itemlist,ratings)
            graphs.append(graph)
       
        return graphs
    
    def buildBprGraph(self,userlist,itemlist,ratings):
        graph1=self.buildGraph(userlist,itemlist,ratings)
       
        return graph1
    
   
        
    def buildGraph(self,userlist,itemlist,ratings):
        user_node_number=len(userlist)
        item_node_number=len(itemlist)
        in_edge_index=[[],[]]
        # user
        for i in range(user_node_number):
            for j in range(i,user_node_number):
                in_edge_index[0].append(i)
                in_edge_index[1].append(j)
        # item
        for i in range(user_node_number,user_node_number+item_node_number):
            for j in range(i,user_node_number+item_node_number):
                in_edge_index[0].append(i)
                in_edge_index[1].append(j)
        # #itemN
        in_edge_index=torch.tensor(in_edge_index).long()
        in_edge_index = to_undirected(in_edge_index)             
        #out interaction
        out_edge_index1=[[],[]]
        #u pos
        for i in range(user_node_number):
            for j in range(user_node_number,user_node_number+item_node_number):
                out_edge_index1[0].append(i)
                out_edge_index1[1].append(j)
        out_edge_index1=torch.tensor(out_edge_index1).long()        
        out_edge_index1=to_undirected(out_edge_index1)
        out_edge_index=out_edge_index1
        
        all_node_list=userlist+itemlist
        x=torch.tensor(all_node_list).long().reshape(-1,1)
        y=torch.tensor([user_node_number,item_node_number,ratings]).long()
        
        graph=Data(x=x,edge_index=in_edge_index,edge_attr=torch.transpose(out_edge_index,0,1),y=y)
        return graph  
    

class BDataset(Dataset):
    def __init__(self,dataset,flag):
        super(BDataset,self).__init__()
        self.user=[]
        self.item_i=[]
        self.item_j=[]
        assert flag>=0  and flag<=2,'flag error'
        if flag==0:
            train_df=pd.read_csv('../data/'+str(dataset)+'/graph/train_df.csv',header=None,sep=':')
            self.user=train_df[0]
            self.item_i=train_df[1]
            self.item_j=train_df[2]
            self.len=len(self.user)
          
        elif flag==1:
            valid_df=pd.read_csv('../data/'+str(dataset)+'/graph/valid_df.csv',header=None,sep=':')
            self.user=valid_df[0]
            self.item_i=valid_df[1]
            self.item_j=valid_df[2]
            self.len=len(self.user)
        else:
            test_df=pd.read_csv('../data/'+str(dataset)+'/graph/test_df.csv',header=None,sep=':')
            self.user=test_df[0]
            self.item_i=test_df[1]
            self.item_j=test_df[2]
            self.len=len(self.user)
          
    def __len__(self):
        return self.len
    
    def getlen(self):
        return self.len
    
    def __getitem__(self, index):
        return self.user[index],self.item_i[index],self.item_j[index]



def getUU(dataset,user_num,item_num):
    ui=pd.read_csv('../data/'+str(dataset)+'/graph/train_df.csv',header=None,sep=':')
    urow=ui[0]-1
    icol=ui[1]-1
    
    index=ui[2]==1
    urow=urow[index]
    icol=icol[index]
    
    ui_matrix=sp.csr_matrix((np.ones_like(urow),(urow,icol)),
                                shape=(user_num,item_num))
    uu=ui_matrix.dot(ui_matrix.transpose())
    uu_matrix=sp.csr_matrix((np.ones_like(uu.data),uu.indices,uu.indptr),
                                shape=(user_num,user_num))
    return uu_matrix

    
def getUA(dataset,user_num):
    ua=pd.read_csv('../data/'+str(dataset)+'/graph/user_att.csv',header=None,sep=':')
    user_min_id=min(ua[0])
    urow=ua[0]-user_min_id
    
    att_min_id=min(ua[1])
    att_max_id=max(ua[1])
    att_num=att_max_id-att_min_id+1
    acol=ua[1]-att_min_id
    
    ua_matrix=sp.csr_matrix((np.ones_like(urow),(urow,acol)),
                            shape=(user_num,att_num))
    return ua_matrix
