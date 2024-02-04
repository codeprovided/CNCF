from re import A
from dataloader import ADataset,BDataset, getUU ,getUA
import argparse
from torch_geometric.loader import DataLoader as ALoader
from torch.utils.data import DataLoader as BLoader
from train import *
from model import *
import torch
import os
import pickle
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml', help='which dataset to use')
parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--num_workers',type=int,default=4)
parser.add_argument('--lamb',type=float,default=1)
parser.add_argument('--weightA',type=float,default=0.5)
parser.add_argument('--weightB',type=float,default=0.5)
parser.add_argument('--drop_p',type=float,default=0.8)
parser.add_argument('--sample_num',type=int,default=50,help='the number of sampled feature')
parser.add_argument('--hhh',type=str,default='test')


args = parser.parse_args()
if args.dataset == 'ml':
    args.num_user_feature = 4
    args.test_per_user=128
elif args.dataset == 'taobao':
    args.num_user_feature = 8 
    args.test_per_user=128
datasetA = ADataset('../data/', args.dataset)
args.user_num=datasetA.info['user_num']
args.item_num=datasetA.info['item_num']
trainA_num,validA_num,testA_num=datasetA.info['split_index']

trainsetA=datasetA[:trainA_num]
# validsetA=datasetA[trainA_num:trainA_num+validA_num]
testsetA=datasetA[trainA_num+validA_num:trainA_num+validA_num+testA_num]

train_loaderA=ALoader(trainsetA,batch_size=args.batch_size,num_workers=4)
# valid_loaderA=ALoader(validsetA,batch_size=2*args.batch_size,num_workers=4)
test_loaderA=ALoader(testsetA,batch_size=args.test_per_user,num_workers=1)

data_num=datasetA.dataNum()
feature_num=datasetA.featureNum()

trainsetB=BDataset(args.dataset,flag=0)
# validsetB=BDataset(args.dataset,flag=1)
testsetB=BDataset(args.dataset,flag=2)

print('len(trainsetB)',len(trainsetB))
train_loaderB=BLoader(trainsetB,batch_size=args.batch_size,num_workers=4)
# valid_loaderB=BLoader(validsetB,batch_size=args.batch_size,num_workers=4)
test_loaderB=BLoader(testsetB,batch_size=args.test_per_user,num_workers=1)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

uu_matrix=getUU(args.dataset,args.user_num,args.item_num)
ua_matrix=getUA(args.dataset,args.user_num)

model=AName(args,feature_num,args.user_num,args.item_num,device,uu_matrix,ua_matrix)
topk=[5,10,20]



for epoch in range(args.epochs):
    Train(args,epoch,model,train_loaderA,train_loaderB,args.lr,args.l2_weight,trainA_num)
    Test(args,model,test_loaderA,test_loaderB,topk,args.user_num,device,args.weightA,args.weightB)

torch.save(model.state_dict(),'mlmodel.pt')
print('finish')
