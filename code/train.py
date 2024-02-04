from cProfile import label
from collections import defaultdict
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import ndcg_score
from dataloader import ADataset,BDataset
import argparse
from torch_geometric.loader import DataLoader as ALoader
from torch.utils.data import DataLoader as BLoader
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
import torch
import os
import datetime

def getTestDf():
    test_df=pd.read_csv('../data/ml/raw/testsort_df.csv',sep=":",header=None)
    df=defaultdict(list)
    for index,row in test_df.iterrows():
        df[row[0]].append(row[1])
    return df


def getAllDf():
    allratings=pd.read_csv('../data/ml/raw/allRating.csv',sep=":",header=None)
    df=defaultdict(list)
    for index,row in allratings.iterrows():
        df[row[0]].append(row[1])
    return df

def Train(args,epochs,model,train_loaderA,train_loaderB,lr,l2_weight,train_num):
    device=torch.device( 'cuda' if torch.cuda.is_available()  else 'cpu')
    model=model.to(device)
    model.train()
    crit = torch.nn.BCELoss()
    crit2 = torch.nn.MSELoss()

    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=l2_weight)
    for _ in range(1):
        epoch_loss=0
        epoch_sslloss=0
        t_A=train_loaderA._get_iterator()
        t_B=train_loaderB._get_iterator()
        # assert len(t_A)==len(t_B),'train len error'
        maxl=len(t_A)
     
        starttime = datetime.datetime.now()
        for tt in range(maxl):
            opt.zero_grad()            
            dataA=t_A.next()
            dataA=dataA.to(device)           
            dataB=t_B.next()
            userB=dataB[0]
            userB=userB.to(device)
            item_i_B=dataB[1]
            item_i_B=item_i_B.to(device)
            label=dataB[2]
            label=label.to(device)
  
            ui_out_A,ui_out_B,sslloss=model(dataA,userB,item_i_B)       
            predict=args.weightA*ui_out_A+args.weightB*ui_out_B
            
            loss1=crit(torch.sigmoid(predict),label.float())
   
            loss=loss1+args.lamb*sslloss     
            loss=loss.to(device)  
            epoch_sslloss+=(sslloss.item()*predict.size(0))/train_num          
            epoch_loss+=(loss.item()*predict.size(0))/train_num

            loss.backward()
   
            opt.step()
  
        print('epoch:',epochs,',averageloss:',epoch_loss)
        if args.lamb==0:
            print('epoch:{},averageloss:{}'.format(epochs,epoch_loss))
        else:
            print('epoch:{},averageloss:{:7f},sslloss:{}'.format(epochs,epoch_loss,epoch_sslloss))
     


def Test(args,model,test_loaderA,test_loaderB,top_k,numofuser,device,weightA,weightB):
    starttime = datetime.datetime.now()
    model=model.to(device)
    model.eval()
    device=torch.device( 'cuda' if torch.cuda.is_available()  else 'cpu')
    t_A=test_loaderA._get_iterator()
    t_B=test_loaderB._get_iterator()
    assert len(t_A)==len(t_B),'test len error'
    maxl=len(t_A)

    predictions = []
    labels = []
    user_ids = []
    for _ in range(maxl):
        data=t_A.next()
        dataB=t_B.next()
        data=data.to(device)
    
        userB=dataB[0]
        userB=userB.to(device)
        item_i_B=dataB[1]
        item_i_B=item_i_B.to(device)

        _, user_id_index = np.unique(data.batch.detach().cpu().numpy(), return_index=True)
        user_id = data.x.detach().cpu().numpy()[user_id_index]
        user_ids.append(user_id)

        data = data.to(device)

        predA,predB,_=model(data,userB,item_i_B)
        
        pred=torch.sigmoid(weightA*predA+weightB*predB)
        
        pred = pred.squeeze().detach().cpu().numpy().astype('float64')
        if pred.size == 1:
            pred = np.expand_dims(pred, axis=0)
        label =dataB[2].cpu().numpy()
        predictions.append(pred)
        labels.append(label)
      
    endtime = datetime.datetime.now()
    
    pre_k,recall_k,ndcg_k=newevaluate(predictions,labels,user_ids,top_k)

    for i in range(len(top_k)):    
        print('top_{}:,pre:{:.5f},recall:{:.5f},ndcg:{:.5f}'.format(top_k[i],pre_k[i],recall_k[i],ndcg_k[i]))
      



def cal_ndcg(predicts, labels, user_ids, k_list):

    d = {'user': np.squeeze(user_ids), 'predict':np.squeeze(predicts), 'label':np.squeeze(labels)}
    df = pd.DataFrame(d)
    user_unique = df.user.unique()

    ndcgs = [[] for _ in range(len(k_list))]
    for user_id in user_unique:
        user_srow = df.loc[df['user'] == user_id]
        upred = user_srow['predict'].tolist()
        if len(upred) < 2:
            #print('less than 2', user_id)
            continue
 
        ulabel = user_srow['label'].tolist()


        for i in range(len(k_list)):
            ndcgs[i].append(ndcg_score([ulabel], [upred], k=k_list[i])) 

    ndcg_mean =  np.mean(np.array(ndcgs), axis=1)
    return ndcg_mean

def cal_recall(predicts, labels, user_ids, k):
    d = {'user': np.squeeze(user_ids), 'predict':np.squeeze(predicts), 'label':np.squeeze(labels)}
    df = pd.DataFrame(d)
    user_unique = df.user.unique()
    pre=[]
    recall = []
    for user_id in user_unique:
        user_sdf = df[df['user'] == user_id]
        if user_sdf.shape[0] < 2:
            #print('less than 2', user_id)
            continue
        user_sdf = user_sdf.sort_values(by=['predict'], ascending=False)
        total_rel = min(user_sdf['label'].sum(), k)
        #total_rel = user_sdf['label'].sum()
        intersect_at_k = user_sdf['label'][0:k].sum()

        try:
            recall.append(float(intersect_at_k)/float(total_rel))
            pre.append(float(intersect_at_k)/float(k))
        except:
            continue

    return np.mean(np.array(pre)),np.mean(np.array(recall))

def newevaluate(predictions,labels,user_ids,top_k):
    predictions = np.concatenate(predictions, 0)
    labels = np.concatenate(labels, 0)
    user_ids = np.concatenate(user_ids, 0)

    labels = labels.astype(int)
    ndcg_list = cal_ndcg(predictions, labels, user_ids, top_k)
    pre5,recall5 = cal_recall(predictions, labels, user_ids, top_k[0]) 
    pre10,recall10 = cal_recall(predictions, labels, user_ids, top_k[1]) 
    pre20,recall20 = cal_recall(predictions, labels, user_ids, top_k[2]) 
    

    return  (pre5,pre10,pre20),(recall5, recall10, recall20),ndcg_list




  
