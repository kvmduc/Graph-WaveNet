import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
################ Path to data ################
parser.add_argument('--data',type=str,default='/data/cs.aau.dk/tungkvt/Trafficstream/district3F11T17/FastData/',help='data path')
parser.add_argument('--adjdata',type=str,default='/data/cs.aau.dk/tungkvt/Trafficstream/district3F11T17/graph/',help='adj data path')
################              ################
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=655,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.01,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=1,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='/data/cs.aau.dk/tungkvt/Trafficstream/result/graph-wavenet/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--begin_year',type=int,default=2017,help='begin year')
parser.add_argument('--end_year',type=int,default=2017,help='end year')
parser.add_argument('--begin_test_year',type=int,default=2011,help='begin year of testing catastrophic forgetting')

args = parser.parse_args()

logger = util.init_log()



def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    for year in range (args.begin_year, args.end_year + 1):


        adj_path = osp.join(args.adjdata, str(year)+"_adj.npz")
        # # data_path = osp.join(args.data, year, str(year)+"_30day.npz")
        adj_mx = util.load_adj(adj_path,args.adjtype)
        dataloader = util.load_dataset(year, args.data, args.batch_size, args.batch_size, args.batch_size)

        
        # scaler = dataloader['scaler']
        ############################################### TEST OLD DATA ###############################################
        # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(str(args.adjdata),args.adjtype)
        # dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)  
        
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        vars(args)['num_nodes'] = supports[0].shape[0]

        logger.info(args)

        if args.randomadj:
            adjinit = None
        else:
            adjinit = supports[0]

        if args.aptonly:
            supports = None

        

        engine = trainer(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                            args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                            adjinit)


        logger.info("start training year {}".format(year))
        his_loss =[]
        val_time = []
        train_time = []
        for i in range(1,args.epochs+1):
            #if i % 10 == 0:
                #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                #for g in engine.optimizer.param_groups:
                    #g['lr'] = lr
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)
                trainx= trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)
                # print(trainx.shape)             # (batchsize, F, N, len)
                # print(trainy.shape)             # (batchsize, F, N, len)
                metrics = engine.train(trainx, trainy[:,0,:,:])
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                if iter % args.print_every == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    logger.info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))

            t2 = time.time()
            train_time.append(t2-t1)
            #validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []


            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:,0,:,:])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            logger.info(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            logger.info(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))
            torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        #testing
        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))


        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1,3)[:,0,:,:]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)
            with torch.no_grad():
                preds = engine.model(testx).transpose(1,3)
            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs,dim=0)
        yhat = yhat[:realy.size(0),...]


        logger.info("Training finished")
        logger.info("The valid loss on best model is {}".format(str(round(his_loss[bestid],4))))


        amae = []
        amape = []
        armse = []
        for i in [2,5,11]:
            # pred = scaler.inverse_transform(yhat[:,:,i])
            pred = yhat[:,:,i]
            real = realy[:,:,i]
            metrics = util.metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            logger.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")

        # Masking and test previous year

        for test_year in range(args.end_year - 1, args.begin_test_year, -1):
            data_test = util.load_masked_test_dataset(test_year, args.data, args.batch_size, args.num_nodes)
            
            outputs = []
            realy = torch.Tensor(data_test['y_test']).to(device)
            realy = realy.transpose(1,3)[:,0,:,:]  

            for iter, (x, y) in enumerate(data_test['test_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1,3)
                with torch.no_grad():
                    preds = engine.model(testx).transpose(1,3)
                outputs.append(preds.squeeze())
            
            yhat = torch.cat(outputs,dim=0)
            yhat = yhat[:realy.size(0),...]
            
            mmae = []
            mmape = []
            mrmse = []

            for i in [2,5,11]:
                # pred = scaler.inverse_transform(yhat[:,:,i])
                pred = yhat[:,:,i]
                real = realy[:,:,i]
                metrics = util.metric(pred,real)
                log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
                logger.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
                mmae.append(metrics[0])
                mmape.append(metrics[1])
                mrmse.append(metrics[2])

            log = 'On average over 12 horizons of YEAR : {}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            logger.info(log.format(test_year, np.mean(mmae),np.mean(mmape),np.mean(mrmse)))




if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
