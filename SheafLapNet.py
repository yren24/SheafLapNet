import argparse, sys, time, random, torch, re, joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def dataset_list(filename):
    dataset = []
    fp = open(filename)
    for line in fp:
        line_split = re.split(',|\n', line)
        dataset.append(line_split[:-1])
    fp.close()
    return dataset

def normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def RMSE(ypred, yexact):
    return torch.sqrt(torch.sum((ypred-yexact)**2)/ypred.shape[0])

def PCC(ypred, yexact):
    a = ypred.cpu().numpy().ravel()
    b = yexact.cpu().numpy().ravel()
    pcc = stats.pearsonr(a, b)
    return pcc

class TopLapNet(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
class MultitaskModule(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MultitaskModule, self).__init__()
     
        self.input_layer = nn.Linear(D_in, H[0], bias=False) 
        nn.init.xavier_uniform_(self.input_layer.weight)
        self.bn_input = nn.BatchNorm1d(H[0])

        self.hiden_layers = nn.ModuleList([
            nn.Linear(H[i], H[i+1], bias=False) for i in range(len(H)-1)
        ])
        for hiden_layer in self.hiden_layers:
            nn.init.xavier_uniform_(hiden_layer.weight)

        self.bn_hidden = nn.ModuleList([
            nn.BatchNorm1d(H[i+1]) for i in range(len(H)-1) 
        ])

        self.output_layer = nn.Linear(H[-1], D_out, bias=True)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, X):
        X = self.input_layer(X)
        # X = self.bn_input(X)
        X = F.relu(X)
        
        for i, hiden_layer in enumerate(self.hiden_layers):
            X = hiden_layer(X)
            X = self.bn_hidden[i](X)
            X = F.relu(X)
            
        y = self.output_layer(X)
        return y

def train(model, device, train_loader, criterion, optimizer, scheduler):
    model.train() #
    for (data, target) in train_loader:
        # move tensor to computing device ('gpu' or 'cpu')
        data, target = data.to(device), target.to(device).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data).view(-1, 1)
        loss = criterion(output, target)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

def test(model, device, test_loader, epoch):
    model.eval() 
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data).view(-1, 1)
            test_loss = F.mse_loss(output, target, reduction='sum').item()
            test_loss /= len(test_loader.dataset)
            pcc = PCC(output, target)[0]
            rmse = RMSE(output, target)
            print('Epoch: %d, test_loss: %.4f, RMSE: %.4f, PCC: %.4f'%(epoch, test_loss, rmse, pcc))
            return output.cpu().numpy(), target.cpu().numpy()

tic = time.perf_counter()

parser = argparse.ArgumentParser(description='CANet')
parser.add_argument('--dataset', type=str, default='S2648',
                    help='input batch size for training (default: 50)')
parser.add_argument('--datatype', type=str, default='all',
                    help='input batch size for training (default: 50)')
parser.add_argument('--batch_size', type=int, default=50,
                    help='input batch size for training (default: 50)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='SGD weight decay (default: 0)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--layers', type=str, default='2048,1024,1024,512,512,64',
                    help='neural network layers and neural numbers')
parser.add_argument('--nlayer', type=int, default=6,
                    help='number of neural network layers')
args = parser.parse_args()
print(args)
print('args.layers:',args.layers)
torch.manual_seed(args.seed)

# setup device cuda or cpu
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")
#device = torch.device("cpu")

# protein stability change upon mutation features and labels
if args.datatype == 'aux':
    X_val = np.load('./S2648/X_'+args.dataset+'_aux.npy')
elif args.datatype == 'FRI':
    X_val = np.load('./S2648/X_'+args.dataset+'_FRI.npy')
elif args.datatype == 'PH0':
    X_val = np.load('./S2648/X_'+args.dataset+'_PH0.npy')
elif args.datatype == 'PH12':
    X_val = np.load('./S2648/X_'+args.dataset+'_PH12.npy')
elif args.datatype == 'ESM':
    X_val = np.load('./S2648/X_'+args.dataset+'_ESM.npy')
elif args.datatype == 'Lap':
    X_val = np.load('./S2648/X_'+args.dataset+'_Lap_b.npy')
elif args.datatype == 'all':
    X_val1 = np.load('./S2648/X_'+args.dataset+'_aux.npy')
    X_val2 = np.load('./S2648/X_'+args.dataset+'_FRI.npy')
    X_val4 = np.load('./S2648/X_'+args.dataset+'_Lap_sheaf_charge01betti.npy')
    X_val5 = np.load('./S2648/X_'+args.dataset+'_ESM.npy')
    print("aux:", X_val1.shape)
    print("FRI:", X_val2.shape)
    print('ESM',X_val5.shape)

    
    #X_val6 = np.load('./S2648/X_'+args.dataset+'_Lap_b.npy')
    # X_val = np.concatenate((X_val1, X_val2), axis=1)
    # X_val = np.concatenate((X_val,  X_val3), axis=1)
    X_val = np.concatenate((X_val1,  X_val4), axis=1)
    X_val = np.concatenate((X_val,  X_val5), axis=1)
    # X_val = np.concatenate((X_val4, X_val5), axis=1)
    # X_val =np.concatenate((X_val1, X_val4), axis=1)
    #X_val = np.concatenate((X_val,  X_val6), axis=1)
    # X_val=np.concatenate((X_val1,  X_val5), axis=1)

X_val = normalize(X_val)[::2]
#normalizer1 = joblib.load('model/normalizer_mini_alphafold.pkl')
#X_val = normalizer1.transform(X_val_skempi2)
#normalizer2 = joblib.load('model/normalizer_Lap_ESM_mini_alphafold.pkl')
#X_val_Lap_ESM = normalizer2.transform(X_val_skempi2_Lap_ESM)

#X_val1 = np.concatenate((X_val[:, :759], X_val[:, 759+648:]), axis=1)
#X_val = np.concatenate((X_val1, X_val_Lap_ESM), axis=1)
y_val = np.load(f'./S2648/Y_{args.dataset}.npy').reshape((-1, 1))[::2]
print('The data shape', X_val.shape, ', label size', y_val.shape)

data = dataset_list(f'./S2648/S350.txt')
all_data = dataset_list(f'./S2648/S2648.txt')
train_idx = list(range(len(all_data)))
test_idx = []
for i in range(len(data)):
    ilist = data[i]
    PDBid, Antibody, Chain, resWT, resID, resMT, pH, ddG = ilist[0], ilist[1], ilist[2], ilist[3], ilist[4], ilist[5], ilist[6], float(ilist[7])
    flag = False
    for j in range(len(all_data)):
        ilist2 = all_data[j]
    
        PDBid2, Antibody2, Chain2, resWT2, resID2, resMT2, pH2, ddG2 = ilist2[0], ilist2[1], ilist2[2], ilist2[3], ilist2[4], ilist2[5], ilist2[6], float(ilist2[7])
        #print(ilist2)
        if PDBid2 == PDBid and Antibody == Antibody2 and Chain2 == Chain and resWT == resWT2 and resID == resID2 and resMT == resMT2 and pH == pH2:
            test_idx.append(j)
            flag = True 
            break 
    
    if flag == False:
        print(ilist)

train_idx = list(set(train_idx)-set(test_idx))
print(len(test_idx), len(train_idx))
X_train, y_train = X_val[train_idx], y_val[train_idx]
X_test, y_test = X_val[test_idx], y_val[test_idx]



hiden_layer = [int(i) for i in args.layers.split(',')]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These two lines force cuDNN to be deterministic (might slightly slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

kwargs = {'shuffle': True, 'num_workers': 5, 'pin_memory': True} if use_cuda else {'shuffle': True}

# 2. Initialize lists to store the results of the 10 repetitions
pcc_list = []
rmse_list = []
pred_list=[]
true_list=[]
for ii in range(10):
   
    current_seed = args.seed + ii 
    set_seed(current_seed)
    
    train_dataset = TopLapNet(X_train, y_train)
    test_dataset  = TopLapNet(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=len(test_idx), shuffle=False,  pin_memory=True)

    model = MultitaskModule(X_val.shape[1], hiden_layer, 1).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(model.parameters(),  lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    lr_adjust = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.3
    )
    
    for epoch in range(args.epochs):
        train(model, device, train_loader, criterion, optimizer, lr_adjust)
        if epoch % args.log_interval == 0:
            test(model, device, test_loader, epoch)
        # lr_adjust.step()
    
    print('epoch %d >>>>>>>>>>>>>>>>>>>>>>>>'%epoch)
    test(model, device, test_loader, epoch)

    model.to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    model.eval()
    with torch.no_grad():
        ypred = model(X_test_tensor)[:, 0].view(-1, 1).cpu().numpy().ravel()

    y_pred = np.reshape(ypred, len(ypred))
    y_real = np.reshape(y_test, len(y_test))
    pred_list.append(y_pred)
    true_list.append(y_real)

    # fp = open(f'./S2648/{args.dataset}_blind_new_{ii}_CAnet.txt', 'w+')
    # for i in range(len(y_real)):
    #     fp.write(f'{y_pred[i]} {y_real[i]}\n')
    # fp.close()
    
    pcc = stats.pearsonr(y_pred, y_real)[0]
    rmse = np.sqrt(mean_squared_error(y_pred, y_real))
 
    
   
    pcc_list.append(pcc)
    rmse_list.append(rmse)
    
    toc = time.perf_counter()
    print('Repetition %d | Seed: %d | RMSE: %.3f, Rp: %.4f\nElapsed time: %.1f [min]' % (ii, current_seed, rmse, pcc, (toc-tic)/60))

pred_list=np.array(pred_list)
true_list=np.array(true_list)

pred_mean=pred_list.mean(axis=0)
true_mean=true_list[0]

# fp = open(f'./S2648/{args.dataset}_blind_new_{ii}_CAnet.txt', 'w+')
# for i in range(len(true_mean)):
#     fp.write(f'{pred_mean[i]} {true_mean[i]}\n')
# fp.close()
fp = open(f'./S2648/{args.dataset}_blind_ensemble_CAnet.txt', 'w+')
for i in range(len(true_mean)):
    fp.write(f'{pred_mean[i]} {true_mean[i]}\n')
fp.close()

pcc_mean = stats.pearsonr(pred_mean, true_mean)[0]
rmse_mean = np.sqrt(mean_squared_error(pred_mean, true_mean))
print('ensemble:', pcc_mean, rmse_mean)
print('plot')

print("\n" + "="*40)
print(f"RESULTS OVER 10 REPETITIONS (Base Seed: {args.seed})")
print("="*40)
print(f"Mean RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
print(f"Mean PCC:  {np.mean(pcc_list):.4f} ± {np.std(pcc_list):.4f}")
print("="*40)
