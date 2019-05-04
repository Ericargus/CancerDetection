import torch
from torchvision.transforms import*
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import albumentations
from albumentations import torch as AT

from dataload import*
from imageTransform import*
from senet import*

import scipy

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

#path = r"E:\Master\HistopathologicCancer\data\cancer"
path = "/mnt/hdd0/zxy/cancer/"
train_labels = pd.read_csv(os.path.join(path, "train_labels.csv"))
train_df = train_labels['id']
target_df = train_labels['label']
ITER_MAX = 7
n_split = 10
splits = list(StratifiedKFold(n_splits=n_split, shuffle=True, random_state=4400).split(train_df, target_df))

#train_df, val_df = train_test_split(train_labels, test_size = 0.2, random_state = 123 )
#train_df.to_csv("train.csv", index = False)
#val_df.to_csv('val.csv', index = False)
class semodel(nn.Module):
    def __init__(self, ):
        super(semodel, self).__init__()
        self.basenet = se_resnext50_32x4d()
        self.linear = nn.Linear(1000 + 2, 16)
        self.bn = nn.BatchNorm1d(16)
        self.drop = nn.Dropout2d(p=0.2)
        self.elu = nn.ELU()
        self.out = nn.Linear(16, 1)
        self.layer0 = se_resnext50_32x4d().layer0
        self.layer1 = se_resnext50_32x4d().layer1
        self.layer2 = se_resnext50_32x4d().layer2

    def forward(self, x):
        out = self.basenet(x)
        batch = out.shape[0]
        max_pool, _ = torch.max(out, 1, keepdim = True)
        avg_pool = torch.mean(out, 1, keepdim = True)
        out = out.view(batch, -1)
        conc = torch.cat((out, max_pool, avg_pool), dim = 1)
        conc = self.linear(conc)
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.drop(conc)
        res = self.out(conc)

        return res

def parameters_to_update(m):
    m.layer0.requires_grad=False
    m.layer1.requires_grad=False
    m.layer2.requires_grad=False
    return m.parameters()
    

def main():
    """
    geometric_transform_prob = 0.5 * 0.33
    geometric_transform = Compose([RandomApply([ElasticDeformation(max_distort=0.15)], p = geometric_transform_prob),
                                RandomApply([rotationTransform(angle = 35)], p =geometric_transform_prob),
                                RandomApply([horizontalShear(max_scale = 0.07)], p = geometric_transform_prob)
                                ])
    color_transform_prob = 0.5 * 0.33 
    color_transform = Compose([RandomApply([brightnessShift(max_value = 0.1)], p = color_transform_prob),
                        RandomApply([brightnessScaling(max_value = 0.08)], p = color_transform_prob),
                        RandomApply([gammaChange(max_value = 0.08)], p = color_transform_prob),
                        ])

    train_transform = Compose([RandomApply([GaussNoise(sigma_sq=0.15)], p=0.5),
                            RandomApply([horizontalFlip()], p = 0.5),
                            RandomApply([saltPepper(probability=0.15)], p = 0.5),
                            geometric_transform,
                            color_transform,
                            ])
    """
    data_transforms = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5), 
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(hue=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3), 
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),]),
        transforms.RandomChoice([
            transforms.RandomRotation((0,0)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation((90,90)),
            transforms.RandomRotation((180,180)),
            transforms.RandomRotation((270,270)),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((90,90)),
                ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((270,270)),
                ])
            ]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    criterion = nn.BCEWithLogitsLoss()
    model = se_resnext50_32x4d()
    num_nfcs = model.last_linear.in_features
    model.last_linear = nn.Sequential(
         nn.Linear(num_nfcs, 1)
    )
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = nn.DataParallel(model)
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    for idx, (train_idx, val_idx) in enumerate(splits):
        #print(train_df[train_idx].values)
        #print(train_df[val_idx])
        train_dataset = CancerDataset(data = train_df[train_idx].values, target = target_df[train_idx].values, datapath = path, modetype = "train",transform = data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=768, shuffle=True, drop_last=True)

        val_dataset = CancerDataset( data = train_df[val_idx].values, target = target_df[val_idx].values , datapath = path, modetype = "val", transform = data_transforms)
        val_loader = DataLoader(val_dataset, batch_size=768, shuffle=False, drop_last=True)

        num_step_per_epoch = len(train_loader)
        best_val_acc = 0
        best_epoch = 0
        epoch_since_best = 0
        early_stop = 2
        model.to(device)
        for iteration in range(ITER_MAX):
            #lr_schedule.step()
            model.train()
            train_loss = []
            train_acc = []
            for i, data in enumerate(train_loader):
                image, label = data
                image = image.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                out = model(image)
                loss = criterion(out.squeeze(), label.float())
                #lr_schedule.step(loss)
                #print('loss: %6f'%(loss))
                loss.backward()
                optimizer.step()
                a = label.data.cpu().numpy()
                b = out.detach().cpu().numpy()
                train_acc.append(roc_auc_score(a, b))
                train_loss.append(loss.item())
            model.eval()
            val_acc = []
            val_loss = []
            valid_acc = 0
            for _, data in enumerate(val_loader):
                image, label = data
                image = image.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                out = model(image)
                loss = criterion(out.squeeze(),label.float())
                a = label.data.cpu().numpy()
                b = out.detach().cpu().numpy()
                val_acc.append(roc_auc_score(a, b))
                val_loss.append(loss.item())
            print('[Fold %d] [Epoch %d] train loss %.6f train acc %.6f valid loss %.6f val_acc %.6f'%(
                idx, iteration, np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc)
            ))
            valid_acc =np.mean(val_acc)
            lr_schedule.step(valid_acc)
            if(valid_acc > best_val_acc):
                best_val_acc = valid_acc
                best_epoch = iteration 
                epoch_since_best = 0
                print('save model ...')
                torch.save(model.state_dict(), 'best_model_{}.pth'.format(idx))
                print('model saved')
            else:
                epoch_since_best += 1
            if epoch_since_best > early_stop:
                break
            print('Finished Training')
            print('best_epoch: %d, best_val_acc %.6f' % (best_epoch, best_val_acc))

def test():
    Num_TTA = 8
    data_transforms = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(hue=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),]),
        transforms.RandomChoice([
            transforms.RandomRotation((0,0)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation((90,90)),
            transforms.RandomRotation((180,180)),
            transforms.RandomRotation((270,270)),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((90,90)),
                ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((270,270)),
                ])
            ]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    data_transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        )
    ])
    data_transforms_tta0 = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    data_transforms_tta1 = transforms.Compose([
        transforms.RandomRotation(270, 270),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    
    
    data_transforms_tta2 = transforms.Compose([
        transforms.RandomRotation(90, 90),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
   
    ])
    
    data_transforms_tta3 = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomRotation((0,0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation((180,180)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])
 
    #val_dataset = CancerDataset(dataFolder = path , modetype = "test" , transform = None)
    #val_loader = DataLoader(val_dataset, batch_size=768, shuffle=False)
    model = se_resnext50_32x4d()
    num_nfcs = model.last_linear.in_features
    model.last_linear = nn.Sequential(
    nn.Linear(num_nfcs, 1)
    )
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = nn.DataParallel(model.cuda())
    state_dict = torch.load("./best_model_0.pth")
    model.load_state_dict(state_dict)
    model.eval()
    sigmoid = lambda x: scipy.special.expit(x)
    for num_tta in range(Num_TTA):
        if num_tta == 0:
            val_dataset = CancerDataset(data = None, target = None, datapath = path, modetype = "test", transform = data_transforms_test)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        elif num_tta == 1:
            val_dataset = CancerDataset(data = None, target = None, datapath = path, modetype = "test", transform = data_transforms_tta0)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        elif num_tta == 2:
            val_dataset = CancerDataset(data = None, target =None, datapth = path, modetype = "test", transform = data_transforms_tta1)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        elif num_tta == 3:
            val_dataset = CancerDataset(data = None, target = None , datapath = path, modetype = "test", transform = data_transforms_tta2)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False) 
        else:
           val_dataset = CancerDataset(data= None, target = None, datapath = path, modetype = "test", transform = data_transforms_tta3)
           val_loader = DataLoader(val_dataset, batch_size =512, shuffle=False)
        preds = []
        print("Prediction start!")
        for i, batch in enumerate(val_loader):
            #print("prediction start")
            image, _ = batch
            image = image.to(device)
            out = model(image)
            pr = out.detach().cpu().numpy()
            for i in pr:
                preds.append(sigmoid(i)/Num_TTA)
        if num_tta == 0:
            test_preds = pd.DataFrame({'imgs': val_dataset.fileName['id'], 'preds': preds})
        else:
            test_preds['preds'] += np.array(preds)
    sub = pd.read_csv('path = "/mnt/hdd0/zxy/cancer/sample_submission.csv')
    print("prediction done!")
    sub = pd.merge(sub, test_preds, left_on = 'id', right_on = 'imgs')
    sub = sub[['id', 'preds']]
    sub.columns = ['id', 'label']
    print(sub.head())
    sub.to_csv("./submissionsTTA1.csv", index = False)






if __name__ == "__main__":
    test()

    
