import argparse, time, os
import numpy as np
import torch
from utils_data import prepare_cv_datasets, prepare_train_loaders
from utils_algo import accuracy_check, chosen_loss_c
from models import mlp_model, linear_model, LeNet
from cifar_models import densenet, resnet, convnet


parser = argparse.ArgumentParser()

parser.add_argument('-lr', help='optimizer\'s learning rate', default=1e-3, type=float)
parser.add_argument('-bs', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-ds', help='specify a dataset', default="mnist", type=str, required=False) # mnist, kmnist, fashion, cifar10
parser.add_argument('-me', help='method type', choices=['SCARCE'], type=str, required=True)
parser.add_argument('-mo', help='model name', default='mlp', choices=['linear', 'mlp', 'resnet', 'densenet', 'lenet','convnet'], type=str, required=False)
parser.add_argument('-ep', help='number of epochs', type=int, default=200)
parser.add_argument('-wd', help='weight decay', default=1e-5, type=float)
parser.add_argument('-seed', help = 'Random seed', default=0, type=int, required=False)
parser.add_argument('-gpu', help = 'used gpu id', default='0', type=str, required=False)
parser.add_argument('-op', help = 'optimizer', default='adam', type=str, required=False)
parser.add_argument('-gen', help = 'the generation process of complementary labels', default='random', choices=['random', 'set1', 'set2'], type=str, required=False)
parser.add_argument('-run_times', help='random run times', default=5, type=int, required=False)

args = parser.parse_args()
device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
acc_run_list = torch.zeros(args.run_times)
save_total_dir = "./result/total"
save_detail_dir = "./result/detail"

if not os.path.exists(save_total_dir):
    os.makedirs(save_total_dir)
if not os.path.exists(save_detail_dir):
    os.makedirs(save_detail_dir)
    
save_total_name = "Res_total_ds_{}_gen_{}_me_{}_mo_{}_op_{}_lr_{}_wd_{}_bs_{}_ep_{}_seed_{}.csv".format(args.ds, args.gen, args.me, args.mo, args.op, args.lr, args.wd, args.bs, args.ep, args.seed)
save_detail_name = "Res_detail_ds_{}_gen_{}_me_{}_mo_{}_op_{}_lr_{}_wd_{}_bs_{}_ep_{}_seed_{}.csv".format(args.ds, args.gen, args.me, args.mo, args.op, args.lr, args.wd, args.bs, args.ep, args.seed)
save_total_path = os.path.join(save_total_dir, save_total_name)
save_detail_path = os.path.join(save_detail_dir, save_detail_name)

if os.path.exists(save_total_path):
    os.remove(save_total_path)
if os.path.exists(save_detail_path):
    os.remove(save_detail_path)
    
if_write = True

if if_write:
    with open(save_total_path, 'a') as f:
        f.writelines("run_idx,acc,std\n")
    with open(save_detail_path, 'a') as f:
        f.writelines("epoch,train_loss,train_accuracy,test_accuracy\n")

for run_idx in range(args.run_times):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed); 
    torch.cuda.manual_seed_all(args.seed);
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.seed = args.seed + 1
    print('the {}-th random round'.format(run_idx))
    
    full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K = prepare_cv_datasets(dataname=args.ds, batch_size=args.bs)
    ordinary_train_loader, complementary_train_loader, ccp, dim = prepare_train_loaders(full_train_loader=full_train_loader, batch_size=args.bs, ordinary_train_dataset=ordinary_train_dataset, complementary_type=args.gen, seed=args.seed)

    if args.mo == 'mlp':
        model = mlp_model(input_dim=dim, hidden_dim=500, output_dim=K)
    elif args.mo == 'linear':
        model = linear_model(input_dim=dim, output_dim=K)
    elif args.mo == 'lenet':
        model = LeNet(output_dim=K) #  linear,mlp,lenet are for MNIST-type datasets.
    elif args.mo == 'densenet':
        model = densenet(num_classes=K)
    elif args.mo == 'resnet':
        model = resnet(depth=32, num_classes=K)
    elif args.mo == 'convnet':
        model = convnet.Cnn(input_channels=3,n_outputs=K,dropout_rate=0.25)  # densenet,resnet are for CIFAR-10.
    meta_method = args.me
    model = model.to(device)
    if args.op == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.wd,momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
    test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
    print('Epoch: 0. Tr Acc: {}. Te Acc: {}'.format(train_accuracy, test_accuracy))

    test_acc_list = []
    train_acc_list = []
    for epoch in range(args.ep):
        model.train()
        for i, (images, labels) in enumerate(complementary_train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_vector = chosen_loss_c(f=outputs, K=K, labels=labels, ccp=ccp, meta_method=meta_method, device=device)
            loss.backward()
            optimizer.step()
        model.eval()
        train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
        test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
        if if_write:
            with open(save_detail_path, "a") as f:
                f.writelines("{},{:.6f},{:.6f},{:.6f}\n".format(epoch + 1, loss.data.item(), train_accuracy, test_accuracy))
        if epoch >= (args.ep-10):
            test_acc_list.extend([test_accuracy])
            train_acc_list.extend([train_accuracy])
        print('Epoch: {}. Tr Acc: {}. Te Acc: {}.'.format(epoch+1, train_accuracy, test_accuracy))

    avg_test_acc = np.mean(test_acc_list)
    avg_train_acc = np.mean(train_acc_list)
    acc_run_list[run_idx] = avg_test_acc
    print('\n')
    if if_write:
        with open(save_total_path, "a") as f:
            f.writelines("{},{:.6f},None\n".format(run_idx + 1, avg_test_acc))  
    print("Average Test Accuracy over Last 10 Epochs:", avg_test_acc)
    print("Average Training Accuracy over Last 10 Epochs:", avg_train_acc,"\n\n\n")
    
print('Avg_acc:{}    std_acc:{}'.format(acc_run_list.mean(), acc_run_list.std()))
if if_write:
    with open(save_total_path, "a") as f:
        f.writelines("in total,{:.6f},{:.6f}\n".format(acc_run_list.mean(), acc_run_list.std()))    
print("NOW is dataset: {} with method {} with model {} weight_decay {} learning rate {} batch_size {} op {}".format(args.ds,args.me,args.mo,args.wd,args.lr,args.bs,args.op))
