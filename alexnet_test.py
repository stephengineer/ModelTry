import numpy as np
import torch
import torchvision
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from alexnet import AlexNet
import ssl
 
# 全局取消证书验证，否则在下载数据集时可能会出现证书验证问题报错
ssl._create_default_https_context = ssl._create_unverified_context
 
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 查看cuda是否能用
print(device)
 
def train(args):
    # 超参数设置，方便管理
    num_epochs = args.max_epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    image_size = args.image_size
    momentum = args.momentum
    # 设置数据集的格式
    transform = transforms.Compose([transforms.Resize((image_size, image_size),
                                    interpolation=InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
                                    ])
    # 数据加载
    # 如果没有这个数据集的话会自动下载
    train_data = torchvision.datasets.CIFAR10(root="dataset",download=True,transform=transform,train=True)
    test_data = torchvision.datasets.CIFAR10(root="dataset",download=True,transform=transform,train=False)
 
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    print('Dataload is Ready')
    # 添加tensorboard路径
    writer = SummaryWriter(log_dir=args.SummerWriter_log)
    # 模型加载
    model = AlexNet(num_classes=10).to(device)
    # 参数量估计
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    # Loss and optimizer
    # 选择交叉熵作为损失函数
    criterion = nn.CrossEntropyLoss()
    # 选择SGD为优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    total_train_step = 0#记录训练次数
    total_test_step=0#记录测试次数
 
 
    # 开始训练
    for epoch in range(num_epochs):
        print("---------------第{}轮训练开始-------------".format(epoch + 1))
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
 
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
 
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            total_train_step = total_train_step + 1
 
            if (i + 1) % args.print_map_epoch == 0:# 100次显示一次loss
                print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
                          .format(epoch + 1, num_epochs, total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
 
        # Test the model
        model.eval()
        total_test_loss = 0
        total_accuary = 0  # 正确率
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_test_loss += loss
                total_accuary += correct
            print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
            writer.add_scalar("test_loss",total_test_loss,total_test_step)
            writer.add_scalar("test_accuary", correct / total, total_test_step)
            total_test_step += 1
    # Save the model checkpoint
    torch.save(model, 'weights.pth')
 
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    '''------------------------------------------调  节  部  分------------------------------------------------------'''
    parser.add_argument('--max_epoch', type=int, default=40, help='total epoch')
    parser.add_argument('--device_num', type=str, default='cpu', help='select GPU or cpu')
    parser.add_argument('--image_size', type=int, default=224, help='if crop img, img will be resized to the size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size, recommended 16')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.90, help='choice a float in 0-1')
    parser.add_argument('--print_map_epoch', type=int, default=100, help='')
    parser.add_argument('--SummerWriter_log', type=str, default='Alexnet', help='')
 
 
    args = parser.parse_args()
 
    train(args)
 
 
 
 
 
 