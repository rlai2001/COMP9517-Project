#
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms, models, datasets
# from sklearn.metrics import f1_score, precision_score, recall_score
# from tqdm import tqdm
# import os
# import pandas as pd
# import shutil
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# def evaluate(model, loader, device, criterion):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             _, predicted = torch.max(outputs, 1)
#
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             total_loss += loss.item()
#
#             all_preds.extend(predicted.cpu().tolist())
#             all_labels.extend(labels.cpu().tolist())
#
#     acc = 100 * correct / total
#     avg_loss = total_loss / len(loader)
#     f1 = f1_score(all_labels, all_preds, average='macro')
#     precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
#     recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#     return acc, avg_loss, f1, precision, recall, all_labels, all_preds
#
# def main():
#     data_dir = './Aerial_Landscapes'
#     batch_size = 32
#     num_epochs = 20
#     learning_rate = 0.001
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
#     class_names = full_dataset.classes
#
#     train_size = int(0.8 * len(full_dataset))
#     test_size = len(full_dataset) - train_size
#     train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#
#     model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
#     model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
#     model.to(device)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#
#     history = []
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         all_preds = []
#         all_labels = []
#
#         with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
#             for inputs, labels in pbar:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 running_loss += loss.item()
#
#                 all_preds.extend(predicted.cpu().tolist())
#                 all_labels.extend(labels.cpu().tolist())
#
#                 pbar.set_postfix(loss=running_loss / len(train_loader), acc=100 * correct / total)
#
#         train_acc = 100 * correct / total
#         train_loss = running_loss / len(train_loader)
#         train_f1 = f1_score(all_labels, all_preds, average='macro')
#         train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
#         train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#
#         test_acc, test_loss, test_f1, test_precision, test_recall, y_true, y_pred = evaluate(model, test_loader, device, criterion)
#
#         print(f"\n Epoch {epoch+1}/{num_epochs}")
#         print(f"Train - Acc: {train_acc:.2f}%, Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
#         print(f"Test  - Acc: {test_acc:.2f}%, Loss: {test_loss:.4f}, F1: {test_f1:.4f}")
#
#         history.append({
#             'epoch': epoch+1,
#             'train_acc': train_acc,
#             'train_loss': train_loss,
#             'test_acc': test_acc,
#             'test_loss': test_loss,
#             'test_f1': test_f1
#         })
#
#         torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
#         torch.save(torch.tensor(y_true), 'y_true.pt')
#         torch.save(torch.tensor(y_pred), 'y_pred.pt')
#
#     # 分析最佳轮数 & 过拟合起始
#     df = pd.DataFrame(history)
#     best_row = df.loc[df['test_f1'].idxmax()]
#     best_epoch = int(best_row['epoch'])
#
#     # 检测过拟合点（train acc 上升且 test acc 开始下降）
#     overfit_epoch = None
#     for i in range(1, len(df)):
#         if df.loc[i, 'train_acc'] > df.loc[i-1, 'train_acc'] and df.loc[i, 'test_acc'] < df.loc[i-1, 'test_acc']:
#             overfit_epoch = df.loc[i, 'epoch']
#             break
#
#     print(f"\n 训练完成，共 {num_epochs} 轮")
#     print(f"最佳测试 F1 分数: {best_row['test_f1']:.4f}, Epoch {best_epoch}")
#     if overfit_epoch:
#         print(f"过拟合可能开始于第 {overfit_epoch} 轮（train acc ↑, test acc ↓）")
#     else:
#         print("未检测到明显过拟合点")
#
#     # 保存最佳模型
#     shutil.copyfile(f'model_epoch_{best_epoch}.pth', 'best_model.pth')
#     print(f"最佳模型权重已保存为 best_model.pth（来自第 {best_epoch} 轮）")
#
#     # 删除非最佳模型，只保留 best_model.pth
#     for i in range(1, num_epochs + 1):
#         path = f"model_epoch_{i}.pth"
#         if i != best_epoch and os.path.exists(path):
#             os.remove(path)
#     print("非最佳模型已删除，仅保留 best_model.pth")
#
#     # 保存训练指标
#     df.to_csv("metrics.csv", index=False)
#     print("训练指标已保存为 metrics.csv")
#
# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.freeze_support()
#     main()


# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms, models, datasets
# from sklearn.metrics import f1_score, precision_score, recall_score
# from tqdm import tqdm
# import os
# import pandas as pd
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# def evaluate(model, loader, device, criterion):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             _, predicted = torch.max(outputs, 1)
#
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             total_loss += loss.item()
#
#             all_preds.extend(predicted.cpu().tolist())
#             all_labels.extend(labels.cpu().tolist())
#
#     acc = 100 * correct / total
#     avg_loss = total_loss / len(loader)
#     f1 = f1_score(all_labels, all_preds, average='macro')
#     precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
#     recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#     return acc, avg_loss, f1, precision, recall, all_labels, all_preds
#
# def main():
#     data_dir = './Aerial_Landscapes'
#     batch_size = 16  # ✅ 更改后的 batch size
#     num_epochs = 20
#     learning_rate = 0.0005  # ✅ 更改后的学习率
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
#     class_names = full_dataset.classes
#
#     train_size = int(0.8 * len(full_dataset))
#     test_size = len(full_dataset) - train_size
#     train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#
#     model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
#     model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
#     model.to(device)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#
#     history = []
#     best_f1 = 0.0
#     best_epoch = 0
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         all_preds = []
#         all_labels = []
#
#         with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
#             for inputs, labels in pbar:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 running_loss += loss.item()
#
#                 all_preds.extend(predicted.cpu().tolist())
#                 all_labels.extend(labels.cpu().tolist())
#
#                 pbar.set_postfix(loss=running_loss / len(train_loader), acc=100 * correct / total)
#
#         train_acc = 100 * correct / total
#         train_loss = running_loss / len(train_loader)
#         train_f1 = f1_score(all_labels, all_preds, average='macro')
#         train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
#         train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#
#         test_acc, test_loss, test_f1, test_precision, test_recall, y_true, y_pred = evaluate(model, test_loader, device, criterion)
#
#         print(f"\n Epoch {epoch+1}/{num_epochs}")
#         print(f"Train - Acc: {train_acc:.2f}%, Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
#         print(f"Test  - Acc: {test_acc:.2f}%, Loss: {test_loss:.4f}, F1: {test_f1:.4f}")
#
#         history.append({
#             'epoch': epoch+1,
#             'train_acc': train_acc,
#             'train_loss': train_loss,
#             'test_acc': test_acc,
#             'test_loss': test_loss,
#             'test_f1': test_f1
#         })
#
#         if test_f1 > best_f1:
#             best_f1 = test_f1
#             best_epoch = epoch + 1
#             torch.save(model.state_dict(), 'temp_best_model.pth')
#             torch.save(torch.tensor(y_true), 'y_true.pt')
#             torch.save(torch.tensor(y_pred), 'y_pred.pt')
#
#     os.rename('temp_best_model.pth', 'best_model.pth')
#
#     df = pd.DataFrame(history)
#     df.to_csv("metrics.csv", index=False)
#
#     print(f"\n训练完成，共 {num_epochs} 轮")
#     print(f"最佳测试 F1 分数: {best_f1:.4f}, Epoch {best_epoch}")
#
#
# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.freeze_support()
#     main()

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import os
import pandas as pd

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = 100 * correct / total
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, avg_loss, f1, precision, recall, all_labels, all_preds

def main():
    data_dir = './Aerial_Landscapes'
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    history = []
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                pbar.set_postfix(loss=running_loss / len(train_loader), acc=100 * correct / total)

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        test_acc, test_loss, test_f1, test_precision, test_recall, y_true, y_pred = evaluate(model, test_loader, device, criterion)

        print(f"\n Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Acc: {train_acc:.2f}%, Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
        print(f"Test  - Acc: {test_acc:.2f}%, Loss: {test_loss:.4f}, F1: {test_f1:.4f}")

        history.append({
            'epoch': epoch+1,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall
        })

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'temp_best_model.pth')
            torch.save(torch.tensor(y_true), 'y_true.pt')
            torch.save(torch.tensor(y_pred), 'y_pred.pt')

    os.rename('temp_best_model.pth', 'best_model.pth')

    df = pd.DataFrame(history)
    df.to_csv("metrics.csv", index=False)

    print(f"\n训练完成，共 {num_epochs} 轮")
    print(f"最佳测试 F1 分数: {best_f1:.4f}, Epoch {best_epoch}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
