import math
import matplotlib.pyplot as plt
import re
import tqdm
import time
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM





drive_dir = "./process_data/"
SEQ_LEN = 512

def preprocess(mode):
    x, y = [], []
    print(mode)
    with open(drive_dir + mode, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        length = []
        count = 0
        for line in lines:
            line_info = line.split(",")
            text = line_info[0]
            text = re.sub(" ", "", text)
            length.append(len(text))      
            count += 1
            x.append(text)
            y.append(int(float(line_info[1])))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    input_ids, attention_masks, input_types = [], [], []
    for text in x:
        input = tokenizer(text, max_length=SEQ_LEN, padding="max_length", truncation=True)
        input_ids.append(input["input_ids"])
        attention_masks.append(input["attention_mask"])
        input_types.append(input["token_type_ids"])

    input_ids, attention_masks, input_types = np.array(input_ids), np.array(attention_masks), np.array(input_types)
    y = np.array(y)
    print("shape: ", input_ids.shape, attention_masks.shape, input_types.shape, y.shape)
    return input_ids, attention_masks, input_types, y

input_ids_train, attention_masks_train, input_types_train, y_train = preprocess("train.csv")
input_ids_val, attention_masks_val, input_types_val, y_val = preprocess("valid.csv")
input_ids_test, attention_masks_test, input_types_test, y_test = preprocess("test.csv")

BATCH_SIZE = 20

train_data = TensorDataset(torch.LongTensor(input_ids_train), torch.LongTensor(attention_masks_train), torch.LongTensor(input_types_train), torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(torch.LongTensor(input_ids_val), torch.LongTensor(attention_masks_val), torch.LongTensor(input_types_val), torch.LongTensor(y_val))
valid_sampler = SequentialSampler(valid_data)
valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test), torch.LongTensor(attention_masks_test), torch.LongTensor(input_types_test), torch.LongTensor(y_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))
print(len(test_data), len(test_loader))

class Bert(nn.Module):
    def __init__(self, classes=34, dropout=0.5):
        super(Bert, self).__init__()
        self.bert =AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
        self.config = self.bert.config
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(21128, classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        output = outputs.logits[:, 0, :]
        # output = self.dropout(output)    # modified
        # print(output)
        output = self.fc(output)
        # output = nn.ReLU(output)
        output = F.log_softmax(output, dim=1)
        return output




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 8
model = Bert().to(device)
# drive_dir_model = "./process_data/"
# pretrained_model_path=(drive_dir + "best_bert_model_preprocess_seqlen____new____" + str(SEQ_LEN) + ".pth")
#
# if os.path.isfile(pretrained_model_path):
#     model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
#     print(f"已从 {pretrained_model_path} 加载预训练权重")
# else:
#     print(f"在 {pretrained_model_path} 未找到预训练模型，将从头开始初始化。")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=epochs * len(train_loader))

def cal_test(label, pred):
    acc = np.mean(label == pred)
    score1 = np.abs(np.log(pred + 1) - np.log(label + 1))
    for i in range(len(score1)):
        if score1[i] <= 0.2:
            score1[i] = 1
        elif score1[i] > 0.2 and score1[i] <= 0.4:
            score1[i] = 0.8
        elif score1[i] > 0.4 and score1[i] <= 0.6:
            score1[i] = 0.6
        elif score1[i] > 0.6 and score1[i] <= 0.8:
            score1[i] = 0.4
        elif score1[i] > 0.8 and score1[i] <= 1:
            score1[i] = 0.2
        else:
            score1[i] = 0

    mae = np.mean(np.abs(label - pred))
    rmse = math.sqrt(np.mean(np.square(label - pred)))
    final_score = acc * 0.3 + np.mean(score1) * 0.7
    print("Acc:", acc, "score1: ", np.mean(score1), "FinalScore: ", final_score, "MAE:", mae, "RMSE:", rmse)
    return final_score

def evaluate(model, data_loader, device):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, att, type, y in data_loader:
            y_pred = model(idx.to(device), att.to(device), type.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())
    final_score = cal_test(np.array(val_true), np.array(val_pred))
    return final_score, accuracy_score(val_true, val_pred)

def predict(model, data_loader, device):    # 测试集
    model.eval()
    test_pred = []
    with torch.no_grad():
        for idx, att, type in tqdm(data_loader):
            y_pred = model(idx.to(device), att.to(device), type.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            test_pred.extend(y_pred)
    return test_pred


def log_absolute_error(pred, label):
    abs_error = torch.abs(torch.log(pred + 1) - torch.log(label + 1))

    # 返回总误差的和作为标量
    return abs_error.sum()

def train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, device, epoch):
    best_score = 0.0
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss().to(device)


    for i in range(epoch):
        torch.cuda.empty_cache()
        start = time.time()
        model.train()
        print("Epoch {}".format(i + 1))
        train_loss = 0.0
        for idx, (ids, att, tpe, y) in enumerate(train_loader):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            y_pred = model(ids, att, tpe)
            y_pred = y_pred.squeeze()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            if (idx + 1) % (len(train_loader) // 50) == 0:
                print("Epoch {:02d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}s".format(i + 1, idx + 1, len(train_loader), train_loss / (idx + 1), time.time() - start))

        model.eval()
        final_score, exact_acc = evaluate(model, valid_loader, device)
        if exact_acc > best_acc:
            best_acc = exact_acc
        if final_score > best_score:
            best_score = final_score
            torch.save(model.state_dict(), drive_dir + "best_bert_model_preprocess_seqlen____regression____" + str(SEQ_LEN) + ".pth")

        print("current final_score is {:.6f}, best final_score is {:.6f}".format(final_score, best_score))
        print("current acc is {:.4f}, best acc is {:.4f}".format(exact_acc, best_acc))


train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, device, epochs)

model.load_state_dict(torch.load(drive_dir + "best_bert_model_preprocess_seqlen____regression____" + str(SEQ_LEN) + ".pth"))

test_score, test_acc = evaluate(model, test_loader, device)
print("test acc is {:.4f}, test final_score is {:.6f}".format(test_acc, test_score))

