import os
from util import get_num_lines, get_vocab, embed_sequence, get_word2idx_idx2word, get_embedding_matrix, get_data, saveSenResult, get_betch
from util import evaluate,get_w2v_attention
from model import RNNSequenceClassifier
from elmoformanylangs import Embedder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import csv
import h5py
import jieba
import argparse
import sys

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
print(torch.cuda.is_available())
using_GPU = True

"""
1. Data pre-processing
"""
'''Q
1.1 VUA
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a index: int: idx of the focus verb
    a label: int 1 or 0
'''
parser = argparse.ArgumentParser()
parser.add_argument("--w2v_type_sign")
parser.add_argument("--save_path_sign")
args = parser.parse_args()
print('w2v_attention_type:  ',sys.argv[2],'\tsave_path:  ',sys.argv[4])

data_path = 'F:\\LiaoSF\\second_data_experiment\\分类数据\\'        #读取数据地址
save_path = 'result_all\\'+sys.argv[2]+'_'+sys.argv[4]+'\\'              #保存模型地址
embedding_sign = 'tence'           #None：不用词向量；  'tence'：使用tence给出的词向量；  'hit'：使用哈工大预训练的elmo；  'allennlp'：使用allenlp给出的elmo
weight_sign = None                  #True：使用倒数权重；  None：不适用权重
num_layers=1                        #网络层数
num_classes = 3                     #类别数
bidir=True                          #单双向
hidden_size = 200                   #隐藏层神经元数
lr=0.01                             #学习率
momentum=0.9                        #动量
num_epochs = 60                     #训练轮次
batch_size = 64                     #批次大小
dropout1=0.3                        #dropout on input to RNN
dropout2=0.2                        #dropout in RNN; would be used if num_layers=1
dropout3=0.2                        #dropout on hidden state of RNN to linear layer
require_improvement = 1000          #多轮未提高提前退出训练
print_per_batch = 200               #每多少轮输出一次结果
embedding_input_type = True              #input词向量训练不训练
embedding_attention_type = True              #attention词向量训练不训练
dim = {None:200, 'tence':200, 'hit':1224, 'allennlp':1224}            #embedding维度
id = {'COAE2015':0, 'EmotionClassficationTest':1, 'ExpressionTest':2, 'NLPCC17':3, 'NLPCC2014':4, '奥运会':5, '春晚':6, '乐视':7, '旅游':8, '微博':9, '雾':10, '端午节':11, '国考':12, '南海':13, '贝克汉姆':14, '闯红灯':15, 'guan':16, 'san':17, '奥巴马':18, '百度':19, '两会':20, 'hui':21, '围棋':22, '比特':23, '韩国':24, '共享':25, 'pin':26, 'han':27, '特朗普':28, 'jiu':29, 'xue':30, '埃':31, '火车票':32, 'shi':33, 'peng':34, 'ipad':35, 'liu':36, 'ming':37, 'fei':38}
#倒数权重
W1 = [100.8695652173913, 13.711583924349881, 2.0647917408330367, 30.606860158311346, 17.873651771956858, 43.28358208955224, 3.389830508474576, 46.963562753036435, 207.14285714285714]
W = W1                              #使用权重

raw_train_vua = get_data(data_path+'train.csv', id)
raw_test_vua = get_data(data_path+'test.csv', id)
raw_val_vua = get_data(data_path+'val.csv', id)
writer_process = open(save_path + 'process.txt', mode='w')
print('VUA dataset division: ', len(raw_train_vua), len(raw_val_vua), len(raw_test_vua))
"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_train_vua + raw_val_vua + raw_test_vua)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, embedding_sign,embedding_input_type, normalization=False)
#词向量注意力
w2v = get_w2v_attention(data_path, sys.argv[2])
# elmo_embeddings
elmos_allennlp = h5py.File(data_path+'data.hdf5', 'r')
elmos_hit = Embedder('C:\\Users\\TitanXp-4\\Downloads\\179')
"""
3. Model training
"""
'''
3. 1 
set up model, loss criterion, optimizer
'''
# Instantiate the model
# embedding_dim = glove + elmo
rnn_clf = RNNSequenceClassifier(num_classes=num_classes, embedding_dim=dim[embedding_sign],w2v = w2v,embedding_attention_type = embedding_attention_type, hidden_size=hidden_size, num_layers=num_layers, bidir=bidir,dropout1=dropout1, dropout2=dropout2, dropout3=dropout3)
# Move the model to the GPU if available
if using_GPU:
    rnn_clf = rnn_clf.cuda()
# Set up criterion for calculating loss
if weight_sign == None:
    nll_criterion = nn.NLLLoss()
elif weight_sign == True:
    nll_criterion = nn.NLLLoss(weight = torch.cuda.FloatTensor([i for i in W]))
# Set up an optimizer for updating the parameters of the rnn_clf
attention_p, net_p = [], []
for name, p in rnn_clf.named_parameters():
    print(name,'\t',p.shape)
    if 'attention' in name:
        attention_p += [p]
    else:
        net_p += [p]
rnn_clf_optimizer_net = optim.SGD([{'params': net_p, 'weight_decay': 0}], lr=lr,momentum=momentum)
rnn_clf_optimizer_att = optim.SGD([{'params': attention_p, 'weight_decay': 0}], lr=lr,momentum=momentum)
'''
3. 2
train model
'''
# A counter for the number of gradient updates
num_iter = 0
best_micro = 0
best_macro = 0
last_micro = 0
last_macro = 0
for epoch in range(num_epochs):
    print("Starting epoch {}".format(epoch + 1))
    train_dataloader_vua = get_betch(raw_train_vua, word2idx, glove_embeddings, elmos_allennlp, elmos_hit,
                                     embedding_sign, batch_size=batch_size, shuffle=True)
    for (example_text, example_lengths, labels) in train_dataloader_vua:
        example_text = Variable(example_text)
        example_lengths = Variable(example_lengths)
        labels = Variable(labels)
        if using_GPU:
            example_text = example_text.cuda()
            example_lengths = example_lengths.cuda()
            labels = labels.cuda()
        predicted, attention_loss,_ = rnn_clf(example_text, example_lengths)
        batch_loss = nll_criterion(predicted, labels)
        num_iter += 1

        rnn_clf_optimizer_att.zero_grad()
        attention_loss.backward()
        rnn_clf_optimizer_att.step()

        rnn_clf_optimizer_net.zero_grad()
        batch_loss.backward()
        rnn_clf_optimizer_net.step()

        # Calculate validation and training set loss and accuracy every 200 gradient updates
        if num_iter % print_per_batch == 0:
            val_dataloader_vua = get_betch(raw_val_vua, word2idx, glove_embeddings, elmos_allennlp, elmos_hit,
                                           embedding_sign, batch_size=1, shuffle=True)
            avg_eval_loss,attention_loss, eval_accuracy, f_macro, f_micro = evaluate(val_dataloader_vua, rnn_clf,nll_criterion, using_GPU, 'train')
            print("Iteration {}. Validation Predict Loss {}. Validation Attention Loss {}. Validation Accuracy {}. Validation f_macro {}. Validation f_micro {}.".format(
                    num_iter, avg_eval_loss,attention_loss, eval_accuracy, f_macro, f_micro))
            writer_process.write("Iteration {}. Validation Predict Loss {}. Validation Attention Loss {}. Validation Accuracy {}. Validation f_macro {}. Validation f_micro {}.\n".format(
                    num_iter, avg_eval_loss,attention_loss, eval_accuracy, f_macro, f_micro))
            if best_macro <= f_macro:
                best_macro = f_macro
                last_macro = num_iter
                torch.save(rnn_clf, save_path+'model_macro.pkl')
            if best_micro <= f_micro:
                best_micro = f_micro
                last_micro = num_iter
                torch.save(rnn_clf, save_path+'model_micro.pkl')
    if num_iter - last_macro > require_improvement and num_iter - last_micro > require_improvement:
        print("No optimization for a long time, auto-stopping...")
        writer_process.write("No optimization for a long time, auto-stopping...\n")
        break
print("Training done!")
writer_process.write("Training done!\n")
"""
4. test the model
"""
#测试micro模型
rnn_clf = torch.load(save_path+'model_micro.pkl')
test_dataloader_vua = get_betch(raw_test_vua, word2idx, glove_embeddings, elmos_allennlp, elmos_hit, embedding_sign, batch_size=1, shuffle=False)
avg_eval_loss, eval_accuracy, p_micro, r_micro, f_micro, y_test_cls, y_pred_cls, confusion_matrix,weights,details_result = evaluate(test_dataloader_vua, rnn_clf, nll_criterion, using_GPU, 'micro_test')
print(details_result)
writer_process.write(details_result)
print("Micro: Test Accuracy {}. Test Precision {}. Test Recall {}. Test F_micro {}.".format(eval_accuracy, p_micro, r_micro, f_micro))
writer_process.write("Micro: Test Accuracy {}. Test Precision {}. Test Recall {}. Test F_micro {}.\nConfusion_matrix:\n{}.\n".format(eval_accuracy, p_micro, r_micro, f_micro,str(confusion_matrix)))
saveSenResult(raw_test_vua, y_test_cls, y_pred_cls, save_path,weights,'micro')
#测试macro模型
rnn_clf = torch.load(save_path+'model_macro.pkl')
avg_eval_loss, eval_accuracy, p_macro, r_macro, f_macro, y_test_cls, y_pred_cls, confusion_matrix,weights,details_result = evaluate(test_dataloader_vua, rnn_clf,nll_criterion, using_GPU, 'macro_test')
print(details_result)
writer_process.write(details_result)
print("Macro: Test Accuracy {}. Test Precision {}. Test Recall {}. Test F_macro {}.".format(eval_accuracy, p_macro, r_macro, f_macro))
writer_process.write("Macro: Test Accuracy {}. Test Precision {}. Test Recall {}. Test F_micro {}.\nConfusion_matrix:\n{}.\n".format(eval_accuracy, p_macro, r_macro, f_macro,str(confusion_matrix)))
saveSenResult(raw_test_vua, y_test_cls, y_pred_cls, save_path,weights,'macro')