from tqdm import tqdm
import torch
import numpy as np
import mmap
import json
import csv
import jieba
from genhtml import GenHtml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from sklearn.metrics import classification_report
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder

# Misc helper functions
# Get the number of lines from a filepath
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def get_betch(raw_vua, word2idx, random_embeddings, elmos_allennlp, elmos_hit, embedding_sign, batch_size=1, shuffle=False):
    embedded_vua = [[embed_sequence(example[0], word2idx, random_embeddings, elmos_allennlp, elmos_hit, embedding_sign), example[2]]
        for example in raw_vua]
    dataset_vua = TextDatasetWithGloveElmoSuffix([example[0] for example in embedded_vua],
                                 [example[1] for example in embedded_vua])
    dataloader_vua = DataLoader(dataset=dataset_vua, batch_size=batch_size, shuffle=shuffle,
                                      collate_fn=TextDatasetWithGloveElmoSuffix.collate_fn)
    return dataloader_vua

def get_data(path, id):
    raw_vua = []
    with open(path, encoding='ansi') as f:
        lines = csv.reader(f)
        for line in lines:
            line[0] = id[jieba._lcut(line[0])[0]]
            line[1] = jieba._lcut(line[1].replace(' ',''))
            line[1] = " ".join(str(i) for i in line[1]).replace('\ue41d', '')
            raw_vua.append([line[1], int(line[0]), int(line[2])])
    return raw_vua

def get_embedding_matrix(word2idx, idx2word, embedding_sign,embedding_input_type, normalization=False):
    """
    assume padding index is 0

    :param word2idx: a dictionary: string --> int, includes <PAD> and <UNK>
    :param idx2word: a dictionary: int --> string, includes <PAD> and <UNK>
    :param normalization:
    :return: an embedding matrix: a nn.Embeddings
    """
    # Load the GloVe vectors into a dictionary, keeping only words in vocab
    embedding_dim = 200
    glove_path = "F:\\WeiJiYao\\metaphor-in-context-master\\Tencent_AILab_ChineseEmbedding.txt"
    glove_vectors = {}
    with open(glove_path, encoding='utf-8') as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(glove_path)):
            split_line = line.rstrip().split()
            word = split_line[0]
            if len(split_line) != (embedding_dim + 1) or word not in word2idx:
                continue
            assert (len(split_line) == embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == embedding_dim
            glove_vectors[word] = vector
            if embedding_sign == None:
                break
    print("Number of pre-trained word vectors loaded: ", len(glove_vectors))
    # Calculate mean and stdev of embeddings
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("Embeddings mean: ", embeddings_mean)
    print("Embeddings stdev: ", embeddings_stdev)
    # Randomly initialize an embedding matrix of (vocab_size, embedding_dim) shape
    # with a similar distribution as the pretrained embeddings for words in vocab.
    vocab_size = len(word2idx)
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    # Go through the embedding matrix and replace the random vector with a
    # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in glove_vectors:
            embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
    if normalization:
        for i in range(vocab_size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    embeddings.weight = nn.Parameter(embedding_matrix)
    return embeddings

def get_vocab(raw_dataset):
    """
    return vocab set, and prints out the vocab size
    :param raw_dataset: a list of lists: each inner list is a triple:
                a sentence: string
                a index: int: idx of the focus verb
                a label: int 1 or 0
    :return: a set: the vocabulary in the raw_dataset
    """
    vocab = []
    for example in raw_dataset:
        vocab.extend(example[0].split())
    vocab = set(vocab)
    print("vocab size: ", len(vocab))
    return vocab

def get_w2v_attention(data_path, w2v_type):
    if w2v_type == 'emo_vector':
        with open(data_path+'emo_vector.json', encoding='utf-8') as f:
            emo_vector = json.load(f)
    elif w2v_type == 'emo_vector_mean':
        with open(data_path+'emo_vector_mean.json', encoding='utf-8') as f:
            emo_vector = json.load(f)
    w2v = torch.FloatTensor(3, 200)
    w2v[0] = torch.Tensor(emo_vector['0'])
    w2v[1] = torch.Tensor(emo_vector['1'])
    w2v[2] = torch.Tensor(emo_vector['2'])
    return w2v

def get_word2idx_idx2word(vocab):
    """
    :param vocab: a set of strings: vocabulary
    :return: word2idx: string to an int
             idx2word: int to a string
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def embed_sequence(sequence, word2idx, random_embeddings, elmo_allennlp, elmo_hit, elmo_sign):
    words = sequence.split()
    # 1. embed the sequence by glove vector
    # Replace words with tokens, and 1 (UNK index) if words not indexed.
    indexed_sequence = [word2idx.get(x, 1) for x in words]
    # glove_part has shape: (seq_len, glove_dim)
    glove_part = random_embeddings(Variable(torch.LongTensor(indexed_sequence)))
    assert (glove_part.shape == (len(words), 200))
    # 2. embed the sequence by elmo vectors
    if elmo_sign == None or elmo_sign == 'tence':
        return glove_part.data
    elif elmo_sign == 'allennlp':
        elmo_part = elmo_allennlp[json.loads(list(elmo_allennlp['sentence_to_index'])[0])[sequence]]
    elif elmo_sign == 'hit':
        elmo_part = elmo_hit.sents2elmo([sequence.split()])[0]
    assert (elmo_part.shape == (len(words), 1024))
    result = np.concatenate((glove_part.data, elmo_part), axis=1)
    return result
def saveSenResult(x_test, y_test_cls, y_pred_cls, save_path,weights,weight_type):
    """获得预测结果"""
    writer_true = open(save_path + 'test_true.txt', mode='w')
    writer_false = open(save_path + 'test_false.txt', mode='w')
    writer_true.write("预测\t真实\t句子\n")
    writer_false.write("预测\t真实\t句子\n")
    data_len = len(x_test)
    squ = []
    for i in range(data_len):
        if y_test_cls[i] == y_pred_cls[i]:
            writer_true.write(str(y_pred_cls[i]) + "\t" + str(y_test_cls[i]) + "\t" + str(x_test[i][0]) + "\n")
        else:
            writer_false.write(str(y_pred_cls[i]) + "\t" + str(y_test_cls[i]) + "\t" + str(x_test[i][0]) + "\n")
        squ.append(str(x_test[i][0]).split(' '))
    dic = {'sequences': 0, 'attention_weights': 1, 'rea_labels': 2, 'pre_labels': 3}
    dic['sequences'], dic['attention_weights'], dic['rea_labels'], dic[
        'pre_labels'] = squ, weights, y_test_cls, y_pred_cls
    with open(save_path + weight_type+"_attn_data.json", 'w', encoding='utf-8') as fw:
        json.dump(dic, fw, ensure_ascii=False, indent=4)
    gh = GenHtml()
    gh.gen(dic, weight_type,save_path)
     

def evaluate(evaluation_dataloader, model, criterion, using_GPU, type):
    """
    Evaluate the model on the given evaluation_dataloader
    :param evaluation_dataloader:
    :param model:
    :param criterion: loss criterion
    :param using_GPU: a boolean
    :return:
    """
    # Set model to eval mode, which turns off dropout.
    model.eval()
    num_correct = 0
    total_examples = 0
    total_eval_loss = 0
    predict, label, weights = [],[],[]
    confusion_matrix = np.zeros((3, 3))
    for (eval_text, eval_lengths, eval_labels) in evaluation_dataloader:
        eval_text = Variable(eval_text)
        eval_lengths = Variable(eval_lengths)
        eval_labels = Variable(eval_labels)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_lengths = eval_lengths.cuda()
            eval_labels = eval_labels.cuda()
        predicted, attention_loss,weight = model(eval_text, eval_lengths)
        # Calculate loss for this test batch. This is averaged, so multiply
        # by the number of examples in batch to get a total.
        total_eval_loss += criterion(predicted, eval_labels).item() * eval_labels.size(0)
        _, predicted_labels = torch.max(predicted.data, 1)
        total_examples += eval_labels.size(0)
        num_correct += torch.sum(predicted_labels == eval_labels.data)
        predict += predicted_labels.cpu().numpy().tolist()
        label += eval_labels.data.cpu().numpy().tolist()
        weights += weight.squeeze(dim=1).cpu().detach().numpy().tolist()
        for i in range(eval_labels.size(0)):
            confusion_matrix[eval_labels.data[i], int(predicted_labels[i])] += 1
    accuracy = 100 * num_correct / total_examples
    average_eval_loss = total_eval_loss / total_examples
    precision = [0, 0, 0]
    recall = [0, 0, 0]
    recall_sum = 0
    precision_sum = 0
    for num in range(3):
        if np.sum(confusion_matrix[num]) != 0:
            precision[num] = 100 * confusion_matrix[num, num] / np.sum(confusion_matrix[num])
        else:
            precision[num] = 0
        if np.sum(confusion_matrix[:,num]) != 0:
            recall[num] = 100 * confusion_matrix[num, num] / np.sum(confusion_matrix[:,num])
        else:
            recall[num] = 0
        recall_sum += recall[num]
        precision_sum += precision[num]
    recall = recall_sum / 3
    precision = precision_sum / 3
    model.train()
    # Set the model back to train mode, which activates dropout again.
    print(confusion_matrix)
    if type == 'train':
        P_macro, R_macro, F_macro, _ = metrics.precision_recall_fscore_support(label, predict, average="macro")
        P_micro, R_micro, F_micro, _ = metrics.precision_recall_fscore_support(label, predict, average="micro")
        print('Classfy P-micro: {0:>7.2} , Classfy R-micro: {1:>7.2} , Classfy F-micro: {2:>7.2}'.format(P_micro, R_micro,F_micro))
        print('Classfy P-macro: {0:>7.2} , Classfy R-macro: {1:>7.2} , Classfy F-macro: {2:>7.2}'.format(P_macro, R_macro,F_macro))
        return average_eval_loss,attention_loss[0][0], accuracy, F_macro, F_micro
    elif type == 'micro_test':
        details_result = classification_report(label, predict)
        P_micro, R_micro, F_micro, _ = metrics.precision_recall_fscore_support(label, predict, average="micro")
        return average_eval_loss, accuracy, P_micro, R_micro, F_micro, label, predict, confusion_matrix,weights,details_result
    elif type == 'macro_test':
        details_result = classification_report(label, predict)
        P_macro, R_macro, F_macro, _ = metrics.precision_recall_fscore_support(label, predict, average="macro")
        return average_eval_loss, accuracy, P_macro, R_macro, F_macro, label, predict, confusion_matrix,weights,details_result

# Make sure to subclass torch.utils.data.Dataset
class TextDatasetWithGloveElmoSuffix(Dataset):
    def __init__(self, embedded_text, labels, max_sequence_length=550):
        """
        :param embedded_text:
        :param labels: a list of ints
        :param max_sequence_length: an int
        """
        if len(embedded_text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        # A list of numpy arrays, where each inner numpy arrays is sequence_length * embed_dim
        # embedding for each word is : glove + elmo
        self.embedded_text = embedded_text
        # A list of ints, where each int is a label of the sentence at the corresponding index.
        self.labels = labels
        # Truncate examples that are longer than max_sequence_length.
        # Long sequences are expensive and might blow up GPU memory usage.
        self.max_sequence_length = max_sequence_length


    def __getitem__(self, idx):
        """
        Return the Dataset example at index `idx`.
        Returns
        -------
        example_text: numpy array
        length: int
            The length of the (possibly truncated) example_text.
        example_label: int 0 or 1
            The label of the example.
        """
        example_text = self.embedded_text[idx]
        example_label = self.labels[idx]
        # Truncate the sequence if necessary
        example_text = example_text[:self.max_sequence_length]
        example_length = example_text.shape[0]
        return example_text, example_length, example_label

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        """
        Given a list of examples (each from __getitem__),
        combine them to form a single batch by padding.
        Returns:
        -------
        batch_padded_example_text: LongTensor
          LongTensor of shape (batch_size, longest_sequence_length) with the
          padded text for each example in the batch.
        length: LongTensor
          LongTensor of shape (batch_size,) with the unpadded length of the example.
        example_label: LongTensor
          LongTensor of shape (batch_size,) with the label of the example.
        """
        batch_padded_example_text = []
        batch_lengths = []
        batch_labels = []
        # Get the length of the longest sequence in the batch
        max_length = max(batch, key=lambda example: example[1])[1]
        # Iterate over each example in the batch
        for text, length, label in batch:
            # Unpack the example (returned from __getitem__)
            # Amount to pad is length of longest example - length of this example.
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])
            # Append the pad_tensor to the example_text tensor.
            # Shape of padded_example_text: (padded_length, embeding_dim)
            # top part is the original text numpy,
            # and the bottom part is the 0 padded tensors
            # text from the batch is a np array, but cat requires the argument to be the same type
            # turn the text into a torch.FloatTenser, which is the same type as pad_tensor
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)
            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_labels.append(label)
        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_labels))
                
