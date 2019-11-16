import torch.nn as nn
import torch.nn.functional as F
from util import sort_batch_by_length
from util import last_dim_softmax
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch

class RNNSequenceClassifier(nn.Module):
    def __init__(self, args, embedding, query_embedding):
        # Always call the superclass (nn.Module) constructor first
        super(RNNSequenceClassifier, self).__init__()
        self.args = args
        self.rnn = nn.LSTM(input_size=args.input_dim, hidden_size=args.hidden_size,
                           num_layers=args.layers_num, dropout=args.dropout, batch_first=True, bidirectional=args.is_bidir)
        self.embedding = embedding
        if args.attention_layer == 'att':
            self.attention_weights = nn.Linear(args.hidden_size * args.is_bidir, 1)
            self.output_projection = nn.Linear(args.hidden_size * args.is_bidir, args.num_classes)
        else:
            self.query_embedding = query_embedding
            self.proquery_weights_mp = nn.Linear(args.hidden_size * args.is_bidir, args.attention_query_size)
            self.multi_output_projection = nn.Linear(args.hidden_size * args.is_bidir * args.num_classes, args.num_classes)
        self.dropout_on_input_to_LSTM = nn.Dropout(args.dropout)
        self.dropout_on_input_to_linear_layer = nn.Dropout(args.dropout)


    def forward(self, inputs, lengths, elmo_embedding):
        if self.args.pretrain_model_type == 'elmo':
            elmo_inputs = torch.Tensor().cuda()
            for i in range(len(inputs)):
                elmo_input = torch.from_numpy(elmo_embedding[' '.join(map(str, inputs[i].cpu().numpy()))].value).type(torch.cuda.FloatTensor)
                try:
                    elmo_inputs = torch.cat((elmo_inputs, elmo_input.unsqueeze(dim=0)))
                except:
                    print(elmo_inputs.shape, elmo_input.shape)
                    elmo_inputs = torch.cat((elmo_inputs, elmo_input.unsqueeze(dim=0)[:,:128,:]), dim=0)
            inputs = elmo_inputs
        else:
            inputs = self.embedding(inputs)


        # 1. input
        embedded_input = self.dropout_on_input_to_LSTM(inputs)
        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        packed_sorted_output, _ = self.rnn(packed_input)
        sorted_output, _ = pad_packed_sequence(packed_sorted_output, batch_first=True)
        output = sorted_output[input_unsort_indices]
        # 2. use attention
        if self.args.attention_layer == 'att':
            attention_logits = self.attention_weights(output).squeeze(dim=-1)
            mask_attention_logits = (attention_logits != 0).type(
                torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor)
            softmax_attention_logits = last_dim_softmax(attention_logits, mask_attention_logits)
            softmax_attention_logits = softmax_attention_logits.unsqueeze(dim=1)
            input_encoding = torch.bmm(softmax_attention_logits, output)
            input_encoding = input_encoding.squeeze(dim=1)
        else:
            input_encoding = torch.Tensor().cuda()
            querys = self.query_embedding(torch.arange(0,self.args.num_classes,1).cuda())
            attention_weights = torch.Tensor(self.args.num_classes, len(output), len(output[0])).cuda()
            for i in range(self.args.num_classes):
                attention_logits = self.proquery_weights_mp(output)
                attention_logits = torch.bmm(attention_logits, querys[i].unsqueeze(dim=1).repeat(len(output),1,1)).squeeze(dim=-1)
                mask_attention_logits = (attention_logits != 0).type(
                    torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor)
                softmax_attention_logits = last_dim_softmax(attention_logits, mask_attention_logits)
                input_encoding_part = torch.bmm(softmax_attention_logits.unsqueeze(dim=1), output)
                input_encoding = torch.cat((input_encoding,input_encoding_part.squeeze(dim=1)), dim=-1)
                attention_weights[i] = softmax_attention_logits
        # 3. run linear layer
        if self.args.attention_layer == 'att':
            input_encoding = self.dropout_on_input_to_linear_layer(input_encoding)
            unattized_output = self.output_projection(input_encoding)
            output_distribution = F.log_softmax(unattized_output, dim=-1)
            return output_distribution, softmax_attention_logits.squeeze(dim=1)
        else:
            input_encoding = self.dropout_on_input_to_linear_layer(input_encoding)
            unattized_output = self.multi_output_projection(input_encoding)
            output_distribution = F.log_softmax(unattized_output, dim=-1)

            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-16)
            attention_loss = abs(cos(querys[0], querys[1])) + abs(cos(querys[1], querys[2])) + abs(cos(querys[0], querys[2]))
            return output_distribution, attention_weights, attention_loss
