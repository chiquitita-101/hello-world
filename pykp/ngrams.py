import torch
import torch.nn as nn
class NGramers(nn.Module):
    def __init__(self, input_size, output_size, max_gram, dropout):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_gram = max_gram
        self.dropout = dropout
        
        self.conv_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=input_size, 
                    out_channels=output_size,
                    kernel_size=n,
                    padding=int((n-1)/2)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for n in range(1, max_gram + 1)])

        self.relu_a = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU()
            )
        self.task_specific_norm_a = nn.LayerNorm(input_size)
        self.co_attention_norm_a = nn.LayerNorm(input_size)
        self.relu_p = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.ReLU()
            )
        self.task_specific_norm_p = nn.LayerNorm(output_size)
        self.co_attention_norm_p = nn.LayerNorm(output_size)
        self.softmax = nn.Softmax(dim=2)

        self.linear_a = nn.Linear(max_gram*input_size, input_size)
        self.linear_p = nn.Linear(max_gram*output_size, output_size)
        
    #padding one conlumn of zero at the end of input
    def customized_padding(self, x):
        #x: batch_size*input_length*embed_size
        #output: batch_size*(input_length + 1)*embed_size
        #ConstantPad1d: N,C,W_i  N,C,W_o  padding_left, padding_right
        ZeroPad = nn.ConstantPad1d(padding=(0, 1), value=0)
        return ZeroPad(x)

    def forward(self, x):
        #x: batch_size * seq_len * embedding_size
        x_ = x.transpose(1, 2)
        y = []
        for i in range(len(self.conv_list)):
            if (i + 1) % 2 == 0:
                y_ = self.conv_list[i](self.customized_padding(x_))
            else:
                y_ = self.conv_list[i](x_)
            y.append(y_.transpose(1, 2))
        #task-specific
        #absent kp generation
        residual_a = x #A^l
        x = self.relu_a(x)
        x = residual_a + x
        x = self.task_specific_norm_a(x)#A^l'
        #present kp extraction
        features_p = []#P^l'
        for y_ in y:
            residual_p = y_ #P^l
            y_ = self.relu_p(y_)
            y_ = residual_p + y_
            y_ = self.task_specific_norm_p(y_)
            features_p.append(y_)
        #co-attention
        y_t = ()
        x_t = ()
        residual_a = x
        for y_ in features_p:
            #present
            residual_p = y_
            y_ = torch.matmul(self.softmax(torch.matmul(y_, x.transpose(1, 2))), x)
            y_ = residual_p + y_
            y_ = self.co_attention_norm_p(y_)
            y_t = y_t + (y_, )
            #absent
            x_ = torch.matmul(self.softmax(torch.matmul(x, y_.transpose(1, 2))), y_)
            x_ = residual_a + x_
            x_ = self.co_attention_norm_a(x_)
            x_t = x_t + (x_, )
        
        result_a = self.linear_a(torch.cat(x_t, 2))
        result_p = self.linear_p(torch.cat(y_t, 2))
        return result_a, result_p