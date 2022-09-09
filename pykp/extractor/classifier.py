import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self, input_size):
        '''
        :param input_size: word_vec_size
        '''
        super().__init__()
        self.input_size = input_size
        self.class_num = 3
        self.linear_1 = nn.Linear(in_features=self.input_size, out_features=self.input_size, bias=True)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(in_features=self.input_size, out_features=self.class_num, bias=True)

    def forward(self, n_grams_f):
        '''
        :param n_grams_f: batch_size * src_len * word_vec_size
        :return class_dist: batch_size * src_len * class_num(3)
        '''
        return nn.functional.softmax(self.linear_2(self.tanh(self.linear_1(n_grams_f))), -1)