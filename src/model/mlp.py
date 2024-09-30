import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.input_dim = configs.get('input_dim')
        self.hidden_dim1 = configs.get('hidden_dim1')
        self.hidden_dim2 = configs.get('hidden_dim2')
        self.hidden_dim3 = configs.get('hidden_dim3')
        self.hidden_dim4 = configs.get('hidden_dim4')
        self.output_dim = configs.get('output_dim')
        self.dropout_ratio = configs.get('dropout_ratio')
        self.use_batch_norm = configs.get('use_batch_norm')

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.batch_normalization1 = nn.BatchNorm1d(self.hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.dropout_ratio)
        self.linear2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.batch_normalization2 = nn.BatchNorm1d(self.hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=self.dropout_ratio)
        self.linear3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.batch_normalization3 = nn.BatchNorm1d(self.hidden_dim3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=self.dropout_ratio)
        self.linear4 = nn.Linear(self.hidden_dim3, self.hidden_dim4)
        self.batch_normalization4 = nn.BatchNorm1d(self.hidden_dim4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=self.dropout_ratio)
        self.output = nn.Linear(self.hidden_dim4, self.output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.batch_normalization1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        if self.use_batch_norm:
            x = self.batch_normalization2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        if self.use_batch_norm:
            x = self.batch_normalization3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        if self.use_batch_norm:
            x = self.batch_normalization4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.output(x)

        return x 
