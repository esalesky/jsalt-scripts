import torch
import numpy as np
import torch.utils.data as utils
import sys

class FFN(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(FFN, self).__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.input_layer  = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu         = torch.nn.ReLU()
        self.hidden_layer = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid      = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.input_layer(x)
        relu   = self.relu(hidden)
        output = self.hidden_layer(relu)
        output = self.sigmoid(output)
        
        return output

##############################################

input_size  = 1024
hidden_size = 1024
lr = 0.01
epochs = 10

#----------
#data setup
def data_prep(source_file, target_file):
    src = np.load(source_file)
    tgt = np.load(target_file)

    #checkme
    srclabel = np.zeros(src.shape[0])
    tgtlabel = np.ones(tgt.shape[0])

    train = torch.FloatTensor(np.concatenate((src, tgt)))
    truth = torch.FloatTensor(np.concatenate((srclabel, tgtlabel)))

    return train, truth

train, train_label = data_prep(sys.argv[1], sys.argv[2]) #src_train, tgt_train
test, test_label   = data_prep(sys.argv[3], sys.argv[4]) #src_test,  tgt_test


#----------
#model setup
model     = FFN(input_size, hidden_size)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)


#----------
#pre-train check
model.eval()
test_pred = model(test)
pre_loss  = criterion(test_pred.squeeze(), test_label)
total     = test_label.size(0)
correct   = ((test_pred.data > 0.5).view(1,-1).int() == test_label.int()).sum().item()
acc       = correct*100/total
print('Test loss before training: %.3f, Accuracy: %.3f' % (pre_loss.item(), acc))

#train
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    #forward pass
    train_pred = model(train)
    #compute loss
    train_loss = criterion(train_pred.squeeze(), train_label)
    
    print('Epoch %d: train loss: %.3f' % (epoch, train_loss.item()))
    #backward pass
    train_loss.backward()
    optimizer.step()

#evaluate    
model.eval()
test_pred = model(test)
post_loss = criterion(test_pred.squeeze(), test_label)
total     = test_label.size(0)
correct   = ((test_pred.data > 0.5).view(1,-1).int() == test_label.int()).sum().item()
acc       = correct*100/total
print('Test loss after training: %.3f, Accuracy: %.3f' % (post_loss.item(), acc))
