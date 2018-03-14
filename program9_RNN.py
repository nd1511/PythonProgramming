
# we use RNNs
# we have many architectures of RNNs, one-to-one, one-to-many, many-to-one, many-to-many

# for time, we use the many-to-one RNN
# we have time, s_{t-1} s_t s_{t+1}

# to understand RNNs, we unroll them
# we unroll and unfold the RNN

# we have x=input, x \in R^{n \times m}, a=hidden recurrent state, a \in R^{p \times m}
# U=weight from x to a, U \in R^{p \times m}
# W=weight from a_{t-1} to a_t, W \in R^{p \times p}
# V=weight from a to h, h=output, V in R^{k \times p}

import torch

import numpy as np

from torch.autograd import Variable

import matplotlib.pyplot as plt



# we do data pre-processing

#with open('lyrics')
#df = pd.read_csv('/Users/dionelisnikolaos/Downloads/creditcard.csv')

with open('/Users/dionelisnikolaos/Downloads/lyrics.rtf', 'r') as file:
    rawtxt = file.read()

# we use lower-case letters
rawtxt = rawtxt.lower()



# we find the unique characters

def create_map(rawtxt):

    letters = list(set(rawtxt))

    lettermap = dict(enumerate(letters))

    return lettermap

num_to_let = create_map(rawtxt)

# we print the mapping
#print(num_to_let)



# we define the inverse mapping
let_to_num = dict(zip(num_to_let.values(), num_to_let.keys()))



def maparray(txt, mapdict):

    # we now use a list
    txt = list(txt)

    for k, letter in enumerate(txt):
        txt[k] = mapdict[letter]

    txt = np.array(txt)

    return txt

X = maparray(rawtxt, let_to_num)

#print(X)

# we use roll, we shift our values by one
Y = np.roll(X, -1, axis=0)



# we use LongTensor, we use discrete values, we use integers
X = torch.LongTensor(X)

# we use LongTensor, we use discrete values, we use integers
Y = torch.LongTensor(Y)

# we now finish pre-processing



def random_chunk(chunk_size):
    k = np.random.randint(0, len(X)-chunk_size)

    return X[k:k+chunk_size], Y[k:k+chunk_size]

print(random_chunk(5))



# we define the unique characters
nchars = len(num_to_let)



# we now define the RNN model
class rnn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.n_layers = n_layers

        self.encoder = torch.nn.Embedding(input_size, hidden_size)

        self.rnn = torch.nn.RNN(hidden_size, hidden_size, n_layers,
                                batch_first=True)

        self.decoder = torch.nn.Linear(hidden_size, output_size)



    def forward(self, x, hidden):
        x = self.encoder(x.view(1, -1))

        # we use view, we re-shape the Tensor

        output, hidden = self.rnn(x.view(1, 1, -1), hidden)

        output = self.decoder(output.view(1, -1))

        # the tanh activation function is used

        return output, hidden



    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))



# we define the hyper-parameters
lr = 0.003

no_epochs = 10

chunk_size = 100



myrnn = rnn(nchars, 100, nchars, 1)

# we use the CE cost function
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(myrnn.parameters(), lr=lr)



# we now plot the cost
costs = []

plt.ion()

fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_xlabel('Epoch')

ax.set_ylabel('Cost')

plt.show()



for epoch in range(no_epochs):
    totcost = 0

    generated = ''

    # we use rounded division
    for _ in range(len(X) // chunk_size):

        h = myrnn.init_hidden()

        cost = 0

        x, y = random_chunk(chunk_size)

        x, y = Variable(x), Variable(y)



        for i in range(chunk_size):

            # we use the k-th chunk
            out, h = myrnn.forward(x[i], h)

            _, outl = out.data.max(1)

            #letter = num_to_let(outl[0])
            letter = num_to_let[outl[0]]

            generated += letter

            cost += criterion(out, y[i])



        optimizer.zero_grad()

        cost.backward()

        optimizer.step()

        totcost += cost



    totcost /= len(X)//chunk_size

    costs.append(totcost.data[0])

    ax.plot(costs, 'b')

    fig.canvas.draw()

    # we pause the plot so as to see the graph
    plt.pause(0.001)

    print('Epoch', epoch, " Avg Cost/chunk: ", totcost)

    print('Generated text: ', generated[0:750], '\n')







