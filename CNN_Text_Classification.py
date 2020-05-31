#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks
# ---
# In this notebook, I'll train a **CNN** to classify the sentiment of movie reviews in a corpus of text. The approach will be as follows:
# * Pre-process movie reviews and their corresponding sentiment labels (positive = 1, negative = 0).
# * Load in a **pre-trained** Word2Vec model, and use it to tokenize the reviews.
# * Create training/validation/test sets of data.
# * Define a `SentimentCNN` model that has a pre-trained embedding layer, convolutional layers, and a final, fully-connected, classification layer.
# * Train and evaluate the model.
# 
# An example of a positive and negative review are shown below.
# 
# <img src='notebook_ims/reviews_ex.png' width=30% height=70% />
# 
# The task of text classification has typically been done with an RNN, which accepts a sequence of words as input and has a hidden state that is dependent on that sequence and acts as a kind of memory. You can see an example that classifies this same review dataset using an RNN in [this Github repository](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/sentiment-rnn/Sentiment_RNN_Solution.ipynb). 
# 
# 
# ## Resources
# 
# This example shows how you can utilize convolutional layers to find patterns in sequences of word embeddings and create an effective text classifier using a CNN-based approach.
# 
# **1. Original paper**
# * The code follows the structure outlined in the paper, [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) by Yoon Kim (2014). 
# 
# **2. Pre-trained Word2Vec model**
# 
# * The key to this approach is convolving over word embeddings, for which I will use a pre-trained [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) model. 
# * I am specifically using a "slim"-version of a model that was trained on part of a Google News dataset (about 100 billion words). The [original model](https://code.google.com/archive/p/word2vec/) contains 300-dimensional vectors for 3 million words and phrases.
# * The "slim" model is cut to 300k English words, as described in [this Github repository](https://github.com/eyaler/word2vec-slim).
# 
# You should be able to modify this code slightly to make it compatible with a Word2Vec model of your choosing.
# 
# **3. Movie reviews data **
# 
# The dataset holds 25000 movie reviews, which were obtained from the movie review site, IMDb.
# 

# ---
# ## Load in and Visualize the Data

# In[2]:


import numpy as np

# read data from text files
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()
with open('data/labels.txt', 'r') as f:
    labels = f.read()


# In[3]:


# print some example review/sentiment text
print(reviews[:1000])
print()
print(labels[:20])


# ---
# ## Data Pre-processing
# 
# The first step, when building a neural network, is getting the data into the proper form to feed into the network. Since I'm planning to use a word-embedding layer, I know that I'll need to encode each word in a reviews as an integer, and encode each sentiment label as 1 (positive) or 0 (negative). 
# 
# I'll first want to clean up the reviews by removing punctuation and converting them to lowercase. You can see an example of the reviews data, above. Here are the processing steps, I'll want to take:
# >* Get rid of any extraneous punctuation.
# * You might notice that the reviews are delimited with newline characters `\n`. To deal with those, I'm going to split the text into each review using `\n` as the delimiter. 
# * Then I can combined all the reviews back together into one big string to get all of my text data.
# 
# First, let's remove all punctuation. Then get all the text without the newlines and split it into individual words.

# In[8]:


reviews_split[0]


# In[4]:


from string import punctuation

# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])

# split by new lines and spaces
reviews_split = all_text.split('\n')

all_text = ' '.join(reviews_split)

# create a list of all words
all_words = all_text.split()


# ### Encoding the Labels
# 
# The review labels are "positive" or "negative". To use these labels in a neural network, I need to convert them to numerical values, 1 (positive) and 0 (negative).

# In[5]:


# 1=positive, 0=negative label conversion
labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])


# ### Removing Outliers
# 
# As an additional pre-processing step, I want to make sure that the reviews are in good shape for standard processing. That is, I'll want to shape the reviews into a specific, consistent length for ease of processing and comparison. I'll approach this task in two main steps:
# 
# 1. Getting rid of extremely long or short reviews; the outliers
# 2. Padding/truncating the remaining data so that we have reviews of the same length.
# 
# Before I pad the review text, below, I am checking for reviews of extremely short or long lengths; outliers that may mess with training.

# In[6]:


from collections import Counter

# Build a dictionary that maps indices to review lengths
counts = Counter(all_words)

# outlier review stats
# counting words in each review
review_lens = Counter([len(x.split()) for x in reviews_split])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# Okay, a couple issues here. I seem to have one review with zero length. And, the maximum review length is really long. I'm going to remove any super short reviews and truncate super long reviews. This removes outliers and should allow our model to train more efficiently.

# In[11]:


reviews_split[1]


# In[7]:


print('Number of reviews before removing outliers: ', len(reviews_split))

## remove any reviews/labels with zero length from the reviews_ints list.

# get indices of any reviews with length 0
non_zero_idx = [ii for ii, review in enumerate(reviews_split) if len(review.split()) != 0]

# remove 0-length reviews and their labels
reviews_split = [reviews_split[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers: ', len(reviews_split))


# ---
# ## Using a Pre-Trained Embedding Layer
# 
# Next, I'll want to tokenize my reviews; turning the list of words that make up a given review into a list of tokenized integers that represent those words. Typically, this is done by creating a dictionary that maps each unique word in a vocabulary to a specific integer value.
# 
# In this example, I'll actually want to use a mapping that already exists, in a pre-trained embedding layer. Below, I am loading in a pre-trained embedding model, and I'll explore its traits.
# 
# > This code assumes I have a downloaded model `GoogleNews-vectors-negative300-SLIM.bin.gz` in the same directory as this notebook, in a folder, `word2vec_model`.

# In[14]:


# # Load a pretrained word2vec model (only need to run code, once)
get_ipython().system('gzip -d word2vec_model/GoogleNews-vectors-negative300-SLIM.bin.gz')


# In[8]:


# import Word2Vec loading capabilities
from gensim.models import KeyedVectors

# Creating the model
embed_lookup = KeyedVectors.load_word2vec_format('word2vec_model/GoogleNews-vectors-negative300-SLIM.bin', 
                                                 binary=True)


# ### Embedding Layer
# 
# You can think of an embedding layer as a lookup table, where the rows are indexed by word token and the columns hold the embedding values. For example, row 958 is the embedding vector for the word that maps to the integer value 958.
# 
# <img src='notebook_ims/embedding_lookup_table.png' width=40% />
# 
# In the below cells, I am storing the words in the pre-trained vocabulary, and printing out the size of the vocabulary and word embeddings. 
# > The embedding dimension from the pret-rained model is 300.

# In[9]:


# store pretrained vocab
pretrained_words = []
for word in embed_lookup.vocab:
    pretrained_words.append(word)


# In[10]:


row_idx = 1

# get word/embedding in that row
word = pretrained_words[row_idx] # get words by index
embedding = embed_lookup[word] # embeddings by word

# vocab and embedding info
print("Size of Vocab: {}\n".format(len(pretrained_words)))
print('Word in vocab: {}\n'.format(word))
print('Length of embedding: {}\n'.format(len(embedding)))
#print('Associated embedding: \n', embedding)


# In[11]:


# print a few common words
for i in range(5):
    print(pretrained_words[i])


# ### Cosine Similarity
# 
# The pre-trained embedding model has learned to represent semantic relationships between words in vector space. Specifically, words that appear in similar contexts should point in roughly the same direction. To measure whether two vectors are colinear, we can use [**cosine similarity**](https://en.wikipedia.org/wiki/Cosine_similarity), which computes the dot product of two vectors. This dot product is largest when the angle between two vectors is 0 (cos(0) = 1) and cosine is at a maximum, so cosine similarity is larger for aligned vectors.
# 
# <img src='notebook_ims/two_vectors.png' width=30% />
# 
# ### Embedded Bias
# 
# Word2Vec, in addition to learning useful similarities and semantic relationships between words, also learns to represent problematic relationships between words. For example, a paper on [Debiasing Word Embeddings](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf) by Bolukbasi et al. (2016), found that the vector-relationship between "man" and "woman" was similar to the relationship between "physician" and "registered nurse" or "shopkeeper" and "housewife" in the trained, Google News Word2Vec model, **which I am using in this notebook**.
# 
# >*"In this paper, we quantitatively demonstrate that word-embeddings contain biases in their geometry that reflect gender stereotypes present in broader society. Due to their wide-spread usage as basic
# features, word embeddings not only reflect such stereotypes but can also amplify them. This poses a
# significant risk and challenge for machine learning and its applications."*
# 
# As such, it is important to note that this example is using a Word2Vec model that has been shown to encapsulate gender stereotypes.
# 
# You can explore similarities and relationships between word embeddings using code. The code below finds words with the highest cosine similarity when compared to the word `find_similar_to`. 

# In[12]:


# Pick a word 
find_similar_to = 'fabulous'

print('Similar words to '+find_similar_to+': \n')

# Find similar words, using cosine similarity
# by default shows top 10 similar words
for similar_word in embed_lookup.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.3f}".format(
        similar_word[0], similar_word[1]
    ))


# ## Tokenize reviews
# 
# The pre-trained embedding layer already has tokens associated with each word in the dictionary. I want to use that same mapping to tokenize all the reviews in the movie review corpus. I will encode any unknown words (words that appear in the reviews but not in the pre-trained vocabulary) as the whitespace token, 0; this should be fine for the purpose of sentiment classification.

# In[13]:


# convert reviews to tokens
def tokenize_all_reviews(embed_lookup, reviews_split):
    # split each review into a list of words
    reviews_words = [review.split() for review in reviews_split]

    tokenized_reviews = []
    for review in reviews_words:
        ints = []
        for word in review:
            try:
                idx = embed_lookup.vocab[word].index
            except: 
                idx = 0
            ints.append(idx)
        tokenized_reviews.append(ints)
    
    return tokenized_reviews


# In[15]:


tokenized_reviews = tokenize_all_reviews(embed_lookup, reviews_split)


# In[16]:


# testing code and printing a tokenized review
print(tokenized_reviews[0])


# ---
# ## Padding sequences
# 
# To deal with both short and very long reviews, I'll pad or truncate all the reviews to a specific length. For reviews shorter than some `seq_length`, I'll left-pad with 0s. For reviews longer than `seq_length`, I'll truncate them to the first `seq_length` words. A good `seq_length`, in this case, is about 200.
# 
# > The function `pad_features` returns an array that contains padded, tokenized reviews, of a standard size, that we'll pass to the network. 
# 
# 
# As a small example, if the `seq_length=10` and an input, tokenized review is: 
# ```
# [117, 18, 128]
# ```
# The resultant, padded sequence should be: 
# 
# ```
# [0, 0, 0, 0, 0, 0, 0, 117, 18, 128]
# ```
# 
# **Your final `features` array should be a 2D array, with as many rows as there are reviews, and as many columns as the specified `seq_length`.**
# 
# This isn't trivial and there are a bunch of ways to do this. But, if you're going to be building your own deep learning networks, you're going to have to get used to preparing your data.

# In[17]:


def pad_features(tokenized_reviews, seq_length):
    ''' Return features of tokenized_reviews, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    
    # getting the correct rows x cols shape
    features = np.zeros((len(tokenized_reviews), seq_length), dtype=int)

    # for each review, I grab that review and 
    for i, row in enumerate(tokenized_reviews):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features


# In[18]:


# Test your implementation!

seq_length = 200

features = pad_features(tokenized_reviews, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(tokenized_reviews), "Features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 8 values of the first 20 batches 
print(features[:20,:8])


# ---
# ## Training, Validation, and Test Data
# 
# With the data in nice shape, I'll split it into training, validation, and test sets.
# 
# In the below code, I am creating features (x) and labels (y). 
# * The split fraction, `split_frac` defines the fraction of data to **keep** in the training set. Usually this is set to 0.8 or 0.9. 
# * Whatever data is left is split in half to create the validation and test data.

# In[19]:


split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# **Check your work**
# 
# With train, validation, and test fractions equal to 0.8, 0.1, 0.1, respectively, the final, feature data shapes should look like:
# ```
#                     Feature Shapes:
# Train set: 		 (20000, 200) 
# Validation set: 	(2500, 200) 
# Test set: 		  (2500, 200)
# ```

# ## DataLoaders and Batching
# 
# After creating training, test, and validation data, I can create DataLoaders for this data by following two steps:
# 1. Create a known format for accessing our data, using [TensorDataset](https://pytorch.org/docs/stable/data.html#) which takes in an input set of data and a target set of data with the same first dimension, and creates a dataset.
# 2. Create DataLoaders and batch our training, validation, and test Tensor datasets.
# 
# ```
# train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
# train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
# ```
# 
# This is an alternative to creating a generator function for batching our data into full batches.

# In[21]:


import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 50

# shuffling and batching data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# ---
# # Sentiment Network with PyTorch
# 
# The complete model is made of a few layers:
# 
# **1. An [embedding layer](https://pytorch.org/docs/stable/nn.html#embedding)**
# * This converts our word tokens (integers) into embedded vectors of a specific size.
# * In this case, the vectors/weights of this layer will come from a **pretrained** lookup table. 
# 
# **2. A few [convolutional layers](https://pytorch.org/docs/stable/nn.html#conv1d)**
# * These are defined by an input size, number of filters/feature maps to output, and a kernel size.
# * The output of these layers will go through a ReLu activation function and pooling layer in the `forward` function.
# 
# **3. A fully-connected, output layer**
# * This maps the convolutional layer outputs to a desired output_size (1 sentiment class).
# 
# **4. A sigmoid activation layer**
# * This turns the output logit into a value 0-1; a class score.
# 
# There is also a dropout layer, which will prevent overfitting, placed between the convolutional outputs and the final, fully-connected layer.
# 
# <img src="notebook_ims/complete_embedding_CNN.png" width=60%>
# 
# *Image from the original paper, [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf).*
# 
# ### The Embedding Layer
# 
# The embedding layer comes from our pre-trained `embed_lookup` model. By default, the weights of this layer are set to the vectors from the pre-trained model and frozen, so it will just be used as a lookup table. You could train your own embedding layer here, but it will speed up the training process to use a pre-trained model.
# 
# ### The Convolutional Layer(s)
# 
# I am creating three convolutional layers, which will have kernel_sizes of (3, 300), (4, 300), and (5, 300); to look at 3-, 4-, and 5- sequences of word embeddings at a time. Each of these three layers will produce  100 filtered outputs. This is following the layer conventions in the paper, [CNNs for Sentence Classification](https://arxiv.org/abs/1408.5882).
# 
# > The kernels only move in one dimension: down a sequence of word embeddings. In other words, these kernels move along a sequence of words, in time!
# 
# ### Maxpooling Layers
# 
# In the `forward` function, I am applying a ReLu activation to the outputs of all convolutional layers and a maxpooling layer over the input sequence dimension. The maxpooling layer will get us an indication of whether some high-level text feature was found. 
# 
# > After moving through 3 convolutional layers with 100 filtered outputs each, these layers will output 300 values that can be sent to a final, fully-connected, classification layer.

# In[22]:


# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


# In[23]:


import torch.nn as nn
import torch.nn.functional as F

class SentimentCNN(nn.Module):
    """
    The embedding layer + CNN model that will be used to perform sentiment analysis.
    """

    def __init__(self, embed_model, vocab_size, output_size, embedding_dim,
                 num_filters=100, kernel_sizes=[3, 4, 5], freeze_embeddings=True, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentCNN, self).__init__()

        # set class vars
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        
        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # set weights to pre-trained
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors)) # all vectors
        # (optional) freeze embedding weights
        if freeze_embeddings:
            self.embedding.requires_grad = False
        
        # 2. convolutional layers
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=k-2) 
            for k in kernel_sizes])
        
        # 3. final, fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size) 
        
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.sig = nn.Sigmoid()
        
    
    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = F.relu(conv(x)).squeeze(3)
        
        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        embeds = self.embedding(x) # (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)
        
        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        
        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        
        # final logit
        logit = self.fc(x) 
        
        # sigmoid-activated --> a class score
        return self.sig(logit)
      


# ![image.png](attachment:image.png)

# In[20]:


import torch.nn as nn
get_ipython().run_line_magic('pinfo2', 'nn.Conv2d')


# ## Instantiate the network
# 
# Here, I'll instantiate the network. First up, defining the hyperparameters.
# 
# * `vocab_size`: Size of our vocabulary or the range of values for our input, word tokens.
# * `output_size`: Size of our desired output; the number of class scores we want to output (pos/neg).
# * `embedding_dim`: Number of columns in the embedding lookup table; size of our embeddings.
# * `num_filters`: Number of filters that each convolutional layer produces as output.
# * `filter_sizes`: A list of kernel sizes; one convolutional layer will be created for each kernel size.
# 
# Any parameters I did not list, are left as the default value.

# In[24]:


# Instantiate the model w/ hyperparams

vocab_size = len(pretrained_words)
output_size = 1 # binary class (1 or 0)
embedding_dim = len(embed_lookup[pretrained_words[0]]) # 300-dim vectors
num_filters = 100
kernel_sizes = [3, 4, 5]

net = SentimentCNN(embed_lookup, vocab_size, output_size, embedding_dim,
                   num_filters, kernel_sizes)

print(net)


# ---
# ## Training
# 
# Below is some training code, which iterates over all of the training data, records some loss statistics and performs backpropagation + optimization steps to update the weights of this network.
# 
# >I'll also be using a binary cross entropy loss, which is designed to work with a single Sigmoid output. [BCELoss](https://pytorch.org/docs/stable/nn.html#bceloss), or **Binary Cross Entropy Loss**, applies cross entropy loss to a single value between 0 and 1.
# 
# I also have some training hyperparameters:
# 
# * `lr`: Learning rate for the optimizer.
# * `epochs`: Number of times to iterate through the training dataset.

# In[25]:


# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


# In[26]:


# training loop
def train(net, train_loader, epochs, print_every=100):

    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()

    counter = 0 # for printing
    
    # train for some number of epochs
    net.train()
    for e in range(epochs):

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output = net(inputs)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = net(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))


# In[27]:


# training params

epochs = 2 # this is approx where I noticed the validation loss stop decreasing
print_every = 100

train(net, train_loader, epochs, print_every=print_every)


# ---
# ## Testing
# 
# 
# * **Test data performance:**  I'll see how our trained model performs on all of the defined test_data, above; I'll calculate the average loss and accuracy over the test data.
# 

# In[28]:


# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0


net.eval()
# iterate over test data
for inputs, labels in test_loader:

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # get predicted outputs
    output = net(inputs)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))


# In[ ]:




