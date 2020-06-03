# NLP, Extracting Numeral Relations in Financial Tweets
A model for detecting numeral attachments given a target entity. Understanding the relationship between entities and numerals in a natural language plays a crucial role in many domains. It is especially important in health records, financial data, and geographic documents. The model presented here is a part of the FinNum2 challenge, which is a task for fine-grained numeral understanding in financial social media data. We take a look at how one-way word-by-word attention and Recurrent Neural Networks can be used for encoding and classifying the relationships between stock symbols and numerals.

 ***Read the report here [report](report.pdf)***


## How to run 
1. `pip install virtualenv`
2. `virtualenv finnum2`
3. `source finnum2/bin/activate` (on windows: `finnum2\Scripts\activate`)
4. `pip install -r requirements.txt`
5. Run `main.py` file


## Architecture
At a high-level, the model contains two encoders and one classifier. The first encoder, encodes the possible relation, while the second encoder, encodes the tweet. When encoding the tweet, the model has a one-way attention mechanism between the encoders, to give the model focus on the relation while encoding the tweet. We then grab the encoded hidden state and last output and feed it into the classifier. The classifier decodes the information using a neural layer. Here is a high-level architecture of the model:

![Architecture](/images/Entity-relation-extraction.png)


#### Word embedding
The model process tokens at a word level, instead of for example characters-by-character. The vocabulary threshold is set to 10, which means word with frequent less than 10 is converted to a common token ("unk"). The embedding is trained with the model. 

#### Pre-processing
To able to extract information from the tweets, all numbers are replaced with a num<index> token. This limits the model to only understand sentences with a fixed amount of numbers, as the embedding only learn the fixed numbers. The same is done with the tickers. 

An alternative approach could be to always assign a given token to the target number and ticker, and assign the rest of the number and ticker a given token respectively, eg "num" and "ticker". This means that we are dropping the information on the index of the numbers. Experiments show that including the index in the number has yield better results than without. 


#### Attention mechanism 
Attention mechanisms is often found in Seq2Seq model. Normally, you would have an attention decoder with teacher enforcing. In the presented model here, the attention mechanism is connected to a second encoder, the tweet encoder. We therefore does not include teacher enforcing as the encoder does not output a sentence, but an internal encoding. The reason to still include the attention mechanism is to make the network able to focus on the relation. The attention weights are calculated with a feed-forward layer, attn. The figure describe one iteration of a word token. The initial hidden state is from the relation encoder, before the tweet encoder produces the next. After attention weights are calculated, a softmax function is applied and we get the final attention-weights. We then perform a batch matrix-matrix product between the attention-weights and the output from the relation encoder. After the attention has been applied, the matrix is combined with the embedded input token and a ReLU function is applied before the matrix is sent to the GRU. 


![attention](/images/attention-decoder-network.png)

#### Classifier
The classifier is a neural network layer, which takes as input the last output vector and the hidden feature vector from the tweet encoder. The two vectors are combined to a matrix and forward-passed to the classifier. The model then outputs a binary classification. 


## Results
| F1-micro      | F1-macro      | Accuracy|
| ------------- |:-------------:| -----:|
| 89.15%      | 78.58% | 89.15% |


v2 (fixed an bug with the implementation of the word embedding)
| F1-micro      | F1-macro      | Accuracy|
| ------------- |:-------------:| -----:|
| 90.54%      | 83.00% | 90.54% |
