import torch
from transformers import BertTokenizer, BertConfig
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import cleanData
import re
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def clean_data(sentence):
	corpus = ""
	#this loads the data from sample_corpus.txt
	temp  = sentence.replace('\n','')
	temp = re.sub(r'[^a-zA-Z ,.,\']', ' ', temp)
	arrayOfWords = temp.split()
	corpus = ""
	for i in range(0,len(arrayOfWords)):
		if(arrayOfWords[i].count('.') >= 2):
			arrayOfWords[i] = arrayOfWords[i].replace('.', ' ')
	for i in range(0, len(arrayOfWords)):
		if(arrayOfWords[i].count(' ') > 1):
			count = arrayOfWords[i].count(' ')
			arrayOfWords[i] = arrayOfWords[i].replace(' ', '', count - 1)
		# if(re.search("\W",arrayOfWords[i]) != None):
		# 	x = re.sub("\W,", " ", arrayOfWords[i])
		# 	print("Not word character, ", i)
		# 	#print(x)
	if(len(arrayOfWords) != 0):
		if(arrayOfWords[len(arrayOfWords)-1].find('.') == -1):
			arrayOfWords[len(arrayOfWords)-1] = arrayOfWords[len(arrayOfWords)-1]+"."

	for word in arrayOfWords:
		corpus = corpus + " " + word
	corpus = corpus.strip()	
	if(len(corpus.split()) < 5):
		return None
	elif(len(corpus.split()) > 400):
		return None
	return corpus
# df = pd.read_csv("../data/training_bad_vape.csv", delimiter = '\t', header = None, names = ['label', 'sentence']) 
def main():
	df_cold_turkey = pd.read_csv("../trainingSet/training_cold_turkey.csv", delimiter = ',', header = None, names = ['label', 'sentence'])
	df_vaping_ex = pd.read_csv("../trainingSet/training_vape_ex.csv", delimiter = ',', header = None, names = ['label', 'sentence'])
	df_bad_vape = pd.read_csv("../trainingSet/training_bad_vape.csv", delimiter = ',', header = None, names = ['label', 'sentence'])

	# df3 = pd.read_csv("../data/training_vape_ex.csv" , delimiter = '\t', header = None, names = ['label', 'sentence'])

	#Stores all training data into a single data frame
	# df.append(df2, ignore_index = True)
	# df.append(df3, ignore_index = True)
	# print(df["label"])
	df_cold_turkey = df_cold_turkey.dropna(subset=['sentence', 'label'])
	df_cold_turkey["sentence"] = df_cold_turkey["sentence"].apply(clean_data)
	df_cold_turkey = df_cold_turkey.dropna(subset=['sentence', 'label'])

	df_vaping_ex = df_vaping_ex.dropna(subset=['sentence', 'label'])
	df_vaping_ex["sentence"] = df_vaping_ex["sentence"].apply(clean_data)
	df_vaping_ex = df_vaping_ex.dropna(subset=['sentence', 'label'])

	df_bad_vape = df_bad_vape.dropna(subset=['sentence', 'label'])
	df_bad_vape["sentence"] = df_bad_vape["sentence"].apply(clean_data)
	df_bad_vape = df_bad_vape.dropna(subset=['sentence', 'label'])

	allSents = df_cold_turkey.sentence.values
	allSents = np.concatenate((allSents,df_bad_vape.sentence.values))
	allSents = np.concatenate((allSents,df_vaping_ex.sentence.values))
	labels = pd.to_numeric(df_cold_turkey.label.values)
	labels = np.concatenate((labels,pd.to_numeric(df_bad_vape.label.values)))
	labels = np.concatenate((labels,pd.to_numeric(df_vaping_ex.label.values)))
	allSents = ["[CLS] " + sentence + " [SEP]" for sentence in allSents]
	print(labels)
	# print(allSents)
	#initalizes pretrianed model and tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	tokenized_texts = [tokenizer.tokenize(sent) for sent in allSents]

	print ("Tokenize the first sentence:")
	print (tokenized_texts[0])
	# #tokenizez's sents and create list of sentences.  Also gets max sequence length
	# maxLen = 0
	# tokenized = []
	# for sents in allSents:
	# 	temp = tokenizer.tokenize(sents)
	# 	if len(temp) > maxLen:
	# 		maxLen = len(temp)
	# 	tokenized.append(temp)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 	n_gpu = torch.cuda.device_count()
# 	torch.cuda.get_device_name(0)

	# Set the maximum sequence length. 
	MAX_LEN = 512
	# Pad our input tokens
	input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
	                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
	# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
	input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")



	# Create attention masks
	attention_masks = []
	# Create a mask of 1s for each token followed by 0s for padding
	for seq in input_ids:
		seq_mask = [float(i>0) for i in seq]
		attention_masks.append(seq_mask)
	# input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized]

	# Use train_test_split to split our data into train and validation sets for training
	train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,random_state=2018, test_size=0.1)
	train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,random_state=2018, test_size=0.1)
	                                             
	# Convert all of our data into torch tensors, the required datatype for our model
	train_inputs = torch.tensor(train_inputs)
	validation_inputs = torch.tensor(validation_inputs)
	train_labels = torch.tensor(train_labels)
	validation_labels = torch.tensor(validation_labels)
	train_masks = torch.tensor(train_masks)
	validation_masks = torch.tensor(validation_masks)

	# Select a batch size for training. 
	batch_size = 32

	# Create an iterator of our data with torch DataLoader 
	train_data = TensorDataset(train_inputs, train_masks, train_labels)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
	validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
	validation_sampler = SequentialSampler(validation_data)
	validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 

	model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
# 	model.cuda()
	# BERT model summary
	# BERT fine-tuning parameters
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'gamma', 'beta']
	optimizer_grouped_parameters = [
	    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
	     'weight_decay_rate': 0.01},
	    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
	     'weight_decay_rate': 0.0}
	]

	optimizer = BertAdam(optimizer_grouped_parameters,lr=2e-5,warmup=.1)

	train_loss_set = []
	# Number of training epochs 
	epochs = 4

	# BERT training loop
	for _ in trange(epochs, desc="Epoch"):  
	  
	  ## TRAINING
	  
	  # Set our model to training mode
	  model.train()  
	  # Tracking variables
	  tr_loss = 0
	  nb_tr_examples, nb_tr_steps = 0, 0
	  # Train the data for one epoch
	  for step, batch in enumerate(train_dataloader):
	    # Add batch to GPU
	    batch = tuple(t.to(device) for t in batch)
	    # Unpack the inputs from our dataloader
	    b_input_ids, b_input_mask, b_labels = batch
	    # Clear out the gradients (by default they accumulate)
	    optimizer.zero_grad()
	    # Forward pass
	    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
	    train_loss_set.append(loss.item())    
	    # Backward pass
	    loss.backward()
	    # Update parameters and take a step using the computed gradient
	    optimizer.step()
	    # Update tracking variables
	    tr_loss += loss.item()
	    nb_tr_examples += b_input_ids.size(0)
	    nb_tr_steps += 1
	  print("Train loss: {}".format(tr_loss/nb_tr_steps))
	       
	  ## VALIDATION

	  # Put model in evaluation mode
	  model.eval()
	  # Tracking variables 
	  eval_loss, eval_accuracy = 0, 0
	  nb_eval_steps, nb_eval_examples = 0, 0
	  # Evaluate data for one epoch
	  for batch in validation_dataloader:
	    # Add batch to GPU
	    batch = tuple(t.to(device) for t in batch)
	    # Unpack the inputs from our dataloader
	    b_input_ids, b_input_mask, b_labels = batch
	    # Telling the model not to compute or store gradients, saving memory and speeding up validation
	    with torch.no_grad():
	      # Forward pass, calculate logit predictions
	      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    
	    # Move logits and labels to CPU
	    logits = logits.detach().cpu().numpy()
	    label_ids = b_labels.to('cpu').numpy()
	    tmp_eval_accuracy = flat_accuracy(logits, label_ids)    
	    eval_accuracy += tmp_eval_accuracy
	    nb_eval_steps += 1
	  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

	# plot training performance
	plt.figure(figsize=(15,8))
	plt.title("Training loss")
	plt.xlabel("Batch")
	plt.ylabel("Loss")
	plt.plot(train_loss_set)
	plt.show()

	# for i in input_ids:
	# 	if len(i) <= maxLen:
	# 		i.append(0)

	# print(input_ids)


if __name__ == "__main__":
    main()	


