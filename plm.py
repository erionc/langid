
'''
Description:    Script that uses multilingual BERT for identifying languages.
Author:         Erion Çano
Reproduce:      Tested on Ubuntu 23.10 with CUDA=12.3, python=3.11.6, torch=2.1.0, tramsformers=4.35.0  
Run:            python plm.py (~25 minutes runtime on RTX 3080 mobile 16 GB)
'''

from utils import *
import torch 
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer,get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import warnings ; warnings.simplefilter("ignore")

# to ensure reproducibility of the results
s = 7 ; random.seed(s) ; np.random.seed(s) ; torch.manual_seed(s) ; torch.cuda.manual_seed_all(s)

# select the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__': 

    # Loading the dataset in a dataframe
    data = pd.read_csv("./data/languages.csv")
    # separating the texts from the language categories 
    X = data["Text"] ; y = data["Language"]

    # iterating through all samples to clean them
    for i, s in enumerate(X):
        X[i] = text_preprocess(s)

    # converting categorical variables to numerical
    le = LabelEncoder() ; y = le.fit_transform(y)
    # load tokenizer - selecting one that preserves casing
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # determine max sequence length - maximal sequence length must not exceed that of the PLM (here BERT)
    max_length = 0
    for t in X:
        input_ids = tokenizer.encode(t, add_special_tokens=True)
        max_length = max(max_length, len(input_ids))


    '''
    Limit max sequence length to that of BERT (512). This should have negative impact on performance. Using PLMs that support longer sequences should provide better performance. 
    '''
    if max_length > 512: max_length = 512


    # encode all samples in dataset
    input_ids, attention_masks = [], []
    for t in X:
        encoded_dict = tokenizer.encode_plus(t, add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt', truncation=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(y)

    # prepare full dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    # train-test splitting of the samples
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

    batch_size = 4  # number of sample per batch to use

    # train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)


    '''
    Selecting the PLM to use. This choice usually has a significant impact on task performance. The specific model selected below is based on BERT, is relatively small and is pretrained on a set of 104 languages covering all the 17 languages on the used dataset. Bigger PLMs with better architectures should provide better results.  
    '''
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=17, output_attentions=False, output_hidden_states=False, 
    )

    model = model.to(device)    # run model on gpu
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)    # AdamW optimizer


    '''
    Number of training epochs usually has a significant impact on task performance. If low the model may be udertrained and lack performance. If high the model may overfit and time / resources could be wasted.  
    '''
    epochs = 2
    total_steps = len(train_dataloader) * epochs    # training steps: nr_batches * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_tick = time.time()    # clock tick to start measuring training time

    for this_epoch in range(0, epochs):     # loop on each training epoch

        print(f"\n----------- Epoch {this_epoch + 1} / {epochs} ----------")

        epoch_tick = time.time()    # clock tick to start measuring training epoch time
        total_train_loss = 0 ; model.train()
        for step, batch in enumerate(train_dataloader):     # iterate on each training batch
            if step % 128 == 0:
                print(f"Step ................ {step:>5} / {len(train_dataloader)}")     # print progress

            # unpack the sample components
            b_input_ids = batch[0].to(device) ; b_input_mask = batch[1].to(device) ; b_labels = batch[2].to(device)
            optimizer.zero_grad()
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = output.loss              # compute batch loss
            total_train_loss += loss.item() # add batch loss to total loss
            
            loss.backward() # backward pass for the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # gradient clipping
            optimizer.step()    # update parameters 
            scheduler.step()    # update learning rate

        # computing average training loss from all batches
        avg_train_loss = total_train_loss / len(train_dataloader)            
        epoch_tok = time.time()    # clock tock to end measuring training epoch time
        epoch_time = format_time(epoch_tok - epoch_tick)    # training time during this epoch

        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Training epoch took: {epoch_time}")
        
        model.eval()
        total_eval_accuracy = 0 ; best_eval_accuracy = 0 ; total_eval_loss = 0 ; nb_eval_steps = 0

        # Evaluate data for this epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device) ; b_input_mask = batch[1].to(device) ; b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                output= model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = output.loss              # computing this loss
            total_eval_loss += loss.item()  # adding this loss to total loss
            
            # put logits and labels to CPU since GPU is used
            logits = output.logits ; logits = logits.detach().cpu().numpy() ; label_ids = b_labels.to('cpu').numpy()
            # accumulateed accuracy over the batches
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)     # accuracy of this validation run
        print(f"\nValidation accuracy: {avg_val_accuracy:.2f}")
        avg_val_loss = total_eval_loss / len(validation_dataloader) # accumulated loss over the batches

    train_tok = time.time() # clock tok to end measuring training time
    print(f"Total training took: {format_time(train_tok - train_tick)}")
