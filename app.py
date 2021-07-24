import streamlit as st
import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
import numpy as np
import pandas as pd

from transformers import BertTokenizer,BertForSequenceClassification
import os
import time

import urllib
from random import randint

# from SessionState import _SessionState, _get_session, _get_state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache()
def load_model(MODEL_PATH = 'sancharidan/scibet_expertfinder'):
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    return (model, tokenizer)

def main():
    st.set_page_config(page_title="SMU Expert Finder", page_icon="🛸")

    model, tokenizer = load_model()
    # set_seed(42)  # for reproducibility

    st.title("SMU Expert Finder")

    text_input = st.text_input("Please enter research area for which you seek experts", key="topic_textbox")
    selectbox = st.selectbox('Please select School from which you wish to retrieve experts for above research area',\
     ('SCIS', 'Business', 'All'),index = 2, key = 'school_select')
    slider = st.slider('Please choose number of experts you wish to retrieve', 1, 10, key = 'num_experts_slider')

    button_generate = st.button("Find Experts")

    if button_generate:
        try:
          QUERY = text_input
          EXPERT_SCHOOL = selectbox
          NUM_EXPERTS = slider
          # get expert database
          print ('\nReading Expert Database...')
          if EXPERT_SCHOOL.lower() == 'scis':
               expert_db = pd.concat([pd.read_csv('./Data/SIS_Faculty_Data.csv',index_col = False)])
          elif EXPERT_SCHOOL.lower() == 'business':
               expert_db = pd.concat([pd.read_csv('./Data/Business_Faculty_Data.csv',index_col = False)])
          elif EXPERT_SCHOOL.lower() == 'all':
               expert_db = pd.concat([pd.read_csv('./Data/SIS_Faculty_Data.csv',index_col = False),\
                              pd.read_csv('./Data/Business_Faculty_Data.csv', index_col = False)])

          # get experts and write to output path
          experts,prob = get_experts(model, tokenizer, QUERY, expert_db, NUM_EXPERTS)
          df = pd.DataFrame({'Name':experts, 'Probability':prob})
          del experts, prob
          df['Query'] = QUERY
          df = df[['Query','Name','Probability']]
#           print('\nWriting to output file...')
#           df.to_csv('./Output/results.csv',index=False)
          st.write('Displaying top {} experts in the field of {} from {} school'.format(NUM_EXPERTS,QUERY.upper(),EXPERT_SCHOOL.upper()))

          # df = pd.read_csv('./Output/results.csv',index_col=False)
          st.dataframe(df)
         
        except:
            pass

# Functions for prediction
def get_features(model, tokenizer, head, relation = None, tail = None, max_seq_length = 128):
    tokens_head = tokenizer.tokenize(head)
    tokens = ["[CLS]"] + tokens_head + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    
    if relation:
        tokens_relation = tokenizer.tokenize(relation)
        tokens += tokens_relation + ["[SEP]"]
        segment_ids += [1] * (len(tokens_relation) + 1)
        
    if tail:
        tokens_tail = tokenizer.tokenize(tail)
        tokens += tokens_tail + ["[SEP]"]
        segment_ids += [1] * (len(tokens_tail) + 1)
        
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    return tokens, input_ids, input_mask, segment_ids

def get_predictions(model, tokenizer, sequences, batch_size):
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    logits = []
    for sequence in sequences:
        
        tokens_enc, input_ids, input_mask, segment_ids = get_features(model, tokenizer, sequence[0], relation = sequence[1], tail = sequence[2])
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
        segment_ids_list.append(segment_ids)
        
    all_input_ids = torch.tensor([input_ids for input_ids in input_ids_list], dtype=torch.long)
    all_input_mask = torch.tensor([input_mask for input_mask in input_mask_list], dtype=torch.long)
    all_segment_ids = torch.tensor([segment_ids for segment_ids in segment_ids_list], dtype=torch.long)

    all_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    sampler = SequentialSampler(all_data)
    dataloader = DataLoader(all_data, sampler=sampler, batch_size=batch_size)
    
    for step, batch in enumerate(dataloader):
        # print ("Getting predictions for Batch",step)
        b_input_ids = batch[0]#.to(device)
        b_input_mask = batch[1]#.to(device)
        b_segment_id = batch[2]#.to(device)
        outputs = model(b_input_ids, 
                        token_type_ids=b_segment_id, 
                        attention_mask=b_input_mask)
        logits.append(outputs[0])
    predictions = torch.cat(logits, dim=0)
    predictions = predictions.detach().cpu().numpy()
    del input_ids_list, input_mask_list, segment_ids_list, all_input_ids, all_input_mask, all_segment_ids, all_data, sampler, dataloader, logits
    return predictions 
    
def get_experts(model, tokenizer, expertise_area, expert_db, num_experts = 50):
     print('\nGetting experts for ',expertise_area)
     experts = expert_db['Name'].unique()
     l = [[]]
     for expert in experts:
          l = l + [[expert,'researches in',expertise_area]]
     pred = get_predictions(model, tokenizer, l[1:],1)
     m = torch.nn.Softmax(dim=1)
     output = m(torch.tensor(pred))
     output = output.detach().cpu().numpy()
     neg,pos = output[:,0], output[:,1]
     pos1 = pos[np.argsort(-pos)][:num_experts]
     experts1 = experts[np.argsort(-pos)][:num_experts]
     # out = experts1[pos1[0].tolist()]
     # df = pd.DataFrame({'Name':experts1, 'Probability':pos1})
     del experts, l, pred, m, output, neg, pos
     return experts1,pos1   


if __name__ == "__main__":
    main()
