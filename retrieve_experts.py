# import torch
# from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
#                               TensorDataset)
# import numpy as np
# import pandas as pd

# from transformers import BertTokenizer,BertForSequenceClassification
# import os

# import argparse
# ###-----Parameters - input from user----###
# parser = argparse.ArgumentParser()
# parser.add_argument('--MODEL_PATH', default = './Model/scibert_6_epochs_3105_pub_yes_distinctsegid_yes_entvocab_no', help = 'Specify path to model directory')
# parser.add_argument('--EXPERT_SCHOOL',  default = "SCIS", help = 'Choose between SCIS, Business and All. Default Value = SCIS')
# parser.add_argument('--NUM_EXPERTS',  type = int, default = 10, help = 'Specify number of experts to be returned')
# parser.add_argument('--QUERY', default = 'Artificial Intelligence', help = 'Query Research Area for which to find experts')
# parser.add_argument('--OUTPUT_PATH',  default = "./Output/results.csv", help = 'Output filepath')

# # parse input arguments
# inP = parser.parse_args()
# MODEL_PATH = inP.MODEL_PATH # path to model binary
# EXPERT_SCHOOL = inP.EXPERT_SCHOOL # to query from a specific school or both schools
# NUM_EXPERTS = inP.NUM_EXPERTS # number of experts to be returned
# QUERY = inP.QUERY # query research area
# OUTPUT_PATH = inP.OUTPUT_PATH # path to output file for storing scraped data in csv format

# # Load a trained model and vocabulary that you have fine-tuned
# model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
# tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# # Copy the model to the GPU.
# # model.to(device)

# # Functions for prediction
# max_seq_length = 128
# def get_features(head, relation = None, tail = None):
#     tokens_head = tokenizer.tokenize(head)
#     tokens = ["[CLS]"] + tokens_head + ["[SEP]"]
#     segment_ids = [0] * len(tokens)
    
#     if relation:
#         tokens_relation = tokenizer.tokenize(relation)
#         tokens += tokens_relation + ["[SEP]"]
#         segment_ids += [1] * (len(tokens_relation) + 1)
        
#     if tail:
#         tokens_tail = tokenizer.tokenize(tail)
#         tokens += tokens_tail + ["[SEP]"]
#         segment_ids += [1] * (len(tokens_tail) + 1)
        
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_mask = [1] * len(input_ids)
#     # Zero-pad up to the sequence length.
#     padding = [0] * (max_seq_length - len(input_ids))
#     input_ids += padding
#     input_mask += padding
#     segment_ids += padding
    
#     assert len(input_ids) == max_seq_length
#     assert len(input_mask) == max_seq_length
#     assert len(segment_ids) == max_seq_length
    
#     return tokens, input_ids, input_mask, segment_ids

# def get_predictions(sequences, batch_size):
#     input_ids_list = []
#     input_mask_list = []
#     segment_ids_list = []
#     logits = []
#     for sequence in sequences:
        
#         tokens_enc, input_ids, input_mask, segment_ids = get_features(sequence[0], sequence[1],sequence[2])
#         input_ids_list.append(input_ids)
#         input_mask_list.append(input_mask)
#         segment_ids_list.append(segment_ids)
        
#     all_input_ids = torch.tensor([input_ids for input_ids in input_ids_list], dtype=torch.long)
#     all_input_mask = torch.tensor([input_mask for input_mask in input_mask_list], dtype=torch.long)
#     all_segment_ids = torch.tensor([segment_ids for segment_ids in segment_ids_list], dtype=torch.long)

#     all_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

#     sampler = SequentialSampler(all_data)
#     dataloader = DataLoader(all_data, sampler=sampler, batch_size=batch_size)
    
#     for step, batch in enumerate(dataloader):
#         # print ("Getting predictions for Batch",step)
#         b_input_ids = batch[0]#.to(device)
#         b_input_mask = batch[1]#.to(device)
#         b_segment_id = batch[2]#.to(device)
#         outputs = model(b_input_ids, 
#                         token_type_ids=b_segment_id, 
#                         attention_mask=b_input_mask)
#         logits.append(outputs[0])
#     predictions = torch.cat(logits, dim=0)
#     predictions = predictions.detach().cpu().numpy()
#     return predictions 
    
# def get_experts(expertise_area, expert_db, num_experts = 50):
#   print('\nGetting experts for ',expertise_area)
#   experts = expert_db['Name'].unique()
#   l = [[]]
#   for expert in experts:
#     l = l + [[expert,'researches in',expertise_area]]
#   pred = get_predictions(l[1:],1)
#   m = torch.nn.Softmax(dim=1)
#   output = m(torch.tensor(pred))
#   output = output.detach().cpu().numpy()
#   neg,pos = output[:,0], output[:,1]
#   pos1 = pos[np.argsort(-pos)][:num_experts]
#   experts1 = experts[np.argsort(-pos)][:num_experts]
#   # out = experts1[pos1[0].tolist()]
#   # df = pd.DataFrame({'Name':experts1, 'Probability':pos1})
#   return experts1,pos1


# def main(MODEL_PATH=MODEL_PATH,EXPERT_SCHOOL=EXPERT_SCHOOL,NUM_EXPERTS=NUM_EXPERTS,QUERY=QUERY,OUTPUT_PATH=OUTPUT_PATH):

#     # get expert database
#     print ('\nReading Expert Database...')
#     if EXPERT_SCHOOL.lower() == 'scis':
#         expert_db = pd.concat([pd.read_csv('./Data/SIS_Faculty_Data.csv',index_col = False)])
#     elif EXPERT_SCHOOL.lower() == 'business':
#         expert_db = pd.concat([pd.read_csv('./Data/Business_Faculty_Data.csv',index_col = False)])
#     elif EXPERT_SCHOOL.lower() == 'all':
#         expert_db = pd.concat([pd.read_csv('./Data/SIS_Faculty_Data.csv',index_col = False),\
#                         pd.read_csv('./Data/Business_Faculty_Data.csv', index_col = False)])

#     # get experts and write to output path
#     experts,prob = get_experts(QUERY, expert_db, NUM_EXPERTS)
#     df = pd.DataFrame({'Name':experts, 'Probability':prob})
#     df['Query'] = QUERY
#     df = df[['Query','Name','Probability']]
#     print('\nWriting to output file...')
#     df.to_csv(OUTPUT_PATH,index=False)

# if __name__ == "__main__":
#     main(MODEL_PATH,EXPERT_SCHOOL,NUM_EXPERTS,QUERY,OUTPUT_PATH)