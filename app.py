import streamlit as st
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
import pandas as pd

from transformers import BertTokenizer,BertForSequenceClassification
import os
import gdown
import zipfile
import time

import urllib
from random import randint

from SessionState import _SessionState, _get_session, _get_state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(MODEL_PATH = 'sancharidan/scibet_expertfinder'):
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    return (model, tokenizer)

def main():
    state = _get_state()
    st.set_page_config(page_title="SMU Expert Finder", page_icon="🛸")

    model, tokenizer = load_model()
    # set_seed(42)  # for reproducibility

    load_page(state, model, tokenizer)

    state.sync()  # Mandatory to avoid rollbacks with widgets, must be called at the end of your app

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
     return experts1,pos1


def load_page(state: _SessionState, model, tokenizer):
#     disclaimer_short = """
#     __Disclaimer__: 
#     _This website is for entertainment purposes only!_
#     This website uses a machine learning model to produce fictional stories.
#     Even though certain bad words get censored, the model may still produce hurtful, vulgar, violent or discriminating text. 
#     Use at your own discretion.
#     View the information in the sidebar for more details.
#     """
#     disclaimer_long = """
#     __Description__:
#     This project uses a [pre-trained GPT2 model](https://huggingface.co/gpt2), which was fine-tuned on [Rick and Morty transcripts](https://rickandmorty.fandom.com/wiki/Category:Transcripts), to generate new stories in the form of a dialog. 
#     For a detailed explanation of GPT2 and its architecture see the [original paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), OpenAI’s [blog post](https://openai.com/blog/better-language-models/) or Jay Alammar’s [illustrated guide](http://jalammar.github.io/illustrated-gpt2/).
#     __Ethical considerations__:
#     The original GPT2 model was trained on WebText, which contains 45 million outbound links from Reddit (i.e. websites that comments reference).
#     While certain domains were removed, the model was trained on largely unfiltered content from the Internet, which contains biased and discriminating language.
    
#     __[Model Card](https://github.com/openai/gpt-2/blob/master/model_card.md) (by OpenAI)__:
#     "_Here are some secondary use cases we believe are likely:_
#     - _Writing assistance: Grammar assistance, autocompletion (for normal prose or code)_
#     - _Creative writing and art: exploring the generation of creative, fictional texts; aiding creation of poetry and other literary art._
#     - _Entertainment: Creation of games, chat bots, and amusing generations._
#     _Out-of-scope use cases:_
#     _Because large-scale language models like GPT-2 do not distinguish fact from fiction, 
#     we don’t support use-cases that require the generated text to be true. Additionally, 
#     language models like GPT-2 reflect the biases inherent to the systems they were trained on, 
#     so we do not recommend that they be deployed into systems that interact with humans unless 
#     the deployers first carry out a study of biases relevant to the intended use-case. We found 
#     no statistically significant difference in gender, race, and religious bias probes between 
#     774M and 1.5B, implying all versions of GPT-2 should be approached with similar levels of 
#     caution around use cases that are sensitive to biases around human attributes._"
#     __Tech stack__:
#     This website was built using [Streamlit](https://www.streamlit.io/) and uses the [Transformers](https://huggingface.co/transformers/) library to generate text.
#     """
#     st.markdown(disclaimer_short)
#     st.sidebar.markdown(disclaimer_long)

#     # st.write("---")

    st.title("SMU Expert Finder")

    state.input = st.text_input("Please enter research area for which you seek experts", key="topic_textbox")
    state.selectbox = st.selectbox('Please select School from which you wish to retrieve experts for above research area',\
     ('SCIS', 'Business', 'All'),index = 2, key = 'school_select')
    state.slider = st.slider('Please choose number of experts you wish to retrieve', 1, 10, key = 'num_experts_slider')

#     if len(state.input) + state.slider > 5000:
#         st.warning("Your story cannot be longer than 5000 characters!")
#         st.stop()

    button_generate = st.button("Find Experts")
#     if st.button("Reset Prompt"):
#         state.clear()

    if button_generate:
        try:
          QUERY = state.input
          EXPERT_SCHOOL = state.selectbox
          NUM_EXPERTS = state.slider
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
          df['Query'] = QUERY
          df = df[['Query','Name','Probability']]
          print('\nWriting to output file...')
          df.to_csv('./Output/results.csv',index=False)
          st.write('Displaying top {} experts in the field of {} from {} school'.format(NUM_EXPERTS,QUERY.upper(),EXPERT_SCHOOL.upper()))
          print('\nStep 1...')

          # df = pd.read_csv('./Output/results.csv',index_col=False)
          st.dataframe(df)
          print('\nStep 2...')
          # outputs = model(
          #       state.input,
          #       do_sample=True,
          #       max_length=len(state.input) + state.slider,
          #       top_k=50,
          #       top_p=0.95,
          #       num_return_sequences=1,
          #   )
          # output_text = filter_bad_words(outputs[0]["generated_text"])
          # state.input = st.text_area(
          #      "Start your story:", output_text or "", height=50
          # )
        except:
            pass

#     st.markdown(
#         '<h2 style="font-family:Courier;text-align:center;">Your Story</h2>',
#         unsafe_allow_html=True,
#     )

#     for i, line in enumerate(state.input.split("\n")):
#         if ":" in line:
#             speaker, speech = line.split(":")

#             st.markdown(
#                 f'<p style="font-family:Courier;text-align:center;"><b>{speaker}:</b><br>{speech}</br></p>',
#                 unsafe_allow_html=True,
#             )
#         else:
#             st.markdown(
#                 f'<p style="font-family:Courier;text-align:center;">{line}</p>',
#                 unsafe_allow_html=True,
#             )
    
#     st.markdown("---")
#     st.markdown(
#         "_You can read about how to create your own story generator application [here](https://towardsdatascience.com/rick-and-morty-story-generation-with-gpt2-using-transformers-and-streamlit-in-57-lines-of-code-8f81a8f92692). The code for this project is on [Github](https://github.com/e-tony/Story_Generator)._"
#     )


if __name__ == "__main__":
    main()

# # App title
# download_model()
# import retrieve_experts


# if query:
#     retrieve_experts.main(QUERY = query, EXPERT_SCHOOL = expert_school, NUM_EXPERTS = num_experts)
    