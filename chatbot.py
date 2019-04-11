import numpy as np
import tensorflow as tf
import re
import time

# Dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')


# Dictinoary
id2line = {}

for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
# Conversations
conversations_ids = []

for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

# Qeustion and Answers
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# clean of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'won't", "will not", text)
    text = re.sub(r"\'can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text


# question
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# dictinoary maping nummer
word2count = {}

for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# dictinoary maping question dan answare to int
threshold = 20

questionwords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionwords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1

# token to dictionary
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']


for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# answare dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}

# add token all answer
for i in range(len(clean_answers)):
    clean_answers[i] += '<EOS>'

# transtalate question and answere into int

question_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    question_into_int.append(ints)


answer_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answer_into_int.append(ints)


# Sorting QnA by lengt question

sorted_clean_questions = []
sorted_clean_answers = []

for length in range (1, 25 + 1 ):
    for i in enumerate(question_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(question_into_int[i[0]])
            sorted_clean_answers.append(answer_into_int[i[0]])
            


# modeling

#Create input and target
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None],name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob

#Preprocessing the target
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocess_targets = tf.concat([left_side,right_side], 1)
    return preprocess_targets

# Encoder RNN layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rrn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rrn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rrn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dytpe = tf.float32)
    return encoder_state

# Decoding training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length,
                        decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,attention_option = 'bahdanau', num_units = decoder_cell.ouput_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_score_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_run_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# Test and validating
def decode_test_set(encoder_state, decoder_cell, decoder_embeddedings_matrix,sos_id, eos_id, maximum_length, num_words, sequence_length,
                        decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,attention_option = 'bahdanau', num_units = decoder_cell.ouput_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_score_function,
                                                                              decoder_embeddedings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_run_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)

    return test_predictions

# Creating decoder RNN

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rrn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rrn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rrn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.trucated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initalizer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      scope = decoding_scope,
                                                                      weights_initializers = weights,
                                                                      biases_initializer = biases)
        training_prediction = decode_training_set(encoder_state,
                                                  decoder_cell,
                                                  decoder_embedded_input,
                                                  sequence_length,
                                                  output_function,
                                                  keep_prob,
                                                  batch_size)
        decoding_scope.reuse_variable()
        test_predictions =  decode_test_set(encoder_state,
                                            decoder_cell,
                                            decoder_embeddings_matrix,
                                            word2int['<SOS>'],
                                            word2int['<EOS>'],
                                            sequence_length - 1,
                                            num_words,
                                            decoding_scope,
                                            output_function,
                                            keep_prob,
                                            batch_size)

    return training_prediction,test_predictions


