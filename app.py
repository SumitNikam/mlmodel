from flask import Flask, request, jsonify
from urllib.request import urlopen
import json
import re
import nltk
import nltk.corpus
from nltk.corpus import nps_chat
from nltk.tokenize import TweetTokenizer
from datetime import datetime
from collections import OrderedDict
import numpy as np
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
app = Flask(__name__)
d = 0.85
min_diff = 1e-5
steps = 10
node_weight = None
stopwords = set(STOP_WORDS) 

@app.route('/summary', methods=['POST'])
def text_summary():
    
    # Transcription
    def trans(data):
        transcription = []
        for i in range(0,len(data['monologues'])):
            for j in range(0,len(data['monologues'][i]['elements'])):
                transcription.append(data['monologues'][i]['elements'][j]['value'])
        transcription = ''.join(transcription)
        return transcription
    
    # Keywords and Summary
    def set_stopwords(stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
        # Get Symmeric matrix
        g = symmetrize(g)
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)
        return g_norm

    def analyze(text, candidate_pos=['NOUN', 'PROPN', 'VERB'], window_size=4, lower=False, stopwords=list(), number=10):
        # Set stop words
        set_stopwords(stopwords)
        # Pare text by spaCy
        doc = nlp(text)
        # Filter sentences
        sentences = sentence_segment(doc, candidate_pos, lower)
        # Build vocabulary
        vocab = get_vocab(sentences)
        # Get token_pairs from windows
        token_pairs = get_token_pairs(window_size, sentences)
        # Get normalized matrix
        g = get_matrix(vocab, token_pairs)
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        # Iteration
        previous_pr = 0
        for epoch in range(steps):
            pr = (1-d) + d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        node_weight = node_weight
        node_weight = OrderedDict(
            sorted(node_weight.items(), key=lambda t: t[1], reverse=True))
        keyword = []
        for i, (key, value) in enumerate(node_weight.items()):
            keyword.append(key)
            if i > number:
                break
        return keyword
    
    # Pricing
    def command_detected(sentence):
        # Detects whether a given String sentence is a command or action-item
        tagged_sentence = pos_tag(word_tokenize(sentence));
        first_word = tagged_sentence[0];
        pos_first = first_word[1];
        first_word = first_word[0].lower()
        for word in command_words:
            if word in sentence:
                return True
        return False

    def retrieve_action_items(data):
        # Returns a list of the sentences containing action items.
        action_items = []
        for sentence in sent_tokenize(trans(data)):
            possible_command = command_detected(str(sentence))
            if possible_command is True:
                action_items += [(str(sentence))]
        return action_items

    def pricing_time(data):
        ts = []
        value = []
        for i in range(len(data['monologues'])):
            for j in range(len(data['monologues'][i]['elements'])):
                if data['monologues'][i]['elements'][j]['value'] in ["price", "pricing","cost"]:
                    value.append(data['monologues'][i]['elements'][j]['value'])
                    ts.append(data['monologues'][i]['elements'][j]['ts'])
        return ts
    
    # Load data
    data = request.json
    #json_url = urlopen(url)
    #data = json.loads(json_url.read()) 
    
    # transcript
    source = trans(data)
    source= re.sub(r'Welcome to ring central,','',source)
    source= re.sub(r'I will be recording this meeting and summarizing it for you. Thanks.','',source)
    
    # Keywords
    Keywords = analyze(source, candidate_pos=['NOUN', 'PROPN', 'VERB'], window_size=4, lower=False)
    
    # Summary
    tokenized_transcript = sent_tokenize(source)
    LANGUAGE = "english"
    parser = PlaintextParser.from_string(source, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(parser.document, len(tokenized_transcript)*0.01)
    transcript_summary = []
    for sentence in summary:
        transcript_summary.append(str(sentence))
    
    if transcript_summary==[]:
        transcript_summary=source
    
    # Pricing and time discussion
    command_words = ["price", "pricing", "cost"]
    pricing = retrieve_action_items(data)
    pricing_time = pricing_time(data)
    
    # Time and sentiment also speaking time its use for speaking ratio
    number = {}
    speakers = {}
    value = {}
    start_time = {}
    end_time = {}
    Time={}
    total={}
    pol ={}
    polarity_transcription = []
    speaking_time =[]

    # Identify Number of speakers    
    for i in range(0, len(data['monologues'])):
        if data['monologues'][i]['speaker'] in number:
            number[data['monologues'][i]['speaker']] = number[data['monologues'][i]['speaker']] + 1
        else:
            number[data['monologues'][i]['speaker']] = 1

    # Finding Times
    for i in range(0,len(number) ):
        value[str(i)] = []
        start_time[str(i)] =[]
        end_time[str(i)] = []

    # Start Time of Speakers
    for i in range(0,len(data['monologues'])):
        for k in list(start_time.keys()):
            if data['monologues'][i]['speaker'] == int(k):
                for j in range(0,len(data['monologues'][i]['elements'])):
                    if j==0:
                        start_time[list(start_time.keys())[int(k)]].append(data['monologues'][i]['elements'][j]['ts'])        
    # End Time of Speakers
    for i in range(0,len(data['monologues'])):
        for k in list(value.keys()):
            if data['monologues'][i]['speaker'] == int(k):
                for j in range(0,len(data['monologues'][i]['elements'])):
                    value[list(value.keys())[int(k)]].append(data['monologues'][i]['elements'][j]['value'])
                try:
                    end_time[list(value.keys())[int(k)]].append(data['monologues'][i]['elements'][j-2]['end_ts'])
                except:
                    end_time[list(value.keys())[int(k)]].append(data['monologues'][i]['elements'][j-1]['end_ts'])
                    
    # Total speaking time
    for i in range(len(number)):
        total['total'+str(i)] = 0
        for j in range(len(start_time[str(i)])):
            total['total'+str(i)] = total['total'+str(i)] + (end_time[str(i)][j]-start_time[str(i)][j])

    # Overall sentiment        
    for i in sent_tokenize(trans(data)):
        polarity_transcription.append(TextBlob(i).sentiment.polarity)
        
    # Speaker by speaker seperation   
    for i in list(value.keys()):
        speakers["Speakers "+str(i)] = "".join(value[str(i)])
    
    # Speaker by speaker sentiment  
    for i in speakers:
        pol[i]=[]
        for j in sent_tokenize(speakers[i]):
            pol[i].append(TextBlob(j).sentiment.polarity)
            
    # Result
    result = {"keywords": Keywords, 'summary': transcript_summary, 'pricing': pricing, 'pricing_time':pricing_time, 'total_time' : total, "overall_sentiment": polarity_transcription, "sentiment": pol}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

