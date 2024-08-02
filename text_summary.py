import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text="Generating long and coherent text is an important but challenging task, particularly for open-ended language generation tasks such as story generation. Despite the success in modeling intra-sentence coherence, existing generation models (e.g., BART) still struggle to maintain a coherent event sequence throughout the generated text. We conjecture that this is because of the difficulty for the decoder to capture the high-level semantics and discourse structures in the context beyond token-level co-occurrence. In this paper, we propose a long text generation model, which can represent the prefix sentences at sentence level and discourse level in the decoding process. To this end, we propose two pretraining objectives to learn the representations by predicting inter-sentence semantic similarity and distinguishing between normal and shuffled sentence orders. Extensive experiments show that our model can generate more coherent texts than state-of-the-art baselines."
def summarizer(rawdocs):       
    stopwords=list(STOP_WORDS)
    #print(stopwords)   ==> used to print all stopwords

    nlp=spacy.load('en_core_web_sm')
    doc=nlp(rawdocs)
    #print(doc)  #used to print text 

    tokens= [token.text for token in doc]
    #print(tokens)  #each word of doc stored in a list called tokens

    word_freq={}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text]=1
            else:
                word_freq[word.text]+=1
    #print(word_freq)

    max_freq=max(word_freq.values())
    #print(max_freq)

    #now we find normalize frequency
    for word in word_freq.keys():
        word_freq[word]=word_freq[word]/max_freq #now this array contains our normalize frequency
        
    sent_tokens=[sent for sent in doc.sents]
    #print(sent_tokens)

    sent_scores={}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent]=word_freq[word.text]
                else:
                    sent_scores[sent]+=word_freq[word.text]

    #print(sent_scores)

    select_length=int(len(sent_tokens)*0.3)
    # print(select_length)

    summary=nlargest(select_length,sent_scores,key=sent_scores.get)
    # print(summary)

    final_summary=[word.text for word in summary]
    summary=' '.join(final_summary)
    
    return summary, doc , len(rawdocs.split(' ')) , len(summary.split(' '))