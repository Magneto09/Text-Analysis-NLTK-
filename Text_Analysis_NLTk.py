#!/usr/bin/env python
# coding: utf-8

# ## Text Analysis Operations using NLTK 

# In[3]:


#pip install NLTK


# In[4]:


import nltk


# In[5]:


#nltk.download()


# In[6]:


from nltk.tokenize import sent_tokenize


# In[ ]:


#Random Text taken


# In[9]:


text="""The register reference without the subscript refers to the logical register reference found in the instruction. The register reference with the subscript refers to a
hardware register allocated to hold a new value. When a new allocation is made for
a particular logical register, subsequent instruction references to that logical register
as a source operand are made to refer to the most recently allocated hardware register (recent in terms of the program sequence of instructions).
In this example, the creation of register R3c in instruction I3 avoids the RAW
dependency on the second instruction and the output dependency on the first instruction, and it does not interfere with the correct value being accessed by I4. The
result is that I3 can be issued immediately; without renaming, I3 cannot be issued
until the first instruction is complete and the second instruction is issued."""


# In[10]:


import nltk
nltk.download('punkt')


# Tokenization:
# 
# Sentence Tokenization

# In[11]:


tokenized_sent=sent_tokenize(text)


# In[13]:


print(tokenized_sent)


# Word Tokenization

# In[15]:


from nltk.tokenize import word_tokenize


# In[16]:


tokenized_word=word_tokenize(text)


# In[17]:


print(tokenized_word)


# Frequnecy Distirbution

# In[18]:


from nltk.probability import FreqDist


# In[19]:


Fdist=FreqDist(tokenized_word)


# In[20]:


print(Fdist)


# In[21]:


Fdist.most_common(2)


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


Fdist.plot(30,cumulative=False)


# Stop Words:
# Noise in the Text

# In[24]:


import nltk
nltk.download('stopwords')


# In[25]:


from nltk.corpus import stopwords


# In[26]:


stop_words=set(stopwords.words("english"))


# In[27]:


print(stop_words)


# Removing Stop-words

# In[28]:


filter_sent=[]


# In[29]:


for w in tokenized_word:
    if w not in stop_words:
        filter_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("filtered Sentence:",filter_sent)


# Stemming-
# Linguistic Normalization

# In[30]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,word_tokenize


# In[31]:


ps=PorterStemmer()


# In[32]:


stemmed_words=[]
for w in filter_sent:
    stemmed_words.append(ps.stem(w))


# In[33]:


print("Filtered Sentence:",filter_sent)


# In[34]:


print("Stemmed Sentence:",stemmed_words)


# Lemmatization-Reducing to the base words

# In[35]:


import nltk
nltk.download('wordnet')


# In[36]:


import nltk
nltk.download('omw-1.4')


# In[37]:


from nltk.stem.wordnet import WordNetLemmatizer
lem=WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
stem=PorterStemmer()

word="Beginner"


# In[38]:


print("Lemmatized Word:",lem.lemmatize(word,"v")) 
print("Stemmed word:",stem.stem(word))


# In[39]:


token=nltk.word_tokenize(text)


# In[40]:


print(token)


# In[41]:


nltk.pos_tag(token)


# In[ ]:




