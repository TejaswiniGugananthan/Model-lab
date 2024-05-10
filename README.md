```python

1)
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.dag import Edge,EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
pd.options.display.max_columns=50
df=pd.read_csv('weatherAUS.csv',encoding='utf-8')
df=df[pd.isnull(df['RainTomorrow'])==False]
df=df.fillna(df.mean())
df['WindGustSpeedCat']=df['WindGustSpeed'].apply(lambda x: '0.<=40'   if x<=40 else '1.40-50' if 40<x<=50 else '2.>50')
df['Humidity9amCat']=df['Humidity9am'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
df['Humidity3pmCat']=df['Humidity3pm'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
print(df)
def probs(data, child, parent1=None, parent2=None):
    if parent1==None:
        # Calculate probabilities
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent1!=None:
            # Check if child node has 1 parent or 2 parents
            if parent2==None:
                # Caclucate probabilities
           prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
          else:
       # Caclucate probabilities
     prob=pd.crosstab([data[parent1],data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    return prob
H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))
bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))
join_tree = InferenceController.apply(bbn)
pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "pink",
    "edgecolors": "blue",
    "edge_color": "green",
    "linewidths": 5,
    "width": 5,}
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()


2)
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
network=BayesianNetwork([('Burglary','Alarm'),('Earthquake','Alarm'),
                          ('Alarm','JohnCalls'),
                         ('Alarm','MarryCalls')])
cpd_burglary=TabularCPD(variable='Burglary',variable_card=2,values=[[0.999],[0.001]])
cpd_earthquake=TabularCPD(variable='Earthquake',variable_card=2,values=[[0.998],[0.002]])
cpd_alarm=TabularCPD(variable='Alarm',variable_card=2,values=[[0.999,0.71,0.06,0.05],[0.001,0.29,0.94,0.95]],evidence=['Burglary','Earthquake'],evidence_card=[2,2])
cpd_john_calls=TabularCPD(variable='JohnCalls',variable_card=2,values=[[0.95,0.1],[0.05,0.9]],evidence=['Alarm'],evidence_card=[2])
cpd_marry_calls=TabularCPD(variable='MarryCalls',variable_card=2,values=[[0.99,0.3],[0.01,0.7]],evidence=['Alarm'],evidence_card=[2])
network.add_cpds(cpd_burglary,cpd_earthquake,cpd_alarm,cpd_john_calls,cpd_marry_calls)
inference=VariableElimination(network)
evidence={'JohnCalls':1,'MarryCalls':0}
query_variable='Burglary'
result=inference.query(variables=[query_variable],evidence=evidence)
print(result)


3)
!pip install pgmpy
!pip install networkx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import GibbsSampling
import networkx as nx
import matplotlib.pyplot as plt
alarm_model = BayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)
# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_burglary = TabularCPD(
    variable="Burglary", variable_card=2, values=[[0.999], [0.001]]
)
cpd_earthquake = TabularCPD(
    variable="Earthquake", variable_card=2, values=[[0.998], [0.002]]
)
cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
)
cpd_johncalls = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.95, 0.1], [0.05, 0.9]],
    evidence=["Alarm"],
    evidence_card=[2],
)
cpd_marycalls = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.1, 0.7], [0.9, 0.3]],
    evidence=["Alarm"],
    evidence_card=[2],
)
# Associating the parameters with the model structure
alarm_model.add_cpds(
    cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls
print("Bayesian Network Structure")
print(alarm_model)
G=nx.DiGraph()
nodes=['Burglary','Earthquake','JohnCalls','MaryCalls']
edges=[('Burglary','Alarm'),('Earthquake','Alarm'),('Alarm','JohnCalls'),('Alarm','MaryCalls')]
G.add_nodes_from(nodes)
G.add_edges_from(edges)
pos={
    'Burglary':(0,0),
    'Earthquake':(2,0),
    'Alarm':(1,-2),
    'JohnCalls':(0,-4),
    'MaryCalls':(2,-4)
    }
nx.draw(G,pos,with_labels=True,node_size=1500,node_color="skyblue",font_size=10,font_weight="bold",arrowsize=20)
plt.title("Bayesian Network: Burglar Alarm Problem")
plt.show()
gibbssampler=GibbsSampling(alarm_model)
num_samples=10000
samples=gibbssampler.sample(size=num_samples)
query_variable="Burglary"
query_result=samples[query_variable].value_counts(normalize=True)
print("\n Approximate probabilities of {}:".format(query_variable))
print(query_result)



4)
import numpy as np
t_m = np.array([[0.7,0.3],
                [0.4, 0.6]])
e_m = np.array([[0.1,0.9],
                [0.8,0.2]])
i_p = np.array([0.5,0.5])

o_s = np.array([1,1,1,0,0,1])

alpha = np.zeros((len(o_s),len(i_p)))

alpha[0, :] = i_p * e_m[:, o_s[0]]

for t in range(1, len(o_s)):

  for j in range(len(i_p)):

    alpha[t, j ] = e_m[j,o_s[t]] * np.sum(alpha[t-1, :] * t_m[:, j ])

prob = np.sum(alpha[-1, :])

print("The Probability of the observed sequence is :",prob)

m_l_s = []
for t in range(len(o_s)):
  if alpha[t, 0] > alpha[t,1]:
    m_l_s.append("sunny")
  else:
    m_l_s.append("rainy")  

print("Thye most likely sequence of weather states is :",m_l_s)


5)
import numpy as np
class KalmanFilter:
    def _init_(self, F, H, Q, R, x0, P0):
        self.F = F 
        self.H = H 
        self.Q = Q 
        self.R = R 
        self.x = x0 
        self.P = P0 
    def predict(self):
      self.x = np.dot(self.F, self.x)
      self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    def update (self, z):
      y=z - np.dot(self.H, self.x)
      S = np.dot (np.dot(self.H, self.P), self.H.T)+ self.R
      K = np.dot (np.dot (self.P, self.H.T), np.linalg.inv(S))
      self.x = self.x + np.dot (K, y)
      self.P = np.dot (np.eye(self.F.shape[0]) -np.dot (K, self.H), self.P)
dt = 0.1
F = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])
Q = np.diag([0.1, 0.1])
R = np.array([[1]])
x0 = np.array([0, 0])
P0 = np.diag([1, 1])
kf=KalmanFilter(F,H,Q,R,x0,P0)
true_states = []
measurements = []
for i in range(100):
  true_states.append([i*dt, 1]) 
  measurements.append(i*dt +np.random.normal(scale=1))
est_states=[]
for z in measurements:
  kf.predict()
  kf.update (np.array([z]))
  est_states.append(kf.x)
import matplotlib.pyplot as plt
plt.plot([s[0] for s in true_states], label='true')
plt.plot([s[0] for s in est_states], label='estimate')
plt.legend()
plt.show()


6)
!pip install nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
f = open("/content/sample_data.txt", "r")
sentences = f.readlines()
f.close()
verbs = [[] for _ in sentences]
i=0
for sentence in sentences:
  print("Sentence",i+1,":", sentence)
  # Tokenize the sentence into words
  words = word_tokenize(sentence)
  # Identify the parts of speech for each word
  pos_tags = nltk.pos_tag(words)
  # Print the parts of speech
  for word,tag in pos_tags:
    print(word,"->",tag)
    # Save verbs
    if tag.startswith('VB'):
      verbs[i].append(word)
  i+=1
  print("\n\n") 
# Identify synonyms and antonyms for each word
print("Synonyms and Antonymns for verbs in each sentence:\n")
i=0
for sentence in sentences:
  print("Sentence",i+1,":", sentence)
  pos_tags = nltk.pos_tag(verbs[i])
  for word,tag in pos_tags:
    print(word,"->",tag)
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
      for lemma in syn.lemmas():
        synonyms.append(lemma.name())
        if lemma.antonyms():
          for antonym in lemma.antonyms():
            antonyms.append(antonym.name())
    # Print the synonyms and antonyms
    print("Synonyms:",set(synonyms))
    print("Antonyms:", set(antonyms) if antonyms else "None")
    print()
  print("\n\n")
  i+=1



  7)
!pip install nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
nltk.download( 'punkt' )
nltk.download( 'stopwords' )
def preprocess_text(text):
	# Tokenize the text into words
	words = word_tokenize(text)
	# Remove stopwords and punctuation
	stop_words= set(stopwords.words( 'english'))
	filtered_words= [word for word in words if word. lower() not in stop_words and word.isalnum()]
# Stemming
	stemmer = PorterStemmer()
stemmed_words= [stemmer. stem(word) for word in filtered_words]
return stemmed_words
def generate_summary(text,num_sentences=3):
sentences= sent_tokenize(text)
preprocessed_text = preprocess_text(text)
# Calculate the frequency of each word
word_frequencies =nltk. FreqDist (preprocessed_text)
# Calculate the score for each sentence based on word frequency
sentence_scores ={}
for sentence in sentences:
for word, freq in word_frequencies.items():
if word in sentence.lower():
if sentence not in sentence_scores:
sentence_scores[sentence] = freq
else:
sentence_scores[sentence]+= freq
# Select top N sentences with highest scores
summary_sentences= sorted(sentence_scores, key=sentence_scores.get,reverse=True) [ : num_sentences]
return ' '. join(summary_sentences)
if _name=="main_":
input_text ="""
Natural language processing (NLP) is a subfield of artificial intelligence.
It involves the development of algorithms and models that enact NLP.
NLP is used in various applications, including chatbots, language Understanding, and language generation.
This program demonstrates a simple text summarization using NLP"""
summary = generate_summary(input_text)
print("Origina1 Text: ")
print (input_text )
print( " \nSummary : " )
print(summary)


8)
pip install SpeechRecognition
pip install pyaudio
import speech_recognition as sr
r = sr.Recognizer()
duration = 5
print("Say Something")
with sr.Microphone() as source:
    audio_data = r.listen(source, timeout=duration)
try:
    text = r.recognize_google(audio_data)
    print('you said:', text)
except sr.UnknownValueError:
    print('Sorry,could not understand audio')
except sr.RequestError as e:
    print(f'Error with the request to Google Speech Recognition service: {e}')
except Exception as e:
    print(f'Error: {e}')


```
