# -*- coding: utf-8 -*-
import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


raw_documents = ["I'm taking the show on the road.",
                 "My socks are a force multiplier.",
             "Woodrow Wilson was first elected president in 1912. When he ran for re-election in 1916, one of the animating issues of that campaign was World War I, which was raging in Europe in which Wilson promised the United States would not join.In 1916, Wilson was reelected barely, but then in 1917, the U.S. declared war on Germany and threw itself into World War I, and by the time the midterm elections came around the year after that, in 1918, Wilson was not only facing the typical headwinds that a president's party usually faces in midterm elections he was also just roundly detested for World War I and for other stuff too.And so, in those midterm elections in 1918, Woodrow Wilson's party, the Democratic Party, got walloped. Republicans took over control of the Senate and they took over control of the House. And the Republicans get in Congress that had far-reaching consequences in all sorts of ways.But in the House, it meant a gigantic and very, very consequential promotion for this man. His name is Albert Johnson. Such a generic name, right? It's actually hard to Google.",
"There was another congressman named Albert Johnson from a different state who had nothing to do with them. There was a federal judge was called Albert Johnson. There was a famous Canadian fugitive called the Mad Trapper of Rat River who was called Albert Johnson.But this was Albert Johnson, a backbench congressman from Washington state whose whole public profile had been built around the defense of the white race and the threat that non-white immigrants posed to white civilization in the United States. Albert Johnson at home in Washington state, he ran a rabble-rousing anti-immigrant newspaper called the home defender. He bragged about having been part of mob violence that chased immigrants out of Washington state and out of the United States into Canada.And so, he had been a rabble-rouser and an orator on that pet issue for decades, but he's never really wielded power on the subject until Woodrow Wilson got shellacked in the 1918 midterm elections, and Albert Johnson's party, the Republican Party, took over leadership of the Congress. And that is what made it possible for Albert Johnson to take real power. He became chairman of the Committee on Immigration and Naturalization. It was his life's dream to be in charge of something like that and he did what he could with it.As soon as soon after he took over, the House of Representatives Committee on Immigration and Naturalization hired themselves an expert eugenics agent. Albert Johnson, the chairman, in addition to serving in Congress, he had become the president of the Eugenics Research Association of America. And once he was chairman of that committee, he brought on one of the officers from the Eugenics Research Association, this guy Harry Laughlin, to become an expert eugenics consultant to the immigration committee in Congress. And together, these two eugenicists – they got to work.In 1922, Harry Laughlin created this chart. Look, science, it's a chart. You can see diagonal there. That's the watermark of Truman State University. Theyve preserved this document online as part of their history of eugenics project.",
            "I make my own fun."]
print("Number of documents:",len(raw_documents))


gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]
print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary[5])
print(dictionary.token2id['road'])
print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    print(i, dictionary[i])


corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
print(" the term frequency model is ", tf_idf)
s = 0
for i in corpus:
    s += len(i)
print(s)

sims = gensim.similarities.Similarity('/home/osboxes/Documents',tf_idf[corpus],num_features=len(dictionary))
print(sims)
print(type(sims))

query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)

sims[query_doc_tf_idf]
