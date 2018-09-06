#-*- coding: utf-8 -*-
from gensim import corpora
from collections import defaultdict
from pprint import pprint  # pretty-printer
from gensim import models

documents = ["Woodrow Wilson was first elected president in 1912. When he ran for re-election in 1916, one of the animating issues of that campaign was World War I, which was raging in Europe in which Wilson promised the United States would not join.In 1916, Wilson was reelected barely, but then in 1917, the U.S. declared war on Germany and threw itself into World War I, and by the time the midterm elections came around the year after that, in 1918, Wilson was not only facing the typical headwinds that a president's party usually faces in midterm elections he was also just roundly detested for World War I and for other stuff too.And so, in those midterm elections in 1918, Woodrow Wilson's party, the Democratic Party, got walloped. Republicans took over control of the Senate and they took over control of the House. And the Republicans get in Congress that had far-reaching consequences in all sorts of ways.But in the House, it meant a gigantic and very, very consequential promotion for this man. His name is Albert Johnson. Such a generic name, right? It's actually hard to Google.",
"There was another congressman named Albert Johnson from a different state who had nothing to do with them. There was a federal judge was called Albert Johnson. There was a famous Canadian fugitive called the Mad Trapper of Rat River who was called Albert Johnson.But this was Albert Johnson, a backbench congressman from Washington state whose whole public profile had been built around the defense of the white race and the threat that non-white immigrants posed to white civilization in the United States. Albert Johnson at home in Washington state, he ran a rabble-rousing anti-immigrant newspaper called the home defender. He bragged about having been part of mob violence that chased immigrants out of Washington state and out of the United States into Canada.And so, he had been a rabble-rouser and an orator on that pet issue for decades, but he's never really wielded power on the subject until Woodrow Wilson got shellacked in the 1918 midterm elections, and Albert Johnson's party, the Republican Party, took over leadership of the Congress. And that is what made it possible for Albert Johnson to take real power. He became chairman of the Committee on Immigration and Naturalization. It was his life's dream to be in charge of something like that and he did what he could with it.As soon as soon after he took over, the House of Representatives Committee on Immigration and Naturalization hired themselves an expert eugenics agent. Albert Johnson, the chairman, in addition to serving in Congress, he had become the president of the Eugenics Research Association of America. And once he was chairman of that committee, he brought on one of the officers from the Eugenics Research Association, this guy Harry Laughlin, to become an expert eugenics consultant to the immigration committee in Congress. And together, these two eugenicists – they got to work.In 1922, Harry Laughlin created this chart. Look, science, it's a chart. You can see diagonal there. That's the watermark of Truman State University. They've preserved this document online as part of their history of eugenics project."]

 # remove common words and tokenize
stoplist = set('for a of the and to in is that we if you not they'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
#print (texts)
 # remove words that appear only once

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]for text in texts]
#print (texts.sort_vocab())
# this saves it as a dictionary
dictionary = corpora.Dictionary(texts)
dictionary.save('./texts.dict') 

# saving it as a corpus 

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('./deerwester.mm', corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=3)
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=5)
doc = "Sean Hannity breaking news stories tonight. President Trump -- he continues to work towards keeping a very key campaign promise. He's now calling on Congress to work to pass a massive tax reform plan. That should help you, the American people, of course, after years of suffering"
comparison_doc = [[word for word in document.lower().split() if word not in stoplist] for document in doc]
comparison_str = ''.join(str(e) for e in comparison_doc)
#vec_bow = dictionary.doc2bow(doc.lower().split())
vec_bow = dictionary.doc2bow(comparison_str.split())
vec_lsi = lda[vec_bow]
print(vec_lsi)
