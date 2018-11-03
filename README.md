<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [POS Tagger](#pos-tagger)
  - [To Do](#to-do)
  - [Process](#process)
  - [Parts of Speech (POS)](#parts-of-speech-pos)
    - [Tag Set](#tag-set)
    - [POS Basics](#pos-basics)
  - [POS Taggers](#pos-taggers)
    - [`TreeTagger` and `treetaggerwrapper`](#treetagger-and-treetaggerwrapper)
    - [Stanford POS Tagger and NLTK](#stanford-pos-tagger-and-nltk)
  - [References](#references)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# POS Tagger

Refer: https://en.wikipedia.org/wiki/Part-of-speech_tagging

## To Do

- Learn vector representation of grammar of any given language (starting with English).
- Use the vector representation to compare two languages and suggest on how to learn grammar of languages other than English.

## Process

- Train your model with sentences tagged with parts of speech (POS). Each POS is represented by a m-dimensional one-hot vector where `m` is number of possible POSs in a given language. A sentence is then represented by a series of POS logits vectors.
- The training forces the RNN to learn the sequential structure of the sentence.
- Predict POS on new sentences.

## Parts of Speech (POS)

Parts of speech are the basic units of grammar making them a natural choice for understanding language.

Since grammar consist of a series of relationships between POS, it follows that the vector space (POS logits vectors) will reflect the grammar of the language.

By comparing the vector space of two languages after training, it is possible to uncover grammatical differences.

After training the model, we visualize the vector space using [t-Distributed Stochastic Neighbor Embedding (t-SNE)](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

Stemming and Lemmatization:

Both of them reduces words to a "base" form. But, stem is more crude from. For example, "wolv" is stem, whereas "wolf" is lemma.

This base form is useful because vocabulary can get too large easily.

Snonymy: Multiple words with the same meaning.

Polysemy: One word with multiple meanings.

### Tag Set

The most popular tag set is [Penn Treebank tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html). Most of the already trained taggers for English are trained on this tag set. Examples of such taggers are:
- NLTK default tagger
- Stanford CoreNLP tagger

Resources for building POS taggers are pretty scarce, simply because annotating a huge amount of text is very tedious task. One resource that is in our reach and that uses our preferred tag set can be found inside NLTK.

```py
import nltk

tagged_sentences = nltk.corpus.treebank.tagged_sents()
```

### POS Basics

Words can function as differnt POS depending on the sentences they are in.

- Noun: Identifies somebody or something.
- Pronoun: Used in place of noun to avoid repeating it.
- Adjective: Describes characteristics of a noun or pronoun.
- Verb: Action word.
- Adverb: Describes the way in which an action is done.
- Preposition: Relation between noun (or pronoun) and another word
- Conjunction: Joins clauses
- Interjection: Remark expressing emotion

## POS Taggers

Overview of available Taggers: https://nlp.stanford.edu/links/statnlp.html#Taggers

***Comparison References:***
- http://www.statmt.org/moses/?n=Moses.ExternalTools
- https://mattwilkens.com/2008/11/08/evaluating-pos-taggers-speed/

***Examples:***
- https://medium.com/wenqins-blog/improve-your-french-by-teaching-a-neural-network-3cf219ed0850
- https://github.com/wenqinYe/language_analysis/blob/master/lstm.ipynb
- https://nlpforhackers.io/training-pos-tagger/

### `TreeTagger` and `treetaggerwrapper`

We can use [TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) tool for annotating text with POS and lemma information. It was developed by Helmut Schmid at University of Stuttgart. To make use of it, we can use its Python wrapper [treetaggerwrapper](https://treetaggerwrapper.readthedocs.io/en/latest/).

Download following files from TreeTagger:

```sh
tree-tagger-MacOSX-3.2.tar.gz
tagger-scripts.tar.gz
install-tagger.sh
english-bnc.par.gz
english.par.gz
english-chunker.par.gz
```

Move them to `"~/MG/projects/tree_tagger"`.

Run `. ./install-tagger.sh`.

Then, add path to `bin` and `cmd` to `$PATH`. For example,

```sh
# File: ~/.zshrc
export PATH="$PATH:/Users/mugoyal/MG/projects/tree_tagger/cmd"
export PATH="$PATH:/Users/mugoyal/MG/projects/tree_tagger/bin"
```

Now, you can test your installation as:

```sh
source ~/.zshrc
echo 'Hello world!' | tree-tagger-english
```

Then,

```py
from pprint import pprint
import treetaggerwrapper as ttw

tagger = ttw.TreeTagger(TAGLANG='en', TAGDIR="MG/projects/tree_tagger")

tags = tagger.tag_text("Kids are playing in nearby garden.")

pprint(tags)
# ['Kids\tNN2\tkid',
#  'are\tVBB\tbe',
#  'playing\tVVG\tplay|playing',
#  'in\tPRP\tin',
#  'nearby\tAJ0\tnearby',
#  'garden\tNN1\tgarden',
#  '.\tSENT\t.']

tags2 = ttw.make_tags(tags)

pprint(tags2)
# [Tag(word='Kids', pos='NN2', lemma='kid'),
#  Tag(word='are', pos='VBB', lemma='be'),
#  Tag(word='playing', pos='VVG', lemma='play|playing'),
#  Tag(word='in', pos='PRP', lemma='in'),
#  Tag(word='nearby', pos='AJ0', lemma='nearby'),
#  Tag(word='garden', pos='NN1', lemma='garden'),
#  Tag(word='.', pos='SENT', lemma='.')]
```

In order to make above work, I had to make a copy of `english.par` to `english-utf8.par`.

You can also directly process files using `TreeTagger.tag_file()` and `TreeTagger.tag_file_to()` methods.

### Stanford POS Tagger and NLTK

Another tool is to use [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml) along with [Natural Language Toolkit (NLTK)](http://www.nltk.org/) as Python interface to Stanford POS tagger. Its basic download contains two trained tagger models for English. The full downloader contains trained English tagger models, an Arabic tagger model, a Chinese tagger model, a French tagger model, and a German tagger model. Both versions include the same source and other required files. The tagger can be retrained on any language, given POS-annotated training text for the language.

Refer: [NLTK Book](http://www.nltk.org/book/)


```py
import nltk
nltk.download()
```

Specify the download directory as `/Users/mugoyal/MG/projects/nltk_data`. If you did not install the data to:

```
Mac: # /usr/local/share/nltk_data
Unix # /usr/share/nltk_data
```

then you'll nead to set the `NLTK_DATA` environment variable to specify the location of data.

Test that the data has been installed:

```py
from nltk.corpus import brown

brown.words()         #=> ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
```

Now, you can use it as:

```py
import nltk

sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
tokens
# ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning',
# 'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']
tagged = nltk.pos_tag(tokens)
tagged[0:6]
# [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'), ('on', 'IN'),
# ('Thursday', 'NNP'), ('morning', 'NN')]
```

But it is recommended that from 2017 on, we should not use the original classes but rather the much better [`CoreNLPPOSTTagger`](http://www.nltk.org/_modules/nltk/tag/stanford.html#CoreNLPPOSTagger) class!

## References

- English to French Translation: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
- English to French Translation: https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/





