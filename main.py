import nltk
import os
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import gensim
from nltk.corpus import stopwords


#Cette fonction permet d'ajouter les racines au fichier d'entree
def add_stems(corpus, name):
    ps = PorterStemmer()
    # ps = LancasterStemmer()
    # ps = SnowballStemmer("english")
    new_corpus = open(name, "wb")
    for line in corpus:
        if(line!="\n"):
            line = line.split('\t')
            word = line[0]
            file = line[1]
            classe = line[2]
            stem = ps.stem(word)
            new_corpus.write(word+"\t"+file+"\t"+stem+"\t"+classe)
        else:
            new_corpus.write("\n")
    new_corpus.close()


#Cette fonction permet d'aller des etiquettes de treebank aux etiquettes de wordnet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


#Cette fonction permet d'ajouter les lemmes au fichier d'entree
def add_lems(corpus, name):
    wl = WordNetLemmatizer()
    new_corpus = open(name, "wb")
    for line in corpus:
        if(line!="\n"):
            line = line.split('\t')
            word = line[0]
            file = line[1]
            stem = line[2]
            tag = line[3]
            classe = line[4]
            lem=wl.lemmatize(word,get_wordnet_pos(tag))
            new_corpus.write(word+"\t"+file+"\t"+stem+"\t"+tag+"\t"+lem+"\t"+classe)
        else:
            new_corpus.write("\n")
    new_corpus.close()

#Cette fonction permet de retourner toutes les phrases du fichier d'entree
def read_sentences(corpus):
    sentence=[]
    sentences=[]
    for line in corpus:
        if(line!="\n"):
            word = line.split("\t")[0]
            sentence.append(word)
        else:
            sentences.append(sentence)
            sentence=[]
    sentences.append(sentence)
    return sentences


#Cette fonction permet d'ajouter les embeddings des mots au fichier d'entree
def add_embeddings(corpus, name):
    new_corpus = open(name, "wb")
    corpus_sentences=read_sentences(corpus)
    model = gensim.models.Word2Vec(corpus_sentences, min_count=1, size=7)
    for line in corpus:
        if(line!="\n"):
            line = line.split('\t')
            word = line[0]
            file = line[1]
            stem=line[2]
            tag=line[3]
            classe = line[4]
            embedding = "\t".join([str(value) for value in model[word]])
            new_corpus.write(word + "\t" + file + "\t" + stem + "\t" +tag+ "\t"+ embedding + "\t" + classe)
        else:
            new_corpus.write("\n")
    new_corpus.close()

#Cette fonction permet d'ajouter les etiquettes morpho-syntaxiques au fichier d'entree
def add_tags(corpus, name):
    cpt_lignes=0
    new_corpus = open(name, "wb")
    sentence=[]
    files=[]
    stems=[]
    classes=[]
    for line in corpus:
        if(line!="\n"):
            line=line.split('\t')
            sentence.append(line[0])
            files.append(line[1])
            stems.append(line[2])
            classes.append(line[3])
        else:
            tagged_sentence=nltk.pos_tag(sentence)
            assert(len(sentence)==len(stems)==len(classes)==len(files)==len(tagged_sentence))
            for (word,file,stem,classe,tagged_word) in zip(sentence,files,stems,classes,tagged_sentence):
                new_corpus.write(word + "\t" + file + "\t" + stem + "\t" + tagged_word[1] + "\t" + classe)
                cpt_lignes+=1
            new_corpus.write("\n")
            cpt_lignes+=1
            sentence = []
            files = []
            stems = []
            classes = []
    tagged_sentence = nltk.pos_tag(sentence)
    assert (len(sentence) == len(stems) == len(classes) == len(files) == len(tagged_sentence))
    for (word, file, stem, classe, tagged_word) in zip(sentence, files, stems, classes, tagged_sentence):
        new_corpus.write(word + "\t" + file + "\t" + stem + "\t" + tagged_word[1] + "\t" + classe)
        cpt_lignes += 1
    new_corpus.close()

#Cette fonction permet de retourner le nom d'une variable a partir de sa reference
def from_var_to_string(var):
    import itertools
    return [tpl[0] for tpl in itertools.ifilter(lambda x: var is x[1], globals().items())]

#Cette fonction permet d'afficher les statistiques sur la distribution des classes dans le fichier d'entree
def print_statistics(corpus):
    corpus_name=from_var_to_string(corpus)
    classes_distribution= nltk.FreqDist()
    for line in corpus:
        if(line!="\n"):
            line =line.encode('utf_8').split('\t')
            classe=line[4].rstrip()
            classes_distribution[classe]+= 1

    print("Classes distribution for "+corpus_name[0])
    total = sum([classes_distribution[key] for key in classes_distribution.keys()])
    for key in classes_distribution.keys():
        value = classes_distribution[key]
        print(key + " -> " + str(value) + " = " + str(float(value) / total * 100)[:5]+"%")

#Cette fonction permet de remplacer les classes qui ne nous interessent pas par la classe O
def eliminate_useless_classes(corpus,name):
    new_corpus = open(name, "wb")
    for line in corpus:
        if (line != "\n"):
            line = line.split('\t')
            word = line[0]
            file = line[1]
            stem = line[2]
            tag = line[3]
            classe = line[4]
            if(classe!="B-AdverseReaction\n" and classe!="I-AdverseReaction\n"):
                classe="O\n"
            new_corpus.write(word + "\t" + file + "\t" + stem + "\t" + tag + "\t" + classe)
        else:
            new_corpus.write("\n")

#Cette fonction permet de verifier si un mot est dans la liste stopwords ou non
def accepted(word, stop_words):
    if word not in stop_words:
        return True
    return False

#Cette fonction permet de supprimer les mots vides
def eliminate_stop_words(corpus, name):
    stop_words = set(stopwords.words())
    punc = [".", ",", "?", "!", ":", "#", "(", ")", "[", "]", ";"]
    new_corpus = open(name, "wb")
    for line in corpus:
        if (line != "\n"):
            line = line.split('\t')
            word=line[0]
            file=line[1]
            stem=line[2]
            tag=line[3]
            classe=line[4]
            if (accepted(word, stop_words)):
                new_corpus.write(word + "\t" + file + "\t" + stem + "\t" + tag + "\t" + classe)
        else:
            new_corpus.write("\n")
    new_corpus.close()


#Cette fonction permet de donner tous les tags utilises dans le fichier d'entree
def extract_tags(corpus):
    my_dico=dict()
    for line in corpus:
        if (line != "\n"):
            line = line.split('\t')
            word=line[0]
            tag=line[3]
            my_dico[tag]=word

    for key in my_dico.keys():
        print(key+" -> "+my_dico[key])


#Cette fonction permet de faire l'apprentissage et le developpement
def learn_dev(pattern):
    print("Learning:")
    os.system("wapiti train -p patterns/"+pattern+".txt corpus/corpus.train results/features")
    print("Developing:")
    os.system("wapiti label -c -m results/features corpus/corpus.dev results/output_dev")

#Cette fonction permet de faire la validation
def final_test():
    print("testing:")
    os.system("wapiti label -c -m results/features corpus/corpus.test results/output_test")


def learn_dev1(pattern):
    print("Learning:")
    os.system("wapiti train -p patterns/"+pattern+".txt corpus/corpus1.train results/features")
    print("Developing")
    os.system("wapiti label -c -m results/features corpus/corpus1.dev results/output_dev")

def final_test1():
    print("testing:")
    os.system("wapiti label -c -m results/features corpus/corpus1.test results/output_test")

#Cette fonction permet de regrouper les deux classes B-adv et I-adv en une seule classe
def unify_adverse_reaction_classes(corpus, name):
    new_corpus = open(name, "wb")
    for line in corpus:
        if (line != "\n"):
            line = line.split('\t')
            word=line[0]
            file=line[1]
            stem=line[2]
            tag=line[3]
            classe=line[4]
            if(classe=="B-AdverseReaction\n" or classe=="I-AdverseReaction\n"):
                classe="AdverseReaction\n"
            new_corpus.write(word + "\t" + file + "\t" + stem + "\t" + tag + "\t" + classe)
        else:
            new_corpus.write("\n")
    new_corpus.close()

# ############MAIN#############################################
#ETAPE0: Hypothese de base
#learn_dev("pattern1")

#ETAPE1: Bigrammes de classes
# learn_dev("pattern2")

#ETAPE2: Racinisation
dev=open("corpus/corpus0.dev","rb").readlines()
train=open("corpus/corpus0.train","rb").readlines()
test=open("corpus/corpus0.test","rb").readlines()
add_stems(train,"corpus/corpus.train")
add_stems(test,"corpus/corpus.test")
add_stems(dev,"corpus/corpus.dev")
# learn_dev("pattern3")

#ETAPE3: Etiquetage morpho-syntaxique
dev=open("corpus/corpus.dev","rb").readlines()
train=open("corpus/corpus.train","rb").readlines()
test=open("corpus/corpus.test","rb").readlines()
add_tags(train,"corpus/corpus.train")
add_tags(test,"corpus/corpus.test")
add_tags(dev,"corpus/corpus.dev")
# learn_dev("pattern5")

#ETAPE4: Ne pas tenir compte de la casse
#learn_dev("pattern7")

#XXXXXX ETAPE5: Lemmatisation
# dev=open("corpus/corpus.dev","rb").readlines()
# train=open("corpus/corpus.train","rb").readlines()
# test=open("corpus/corpus.test","rb").readlines()
# add_lems(train,"corpus/corpus.train")
# add_lems(test,"corpus/corpus.test")
# add_lems(dev,"corpus/corpus.dev")
# learn_dev("pattern8")

#XXXXXX ETAPE6: Representation vectorielle
# dev=open("corpus/corpus.dev","rb").readlines()
# train=open("corpus/corpus.train","rb").readlines()
# test=open("corpus/corpus.test","rb").readlines()
# add_embeddings(train,"corpus/corpus.train")
# add_embeddings(test,"corpus/corpus.test")
# add_embeddings(dev,"corpus/corpus.dev")
# learn_dev("pattern9")

#ETAPE7: Eliminer les classes inutiles
dev=open("corpus/corpus.dev","rb").readlines()
train=open("corpus/corpus.train","rb").readlines()
test=open("corpus/corpus.test","rb").readlines()
eliminate_useless_classes(train, "corpus/corpus.train")
eliminate_useless_classes(test, "corpus/corpus.test")
eliminate_useless_classes(dev,"corpus/corpus.dev")
learn_dev("pattern7")

#XXXXXXXXXX ETAPE8: Eliminer les mots vides
# train=open("corpus/corpus.train","rb").readlines()
# dev=open("corpus/corpus.dev","rb").readlines()
# test=open("corpus/corpus.test","rb").readlines()
# eliminate_stop_words(train,"corpus/corpus.train")
# eliminate_stop_words(test,"corpus/corpus.test")
# eliminate_stop_words(dev,"corpus/corpus.dev")
# learn_dev("pattern7")


# ETAPE Z : Et si on veut savoir seulement si la classe est adverse reaction ou non?
# train=open("corpus/corpus.train","rb").readlines()
# dev=open("corpus/corpus.dev","rb").readlines()
# test=open("corpus/corpus.test","rb").readlines()
# unify_adverse_reaction_classes(train,"corpus/corpus1.train")
# unify_adverse_reaction_classes(test,"corpus/corpus1.test")
# unify_adverse_reaction_classes(dev,"corpus/corpus1.dev")
# train=open("corpus/corpus.train","rb").readlines()
# dev=open("corpus/corpus.dev","rb").readlines()
# test=open("corpus/corpus.test","rb").readlines()
# learning_testing1("pattern7")


#Testing:
final_test()