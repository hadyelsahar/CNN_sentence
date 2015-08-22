"""
this script is about data processing for the Conv. Neural network for large data
it takes training set and returns csv files (with the number of CV  default 10 )
in the form of docvector, id  , where docvector is a list of indices.

the script also saves a mr.p file that contains word_idx_map

todo : find a way to skip padding for example using sparse matrix instead
todo : adapt script to work with unlabeled data
"""
import os
import re
from collections import defaultdict
import argparse
import cPickle
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold, train_test_split

##########################
# COMMAND LINE ARGUMENTS #
##########################
parser = argparse.ArgumentParser(description='script create sub sample of datasets')
parser.add_argument('-i', '--input', help='training file path', required=True)
parser.add_argument('-f', '--data_folder', help='data folder', required=True)
parser.add_argument('-w2v', '--w2v', help='word 2 vec binary file', required=True)
parser.add_argument('-percentage', '--percentage', help=" percentage of training set to experiment on", required=False)
parser.add_argument('-uniq', '--unique', help="unique words or all from each file", action='store_true', required=False)
parser.add_argument('-idxfname', '--idxfolder_name', help="optional custom folder path to save all idx files created", required=False)

args = parser.parse_args()
if not hasattr(args, "percentage"):
    args.percentage = 1
else:
    args.percentage = float(args.percentage)

if not hasattr(args, "idxfolder_name"):
    args.idxfolder_name = "./idx_data/"
elif args.idxfolder_name[-1] is not "/" :  # additional of extra / if path doesn't have one
    args.idxfolder_name = args.idxfolder_name + "/"

####################
# HELPER FUNCTIONS #
####################

def get_ids(filename, take_percentage=1):
    """
    function take training file name and return dataframe of ids,label to with percentage to train with

    :param filename: input csv file name containing all training data in form of (file, label)
    :param take_percentage: percentage of data to train with default 1 (eg. use 0.0001 for debugging)
    :return:
    """

    df = pd.read_csv(filename, names=["file", "label"], encoding="utf-8", header=0)

    # to keep both labels in the same percentage as training set even after taking portion of data
    labels = df.label.unique()  # collecting unique labels in case file is labeled in different way than 0,1
    d1 = df[df.label == labels[0]]
    d1 = d1[:int(d1.shape[0]*take_percentage)]   # limiting the first class to % of it
    d2 = df[df.label == labels[1]]
    d2 = d2[:int(d2.shape[0]*take_percentage)]   # limiting the first class to % of it
    df = pd.concat([d1, d2])

    return df

def read_data_files(filenames, datapath, ids=None):
    """
    function to read data files from folder and return as iterator in order not to load on data

    :param filename: list or iterator of file names
    :param datapath: datapath to concatinate before each filename
    :return: iterator of text containing all text in all files in order
    """
    if ids is None:
        ids = range(0, len(filenames))

    for i in [filenames[k] for k in ids]:
        yield str(open(datapath+i, 'r').read())



#############
# FUNCTIONS #
#############

def create_vocab(df, datapath):
    """
    create vocabulary and maximum length of sentence out of the dataset if not already existing
    and save vocabulary hash in vocab_max_l.p pickle file

    :return: set of all unique words in the used dataset
    """
    if os.path.isfile("vocab_max_l.p"):
        o = pickle.load( open( "vocab_max_l.p", "rb" ))  # search if vocab file is already existing
        vocab = o[0]
        max_l = o[1]
    else:
        vocab = defaultdict(int)
        max_l = 0
        for d in read_data_files(df.files, datapath):
            words = clean_str(d).split(" ")
            if len(words) > max_l:
                max_l = len(words)

            for w in words:
                vocab[w] += 1

        cPickle.dump([vocab, max_l], open("vocab_max_l.p", "wb"))
    return  vocab, max_l

def build_data_cv(df, datapath, n_folds=5):
    """
    function creating CV cross folds

    :param df: pandas dataframe containing training examples in form of {file, label}
    :param datapath: data path that has all training files
    :param n_folds: number of folds in case of cross validation
    :return: array of tuples in form of (id_train, X_train, y_train, id_test, X_test, y_test) where
    id is the file name (document id), X is the document text, and y is the class label
    """

    y = np.array(df.label)

    kf = StratifiedKFold(y, n_folds=n_folds, shuffle=False, random_state=2)

    kfolds_data = []

    for trainids, testids in kf:

        kfolds_data.append({
            "id_train": [i for i in list(df.file) if i in trainids]
            "x_train": read_data_files(list(df.file), datapath, trainids),
            "y_train": y[trainids],
            "id_test": [i for i in list(df.file) if i in testids]
            "x_test": read_data_files(list(df.file), datapath, testids),
            "y_test": y[testids]
        })

    return kfolds_data

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)

    for i, word in enumerate(word_vecs):
        W[i+1] = word_vecs[word]  # i+1 as i=0 is already filled with zeros
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=10, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones

    :param word_vecs: already loaded word_vectors from unsupervised training (word2vec) to add unknown words on them
    :param vocab: dictionary of words from training set and their document frequency.
    :param min_df: minimum document frequency of a word in order to make word vector for it
    :param k: length of vector
    :return: this method updates to the sent word_vecs referenced object
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC

    :param string: input string
    :param TREC: leave normal case
    :return: string after cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\n{2,}", "\n", string)
    string = re.sub(r"\t{2,}", "\n", string)
    return string.strip() if TREC else string.strip().lower()


if __name__ == "__main__":

    # create folder to save idx data inside
    if not os.path.exists(args.idxfolder_name):
        os.makedirs(args.idxfolder_name)

    # loading data into kfolds containing iterators of equivalent documents
    print "starting loading data..."

    df = get_ids(args.input,args.percentage)

    # creating vocabulary:
    print "creating vocab file.."
    vocab, max_l = create_vocab(df, args.data_folder)
    print "finished creating vocab file"
    print "vocab size: %s " % len(vocab)
    print "number of sentences: %s " % len(df.file)
    print "max sentence length: " + str(max_l)

    print "loading word2vec vectors...",
    # loading word2vec file
    w2v = load_bin_vec(args.w2v, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))


    print "initializing word vectors for each word in the vocab"
    # w2v is already initialized with word vectors representations
    # initialize random vectors for vocab entries that are not in w2v
    add_unknown_words(w2v, vocab)

    # W[i] is the wordvector of word with id i , while word_idx_map["word"] = id of this word
    W, word_idx_map = get_W(w2v)
    # initialize random vectors for all words
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)


    ################################################
    # CREATING IDX MATRICES FOR EACH TRAINING FOLD #
    ################################################
    # in order to save some memory we load word indices of each cross fold into one nd file to load them separately:
    # dividing training data into Kfolds
    # iterating over each Kfold
    # translating text into equivalent word_idx
    # save every fold data into 2dmatrix one file in idxfolder_name folder

    print "creating idx matrices for each fold 0.."
    kfolds = build_data_cv(df, args.data_folder, args.percentage)

    for k, fold in enumerate(kfolds):

        print "fold %s of $s ..." % (k, len(kfolds))
        for n in [("x_train", "y_train"), ("x_test", "y_test")]:

            x = fold[n[0]]
            y = fold[n[1]]

            patch_2darray = np.empty((0,max_l+1), int)  # +1 for class label todo:  remove class label as well

            for i, document in enumerate(x):

                document = clean_str(document)
                words = document.split(" ")

                row = [word_idx_map[w] for w in words]

                while len(row) < max_l:   #append zeros until all documents are equal no. of words # todo:sparse matrix
                    row.append(0)

                # add class label at the end of the sentence vector as compatible by 2nd file
                # todo : change that to better representation for idx vectors
                row.append(y[i])

                patch_2darray = np.append(patch_2darray, np.array([row]), axis=0)

            outfile = open(args.idxfolder_name + n[0].replace("x_","")+"_"+ str(k) + ".p","w+")
            cPickle.dump(patch_2darray)

    print "data loaded!"
    # we dont need to load word_idx_map anymore
    # cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    cPickle.dump(W, open("mr_w2v.p", "wb"))
    cPickle.dump(W2, open("mr_rnd.p", "wb"))
    print "dataset created!"
