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
from threading import Thread
import Queue
import cPickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import random
import copy
from  IPython.core.debugger import Tracer as Tracer

# from sklearn.cross_validation import StratifiedKFold, train_test_split

##########################
# COMMAND LINE ARGUMENTS #
##########################
parser = argparse.ArgumentParser(description='script create sub sample of datasets')
parser.add_argument('-i', '--input', help='training file path', required=True)
parser.add_argument('-f', '--data_folder', help='data folder', required=True)
parser.add_argument('-w2v', '--w2v', help='word 2 vec binary file', required=True)
parser.add_argument('-p', '--percentage', help=" percentage of training set to experiment on", required=False)
# parser.add_argument('-uniq', '--unique', help="unique words or all from each file", action='store_true', required=False)
parser.add_argument('-idxfname', '--idxfolder_name', help="optional custom folder path to save all idx files created", required=False)

args = parser.parse_args()
if args.percentage is None:
    args.percentage = 1
else:
    args.percentage = float(args.percentage)

if args.idxfolder_name is None:
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
    df = df.reset_index()

    return df

def read_data_files(filenames, datapath, ids=None):
    """
    function to read data files from folder and return as iterator in order not to load on data

    :param filename: list or iterator of file names
    :param datapath: datapath to concatinate before each filename
    :return: iterator of text containing all text in all files in order
    """
    filenames = np.array(filenames) # make sure it's array
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
        o = cPickle.load(open("vocab_max_l.p", "rb"))  # search if vocab file is already existing
        vocab = o[0]
        max_l = o[1]
    else:
        vocab = defaultdict(int)
        max_l = 0
        for d in read_data_files(df.file, datapath):
            words = clean_str(d).split(" ")
            if len(words) > max_l:
                max_l = len(words)

            for w in words:
                vocab[w] += 1

        cPickle.dump([vocab, max_l], open("vocab_max_l.p", "wb"))
    return  vocab, max_l

def build_data_cv(df, datapath, n_folds=10):
    """
    function creating splits for cross folds

    :param df: pandas dataframe containing training examples in form of {file, label}
    :param datapath: data path that has all training files
    :param n_folds: number of folds in case of cross validation
    :return: array of tuples in form of (id_train, X_train, y_train, id_test, X_test, y_test) where
    id is the file name (document id), X is the document text, and y is the class label
    """

    kf_ids = [np.array([], dtype=int) for _ in range(0, 10)]

    # split each class ids into number of folds and add to each fold
    ls = df.label.unique()
    for l in ls:
        c = df.label[df.label == l].index
        for i in c:
            rnd = random.randint(0, n_folds-1)
            kf_ids[rnd] = np.append(kf_ids[rnd], i)

    splits_data = []

    for ids in kf_ids:

        splits_data.append({
            "id": np.array(df.file[ids]),
            "x": read_data_files(list(df.file), datapath, ids),
            "y": np.array(df.label[ids])
        })

    return splits_data

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

    # padding = filtersize - 1 see : conv_net_sentence.py  #todo : add padding implicitly
    pad = 4

    # create folder to save idx data inside
    if not os.path.exists(args.idxfolder_name):
        os.makedirs(args.idxfolder_name)

    # loading data into kfolds containing iterators of equivalent documents
    print "starting loading data..."

    df = get_ids(args.input, args.percentage)

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

    print "creating idx matrices for each fold .."
    n_folds = 10
    splits = build_data_cv(df, args.data_folder, n_folds)


    def create_idx(document, label, word_idx_map, pad, max_l, patch_2darray):

        q = patch_2darray.get()

        document = clean_str(document)
        words = document.split(" ")
        row = [word_idx_map[w] for w in words if w in word_idx_map]
        while len(row) < max_l + 2*pad:   #append zeros until all documents are equal no. of words # todo:sparse matrix
                row.append(0)

        row.append(label)
        q = np.append(q, np.array([row]), axis=0)
        patch_2darray.put(q)


    # parallelizing generation of indices
    # iterate over patches and add to thread list
    # if active threads = max thread number wait until threads empty
    # add more until all patches are over
    active_threads = []
    max_thread_no = 7

    for k, fold in enumerate(splits):

        print "patch %s of %s  started" % (k+1, n_folds)

        x = fold["x"]
        y = fold["y"]

        patch_2darray = Queue.Queue()
        patch_2darray.put(np.zeros((0, max_l+1+2*pad), int))

        for i, document in enumerate(x):

            t = Thread(target=create_idx, args=(document, y[i], word_idx_map, pad, max_l, patch_2darray))
            t.start()

            active_threads.append(t)

            if len(active_threads) > max_thread_no:
                for t in active_threads:
                    t.join()
                active_threads = []

        outfile = open("%s%s.p" % (args.idxfolder_name, k), "wb")
        matrix = csr_matrix(patch_2darray.get())
        # matrix = patch_2darray.get()
        cPickle.dump(matrix, outfile)
        outfile.close()

    print "data loaded!"
    # we dont need to load word_idx_map anymore
    cPickle.dump(W, open("mr_w2v.p", "wb"))
    cPickle.dump(W2, open("mr_rnd.p", "wb"))
    print "dataset created!"
