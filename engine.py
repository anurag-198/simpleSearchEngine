from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from scipy.sparse.linalg import svds
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import randomized_svd,svd_flip

import numpy as np

categories = [
    'comp.sys.mac.hardware',
    'rec.sport.baseball',
    'sci.med',
    'talk.politics.guns',
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
    'comp.os.ms-windows.misc',
    'rec.autos',
    'sci.crypt',
    'misc.forsale',
    'comp.sys.ibm.pc.hardware',
    'rec.motorcycles',
    'sci.electronics',
    'talk.politics.misc',
    'comp.windows.x',
    'rec.sport.hockey',
    'talk.politics.mideast',
    'soc.religion.christian'
]


def querymatcher(query, transposedMatrix) :

    return

def similar(X,Y) :
    prod = 0
    for i in range(len(X)):
        prod = prod + X[i] * Y[i]

    b1 = 0
    for i in range(len(X)):
        b1 += X[i] * X[i]

    b2 = 0
    for i in range(len(Y)) :
        b2 += Y[i] * Y[i]

    if (b1 == 0) or (b2 == 0):
        return -9999999

    return prod / (b1 * b2)

class StemTokenizer(object):
    def __init__(self):
        self.stm = SnowballStemmer("english")

    def __call__(self, doc):
        list1 = [self.stm.stem(t) for t in word_tokenize(doc)]
        list2 = []
        for a in list1:
            fl = 0
            for i in range(len(a)):
                if (not (((a[i] >= 'a') and (a[i] <= 'z')) or ((a[i] >= 'A') and (a[i] <= 'Z')))):
                    fl = 1
                    break
            if fl == 0:
                list2.append(str(a).lower())

        return list2

def calculatedf(lis):
    dicti = {}

    for i in range(len(lis)):
        dicti[lis[i]] = 0

    for i in range(len(lis)):
        dicti[lis[i]] += 1

    for i in range(len(lis)):
        if (dicti[lis[i]] >= 1) :
            dicti[lis[i]] /= len(lis)

    return dicti


stp = {"abc","hrs","af","are","ar","aa", "aaa","ab","abl","aap","adb","zz", "think","does","just","like","said","did","the","they","our","not","a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"}
stop = text.ENGLISH_STOP_WORDS.union(stp)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

vectorizer = TfidfVectorizer(stop_words=stop,min_df=0.01,max_df=0.15,lowercase=1,tokenizer=StemTokenizer())
v1 = CountVectorizer(stop_words=stop,min_df=0.01,max_df=0.15,lowercase=1,tokenizer=StemTokenizer())
X = vectorizer.fit_transform(dataset.data)
Y = v1.fit_transform(dataset.data)

Ych = Y.toarray()
Xch = X.toarray()

myvocab = vectorizer.vocabulary_

print(myvocab)

features = v1.get_feature_names()
print(features)

######## doing dimensionality reduction for #############################

kin = len(features)/2
kin = int(kin)

U, Sigma, VT = svds(X, k=kin, tol=0)
Sigma = Sigma[::-1]
U, VT = svd_flip(U[:, ::-1], VT[::-1])

U = np.asarray(U)
Sigma = np.asarray(Sigma)
VT = np.asarray(VT)
V = np.transpose(VT)

UT = np.transpose(U)

print(VT.shape)
print(X.shape)
Xin = X.todense()
Xin = np.asarray(Xin)
print(Xin.shape)
res = np.dot(Xin, V)


###############     handling the input query      ###############################

for i in range(10):
    print("##################################################################################################")
    query = input("Enter your queries \n")
    stm = StemTokenizer()
    lis = stm(query)

    virtuallist = []
    for i in range(len(lis)):
        if lis[i] in stp:
            virtuallist.append(i)

    lis2 = []
    for i in range(len(lis)):
        if i not in virtuallist:
            lis2.append(lis[i])

    print(lis2)
    lis = lis2

    idf = [0] * len(vectorizer.get_feature_names())
    dfdic = calculatedf(lis)                         #dictionary for tf terms

    df = [0] * len(vectorizer.get_feature_names())  #initialised list for the tf terms

    finallis = []
    for i in range(len(lis)):
        if (lis[i] in features):
            finallis.append(lis[i])

    lis = finallis

    for i in range(len(lis)):
        if (lis[i] in features):
            index = myvocab[lis[i]]
            df[index] = dfdic[lis[i]]

    for i in range(len(lis)) :
        if (lis[i] in features) :
            index = myvocab[lis[i]]

            for j in range(Y.shape[0]):
                if (Ych[j][index] > 0) :
                    idf[index] += 1

            idf[index] = np.log(Y.shape[0]/(idf[index] + 1))


    tfidf = [0] * len(vectorizer.get_feature_names())
    for i in range(len(idf)):
        tfidf[i] = df[i] * idf[i]

    print(df)
    print(idf)
    print(tfidf)

    query = np.dot(tfidf, V)

    #print(query)

    rank = [0] * res.shape[0]

    print("#########################")

    for i in range(res.shape[0]):
        rank[i] = similar(res[i],query)

    mylist = list(range(res.shape[0]))
    mylist1 = list(zip(mylist, rank))
    mylist1 = sorted(mylist1, key = lambda x: x[1], reverse = True)



    print("printing the top 5 documents retrived for given query \n")

    for i in range(5):
        print("printing the " + str(i + 1) + " record for you which is in doc no " + str(mylist1[i][0]) + "\n\n\n ")
        print(dataset.data[mylist1[i][0]])