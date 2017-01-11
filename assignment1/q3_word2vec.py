import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    x = x/np.expand_dims(np.linalg.norm(x, axis=1), axis=1)
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE
    logits = np.expand_dims(np.matmul(predicted, np.transpose(outputVectors)), axis=0)
    y_pred = np.expand_dims(softmax(logits), axis=1)
    cost = -np.log(y_pred)[target]

    one_hot_label = np.zeros((outputVectors.shape[0],1))
    one_hot_label[target] = 1
    target_vector = outputVectors[target,:]
    weighted_vectors = outputVectors*y_pred

    gradPred = -target_vector + np.sum(weighted_vectors, axis=0)
    delta = y_pred
    delta[target] -= 1
    grad = delta*predicted

    ### END YOUR CODE
    
    return cost, gradPred, grad

def softmaxCostAndGradient_wrapper(vec, target, dataset):
    predicted = vec[0,:]
    outputVectors = vec[1:,:]
    cost, gradPred, grad = softmaxCostAndGradient(predicted, target, outputVectors, dataset)
    grad = np.vstack((gradPred, grad))
    return cost,grad

def test_softmaxCostAndGradient():
    print "Testing softmaxCostAndGradient..."
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(6,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    gradcheck_naive(lambda vec: softmaxCostAndGradient_wrapper(vec, 1, dataset), dummy_vectors)

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    counts = np.zeros((outputVectors.shape[0],1))
    neg_samples = []
    total_count = 0
    all_samples = []
    while(total_count<K):
        sample = dataset.sampleTokenIdx()
        if(sample!=target):
            if(counts[sample]==0):
                neg_samples.append(sample)
            all_samples.append(sample)
            counts[sample] = counts[sample] + 1
            total_count = total_count + 1
    logits = np.expand_dims(np.matmul(predicted, np.transpose(outputVectors)), axis=1)
    cost = np.squeeze(-np.log(sigmoid(logits[target]))) - np.sum(np.log(sigmoid(-logits[neg_samples]))*counts[neg_samples])

    gradPred = ((sigmoid(logits[target])-1)*outputVectors[target]) + np.sum((counts[neg_samples]*(1-sigmoid(-logits[neg_samples])))*outputVectors[neg_samples], axis=0)

    grad = np.zeros(outputVectors.shape)
    grad[target,:] = -(1-sigmoid(logits[target]))*predicted
    grad[neg_samples,:] = ((1-sigmoid(-logits[neg_samples]))*counts[neg_samples])*predicted
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient_wrapper(vec, target, dataset):
    predicted = vec[0,:]
    outputVectors = vec[1:,:]
    cost, gradPred, grad = negSamplingCostAndGradient(predicted, target, outputVectors, dataset)
    grad = np.vstack((gradPred, grad))
    return cost,grad

def test_negSamplingCostAndGradient():
    print "Testing negSamplingCostAndGradient..."
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(6,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    gradcheck_naive(lambda vec: negSamplingCostAndGradient_wrapper(vec, 1, dataset), dummy_vectors)

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in 
    #      the word vector list 
    # - inputVectors: "input" word vectors (as rows) for all tokens 
    # - outputVectors: "output" word vectors (as rows) for all tokens 
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors, 
    #      could be one of the two cost functions you  
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    currentIdx = tokens[currentWord]
    predicted = inputVectors[currentIdx, :]
    for word in contextWords:
        idx = tokens[word]
        c, gin, gout = word2vecCostAndGradient(predicted, idx, outputVectors, dataset)
        cost += c
        gradOut += gout
        gradIn[currentIdx, :] += gin
    ### END YOUR CODE
    return cost, gradIn, gradOut

def negSamplingCostAndGradient_orig(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE

    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    
    indices = [target]
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]
        
    labels = np.array([1] + [-1 for k in xrange(K)])
    vecs = outputVectors[indices,:]
    
    t = sigmoid(vecs.dot(predicted) * labels)
    cost = -np.sum(np.log(t))
    delta = labels * (t - 1)
    gradPred = delta.reshape((1,K+1)).dot(vecs).flatten()
    gradtemp = delta.reshape((K+1,1)).dot(predicted.reshape(
        (1,predicted.shape[0])))
    for k in xrange(K+1):
        grad[indices[k]] += gradtemp[k,:]
    
    return cost, gradPred, grad

def skipgram_orig(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in 
    #      the word vector list 
    # - inputVectors: "input" word vectors (as rows) for all tokens 
    # - outputVectors: "output" word vectors (as rows) for all tokens 
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors, 
    #      could be one of the two cost functions you  
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    currentIdx = tokens[currentWord]
    predicted = inputVectors[currentIdx, :]
    for word in contextWords:
        idx = tokens[word]
        c, gin, gout = negSamplingCostAndGradient_orig(predicted, idx, outputVectors, dataset)
        cost += c
        gradOut += gout
        gradIn[currentIdx, :] += gin
    ### END YOUR CODE
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    currentIdx = tokens[currentWord]
    counts = np.zeros((inputVectors.shape[0],1))
    predicted = np.zeros(inputVectors.shape[1])
    unique_context_words = []
    for word in contextWords:
        idx = tokens[word]
        if(counts[idx]==0):
            unique_context_words.append(idx)
        counts[idx] += 1
    predicted = np.sum(inputVectors[unique_context_words,:]*counts[unique_context_words,:], axis=0)
    cost, gin, gradOut = word2vecCostAndGradient(predicted, currentIdx, outputVectors, dataset)
    gradIn[unique_context_words,:] = gin*counts[unique_context_words,:]
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def word2vec_wrapper(word2vecModel, tokens, vec, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    N = vec.shape[0]
    inputVectors = vec[:N/2,:]
    outputVectors = vec[N/2:,:]

    C1 = random.randint(1,C)
    centerword, context = dataset.getRandomContext(C1)
    
    cost, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
    grad = np.vstack((gin,gout))
    return cost,grad

def test_word2vec_wrapper():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]

    cost1 = 0.0
    grad1 = np.zeros(wordVectors.shape)
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    # test_word2vec()
    test_word2vec_wrapper()
    # test_negSamplingCostAndGradient()