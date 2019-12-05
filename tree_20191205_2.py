# Experimental implementation of the decision tree learning
# algorithm for discrete input values and exactly two possible
# output values (Boolean classification)

# By Gerald Hampapa, from October to December 2019
# Lines of code: 188

import sys
import random
import pandas as pd
import numpy as np

def generate_sample(n):
    sample_size = n
    np.random.seed(23)
    s_alt = np.random.choice([0,1], sample_size, replace=True)
    s_bar = np.random.choice([0,1], sample_size, replace=True)
    s_fri = np.random.choice([0,1], sample_size, replace=True)
    s_hun = np.random.choice([0,1], sample_size, replace=True)
    patrons = ['none','some','full']
    s_pat = np.random.choice(patrons, sample_size, replace=True)
    price = ['high','mid','low']
    s_pri = np.random.choice(price, sample_size, replace=True)
    food_type = ['French','Thai','Burger','Italian']
    s_type = np.random.choice(food_type, sample_size, replace=True)
    s_rai = np.random.choice([0,1], sample_size, replace=True)
    s_res = np.random.choice([0,1], sample_size, replace=True)
    wait_time = ['short','mid','long','too_long']
    s_wai = np.random.choice(wait_time, sample_size, replace=True)
    s_y = [0]*sample_size

    for i in range(sample_size):
        if (s_pat[i]=='none'):
            s_y[i] = 0
            continue
        if (s_pat[i]=='some'):
            s_y[i] = 1
            continue
        # no need to check if s_pat is 'full', if the first two if clauses
        # are false, 'full' can be implied
        if (s_wai[i]=='too_long'):
            s_y[i] = 0
            continue
        if (s_wai[i]=='short'):
            s_y[i] = 1
            continue
        if (s_wai[i]=='mid' and s_hun[i]==0):
            s_y[i] = 1
            continue
        if (s_wai[i]=='mid' and s_alt[i]==0):
            s_y[i] = 1
            continue
        if (s_wai[i]=='mid' and s_rai[i]==0):
            s_y[i] = 0
            continue
        else:
            s_y[i] = 1
            continue
        if (s_wai[i]=='long' and s_alt[i]==0 and s_res[i]==1):
            s_y[i] = 1
            continue
        elif (s_bar[i]==0):
            s_y[i] = 0
            continue
        else:
            s_y[i] = 1
            continue
        if (s_wai[i]=='long' and s_fri[i]==0):
            s_y[i] = 0
            continue
        else:
            s_y[i] = 1

    s_data = {'x.patrons':s_pat,
              'x.wait':s_wai,
              'x.alt':s_alt,
              'x.hungry':s_hun,
              'x.rain':s_rai,
              'x.res':s_res,
              'x.frisat':s_fri,
              'x.bar':s_bar,
              'x.price':s_pri,
              'x.type':s_type,
              'y':s_y }
    return pd.DataFrame(s_data)

def generate_no_A_sample(n):
    # This function is generating the samples for testing the
    # case where no attributes are left in the example set, but
    # positive and negative examples are in the example set.
    # This means that these examples have the same description,
    # but different classifications.
    # The code below shows that noise in the generative function
    # (in this implementation a random selection) is the reason.
    sample_size = n
    np.random.seed(33)

    v_k = ['one','two','three']
    s_A_1 = np.random.choice(v_k, sample_size, replace=True)
    s_y = np.random.choice([0,1], sample_size, replace=True)
    # y_k = ['t','f']
    # s_y = np.random.choice(y_k, sample_size, replace=True)
    
    s_data = {'x.A_1': s_A_1,
              'y': s_y }
    return pd.DataFrame(s_data)

def generate_no_ex_sample():
    # This function is generating samples which showcase data combinations
    # which lead the algorithm to attribute selections resulting in branches
    # without examples. This is used to test the "no examples" condition
    # in the tree learning algorithm.
    s_data = {'A1': ["one","one","one","two","two","two",
                     "three","three","three"],
              'A2': ["x","y","z","x","y","z","y","z","y"],
              'y':  [1,1,1,0,0,0,1,0,1]}
    return pd.DataFrame(s_data)
    
class DecisionTree:
    def __init__(self, parent, name, clfn=None):
        self.parent = parent
        self.name = name
        self.child = []
        self.clfn = clfn
        
    def add_node(self, node):
        self.child.append(node)

    def set_parent(self, node):
        self.parent = node

    def set_name(self, name):
        self.name = name
        
def B_q(q):
    # entropy of a Boolean random variable
    if (q == 0 or q == 1):
        return 0
    return -(q*np.log2(q)+(1-q)*np.log2(1-q))

def plurality_value(y):
    # find the frequencies a value appears and select the value which
    # appears most often. Select randomly if there is more than one value
    # appearing at the same time.
    
    vc = y.value_counts()
    print(vc)
    mxv = vc.max()
    chk_dupl = vc.loc[vc == mxv]
    if (len(chk_dupl) > 1):
        choices = chk_dupl.index.tolist()
        clfn_rc = random.choice(choices)
        return clfn_rc
    else:
        # only one value possible here, so use [0] to get correct value type
        return chk_dupl.index.values[0]

def importance(A, X, Y):
    p = Y[Y == 1].count()
    n = Y[Y == 0].count()
    B = B_q(p/(p+n))
    max_gain = 0
    max_i = 0
    
    for i, a_i in enumerate(A):
        v_ik = X[a_i].unique()
        num_v = len(v_ik)
        p_k = pd.Series(np.zeros(num_v))
        n_k = pd.Series(np.zeros(num_v))
        
        for j, v in enumerate(v_ik):
            ex_loc = X[X[a_i] == v].index
            p_count = Y.loc[ex_loc][Y == 1].count()
            n_count = Y.loc[ex_loc][Y == 0].count()
            p_k[j] = p_count
            n_k[j] = n_count

        q = p_k/(p_k+n_k)
        B_entropy = q.apply(B_q)
        remainder_a_i = ((p_k+n_k)/(p+n)*B_entropy).sum()
        gain_a_i = B - remainder_a_i
        if (gain_a_i > max_gain):
            max_gain = gain_a_i
            max_i = i
            
    return A[max_i]
        
class DecisionTreeClassifier:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y  # expects 0 or 1 
        self.A = list(self.X.columns)
        self.dt = None

    def dtl(self, x, y, A, pex, pey):
        # dtl() is the implementation of the core decision tree learning
        # algorithm based on the pseudo-code from AIMA 3ed., p. 702
        
        # Test if examples (x) are empty, if so return PLURALITY-VALUE of
        # parent examples (pex).
        # My conclusion after testing is that this part of the code will
        # never be reached unless N/A values are allowed in the dataset.
        if (len(x.index) == 0):
            print("enter 1")
            pv_clfn = plurality_value(pey)
            dt = DecisionTree(None,None,pv_clfn)
            return dt
        # else if all examples have the same classification return the
        # classification
        elif y.nunique() == 1:
            clfn = y.iloc[0]
            dt = DecisionTree(None,None,clfn)
            return dt
        # Else if attributes A is empty return the PLURALITY-VALUE of
        # the examples x. This can happen when no attributes are
        # left, but both positive and negative examples. Usually error
        # or noise in the data can be a cause for this condition to be
        # triggered.
        elif len(A) == 0:
            pv_clfn = plurality_value(y)
            dt = DecisionTree(None,None,pv_clfn)
            return dt
        else:
            a_max = importance(A, x, y)
            print(">>> a_max: ", a_max)
            dt = DecisionTree(None, a_max)
            v_k = x[a_max].unique()
            A.remove(a_max)
            for v in v_k:
                exs = x[x[a_max] == v]
                #print("v: ", v)
                yxs = y.loc[exs.index]
                subtree = self.dtl(exs, yxs, A, x, y)
                subtree.set_parent(dt)
                subtree.set_name(v)
                dt.add_node(subtree)
                
        return dt    
        
    def fit(self):
        self.dt = self.dtl(self.X, self.Y, self.A, self.X, self.Y)

    def predict(self, X):
        pred_len = len(X)
        pred_ser = pd.Series(np.zeros(pred_len), index = X.index)

        node = self.dt
        print(node.name)
        for c in node.child:
            print(c.name)
            for cc in c.child:
                print(cc.name)

        #for i, x in X.iterrows():
        #    node = self.dt
        #    next_node = x.loc[node.name]
        #    print("next_node: ", next_node)
        #    node = node.child[0]
        #    print("nn: ", node.child)
            
            #while !node.clfn:
            #    print("x_obs: ", type(x))
            #    chld = x.loc[node.name]
        
        return pred_ser
                

def main():
    
    s_df = generate_sample(1200)
    #s_df = generate_no_A_sample(20)
    #s_df = generate_no_ex_sample()
    #print(s_df)

    x_df = s_df.iloc[:,:-1]
    y = s_df.iloc[:,-1]
    
    # remove data samples with n/a values in their feature set
    # before calling the classifier
    dtree = DecisionTreeClassifier(x_df, y)
    dtree.fit()

    s_test_df = generate_sample(12)
    x_test_df = s_test_df.iloc[:,:-1]
    y_test = s_test_df.iloc[:,-1]
    y_pred = dtree.predict(x_test_df)
    
    
if __name__ == "__main__":
    main()
