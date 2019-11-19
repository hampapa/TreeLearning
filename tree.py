import sys
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
        # are fales, 'full' can be implied
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


class DecisionTree:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.child = []
        self.is_leave = True
        
    def add_node(self, name):
        node = DecisionTree(self, name)
        self.child.append(node)
        self.is_leave = False

def importance():
    return 0
        
class DecisionTreeClassifier:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.A = list(self.X.columns)
        self.dt = DecisionTree(None, "root")

    def dtl(self, x, y, A, pex):
        # test if examples (x) are empty, if so return PLURALITY-VALUE of
        # parent examples (pex)
        if (len(x.index) == 0):
            pv = plurality_value(pex, y)
            return
        # else if all examples have the same classification return the
        # classification
        elif y.nunique() == 1:
            return
        # else if attributes A is empty return the PLURALITY-VALUE of
        # the examples x
        elif len(A) == 0:
            return
        else:
            a_max = importance()
        
        
        v_ik = x[A[0]].unique()
        for v in v_ik:
            ex_loc = x[x[A[0]] == v].index
            print(v)
            print(y.loc[ex_loc])
            
            
    def fit(self):
        self.dtl(self.X, self.Y, self.A, None)
                

def plurality_value(ex, y):

    # find the unique values
    uv = ex.unique()
    print (type(uv))

    sys.exit(0)
    
    loc = ex[ex['x.patrons'] == 'full'].index
    #print(loc)
    un = y.loc[loc].unique()
    print(type(un))
    print(un)
    return

def main():

    s_df = generate_sample(12)
    print(s_df)

    x_df = s_df.iloc[:,:-1]
    y = s_df.iloc[:,-1]

    dtree = DecisionTreeClassifier(x_df, y)
    dtree.fit()
    
    sys.exit(0)

    # experimental code
    loc = y.loc[y == 1]
    print(loc)
    newDF = x_df.iloc[loc.index]
    print(newDF)
    
    A = list(x_df.columns)
    print(x_df.index)
    #dtree = dtl(x_df, y, A, None)
    
    
if __name__ == "__main__":
    main()
    
