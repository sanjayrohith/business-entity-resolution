"""Original MIT-licensed NumPy fallbacks; trained from scratch, no pretrained weights."""
import numpy as np

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-35,35)))

class Logistic:
    def fit(self,X,y,w,steps=240):
        self.mean=X.mean(axis=0);self.scale=np.maximum(X.std(axis=0),.05)
        z=(X-self.mean)/self.scale
        z=np.column_stack([np.ones(len(X),dtype=np.float32),z]).astype(np.float32)
        self.coef=np.zeros(z.shape[1],dtype=np.float32)
        prevalence=np.dot(w,y)/w.sum(); self.coef[0]=np.log(prevalence/(1-prevalence))
        m=np.zeros_like(self.coef);v=m.copy(); total=w.sum()
        for t in range(1,steps+1):
            p=sigmoid(z@self.coef)
            grad=z.T@(w*(p-y))/total
            grad[1:]+=1e-4*self.coef[1:]
            m=.9*m+.1*grad;v=.999*v+.001*grad*grad
            self.coef-=.04*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        return self
    def predict(self,X):return sigmoid(self.coef[0]+((X-self.mean)/self.scale)@self.coef[1:])
    def state(self):return {'type':'logistic','mean':self.mean.tolist(),'scale':self.scale.tolist(),'coef':self.coef.tolist()}
    @classmethod
    def load(cls,s):
        obj=cls()
        for key in ['mean','scale','coef']:setattr(obj,key,np.asarray(s[key],dtype=np.float32))
        return obj

class HistogramBoost:
    def fit(self,X,y,w,trees=70,depth=3,bins=32,lr=.12):
        self.lr=lr; self.cuts=[np.unique(np.quantile(X[:,j],np.linspace(0,1,bins+1)[1:-1])).astype(np.float32) for j in range(X.shape[1])]
        B=self.bin(X);self.trees=[]
        prior=np.dot(y,w)/w.sum();self.base=float(np.log(prior/(1-prior))); score=np.full(len(X),self.base)
        for t in range(trees):
            p=sigmoid(score);g=(y-p)*w;h=np.maximum(p*(1-p),1e-5)*w
            tree=self.grow(B,g,h,np.arange(len(X)),depth,bins)
            self.trees.append(tree);score+=lr*self.apply(B,tree)
        return self
    def bin(self,X):return np.column_stack([np.searchsorted(c,X[:,j],side='right') for j,c in enumerate(self.cuts)]).astype(np.uint8)
    def grow(self,B,g,h,idx,depth,bins):
        G=g[idx].sum();H=h[idx].sum();leaf=float(np.clip(G/(H+5.),-5.,5.))
        if depth==0 or len(idx)<40:return {'leaf':leaf}
        best=(0.,None,None)
        for j in range(B.shape[1]):
            gb=np.bincount(B[idx,j],weights=g[idx],minlength=bins+1).cumsum()[:-1]
            hb=np.bincount(B[idx,j],weights=h[idx],minlength=bins+1).cumsum()[:-1]
            counts=np.bincount(B[idx,j],minlength=bins+1).cumsum()[:-1]
            gain=gb*gb/(hb+5)+(G-gb)**2/(H-hb+5)-G*G/(H+5)
            gain[(counts<20)|(len(idx)-counts<20)|(hb<1)|(H-hb<1)]=-np.inf
            k=int(np.argmax(gain))
            if gain[k]>best[0]:best=(float(gain[k]),j,k)
        if best[1] is None:return {'leaf':leaf}
        _,j,k=best; mask=B[idx,j]<=k
        return {'feature':j,'bin':k,'gain':best[0],'left':self.grow(B,g,h,idx[mask],depth-1,bins),'right':self.grow(B,g,h,idx[~mask],depth-1,bins)}
    def apply(self,B,t):
        out=np.zeros(len(B),dtype=np.float32)
        def visit(tree,idx):
            if 'leaf' in tree:out[idx]=tree['leaf'];return
            mask=B[idx,tree['feature']]<=tree['bin']
            visit(tree['left'],idx[mask]);visit(tree['right'],idx[~mask])
        visit(t,np.arange(len(B)));return out
    def predict(self,X):
        B=self.bin(X);s=np.full(len(X),self.base)
        for t in self.trees:s+=self.lr*self.apply(B,t)
        return sigmoid(s)
    def state(self):return {'type':'histogram_boost','base':self.base,'lr':self.lr,'cuts':[c.tolist() for c in self.cuts],'trees':self.trees}
    @classmethod
    def load(cls,s):
        obj=cls();obj.base=s['base'];obj.lr=s['lr'];obj.trees=s['trees'];obj.cuts=[np.asarray(c,dtype=np.float32) for c in s['cuts']];return obj

def parameters(s):
    if s['type']=='logistic':return len(s['coef'])+len(s['mean'])+len(s['scale'])
    def count(t):return 1 if 'leaf'in t else 2+count(t['left'])+count(t['right'])
    return 2+sum(map(len,s['cuts']))+sum(count(t) for t in s['trees'])
