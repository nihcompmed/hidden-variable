##========================================================================================
import numpy as np
from scipy import linalg
from scipy.integrate import quad
from scipy.optimize import fsolve

import function as ft
""" --------------------------------------------------------------------------------------
Inferring interaction from data by Free Energy Minimization (FEM)
input: time series s
output: interaction w, local field h0
"""
def fem(s):
    l,n = np.shape(s)
    m = np.mean(s[:-1],axis=0)
    ds = s[:-1] - m
    l1 = l-1

    c = np.cov(ds,rowvar=False,bias=True)
    c_inv = linalg.inv(c)
    dst = ds.T

    W = np.empty((n,n)) #; H0 = np.empty(n)
    
    nloop = 10000

    for i0 in range(n):
        s1 = s[1:,i0]
        h = s1
        cost = np.full(nloop,100.)
        for iloop in range(nloop):
            h_av = np.mean(h)
            hs_av = np.dot(dst,h-h_av)/l1
            w = np.dot(hs_av,c_inv)
            #h0=h_av-np.sum(w*m)
            h = np.dot(s[:-1,:],w[:]) # + h0
            
            s_model = np.tanh(h)
            cost[iloop]=np.mean((s1[:]-s_model[:])**2)
                        
            if cost[iloop] >= cost[iloop-1]: break
                       
            h *= np.divide(s1,s_model, out=np.ones_like(s1), where=s_model!=0)
            #t = np.where(s_model !=0.)[0]
            #h[t] *= s1[t]/s_model[t]
            
        W[i0,:] = w[:]
        #H0[i0] = h0
    return W #,H0 


"""---------------------------------------------------------------------------------------
Inferring interaction by Maximum Likelihood Estimation (MLE)
"""
def mle(s,rate,stop_criterion): 

    l,n = s.shape
    rate = rate/l
    
    s1 = s[:-1]
    W = np.zeros((n,n))

    nloop = 10000
    for i0 in range(n):
        st1 = s[1:,i0]
        
        #w01 = w0[i0,:]    
        w = np.zeros(n)
        h = np.zeros(l-1)
        cost = np.full(nloop,100.)
        for iloop in range(nloop):        
            dw = np.dot(s1.T,(st1 - np.tanh(h)))        
            w += rate*dw        
            h = np.dot(s1,w)            
            cost[iloop] = ((st1 - np.tanh(h))**2).mean() 
                       
            if ((stop_criterion=='yes') and (cost[iloop] >= cost[iloop-1])):
                break              
        
        W[i0,:] = w
    
    return W

"""---------------------------------------------------------------------------------------
Inferring interaction from data by nMF method
input: time series s
output: interaction w
"""
def nmf(s):
    l,n = s.shape
    # empirical value:  
    m = np.mean(s,axis=0)
    
    # A matrix
    A = 1-m**2
    A_inv = np.diag(1/A)
    A = np.diag(A)
    
    # equal-time correlation:
    ds = s - m
    C = np.cov(ds,rowvar=False,bias=True)
    C_inv = linalg.inv(C)
    
    # one-step-delayed correlation:
    s1 = s[1:]
    ds1 = s1 - np.mean(s1, axis=0)
    D = ft.cross_cov(ds1,ds[:-1])    
    ##------------------
    ## predict W:
    B = np.dot(D,C_inv)
    w = np.dot(A_inv,B)
    
    ##------------------
    #MSE = np.mean((w0 - w)**2)
    #slope = np.sum(w0*w)/np.sum(w0**2)    
    #print(MSE,slope)

    return w

"""---------------------------------------------------------------------------------------
Inferring interaction from data by TAP method
input: time series s
output: interaction w
"""
def tap(s):
    n = s.shape[1]
    # nMF part: ---------------------------------------------------    
    m = np.mean(s,axis=0)
    # A matrix
    A = 1-m**2
    A_inv = np.diag(1/A)
    A = np.diag(A)

    # equal-time correlation:
    ds = s - m
    C = np.cov(ds,rowvar=False,bias=True)
    C_inv = linalg.inv(C)
    
    # one-step-delayed correlation:
    s1 = s[1:]
    ds1 = s1 - np.mean(s1, axis=0)
    D = ft.cross_cov(ds1,ds[:-1])    
    ##------------------
    ## predict W_nMF:
    B = np.dot(D,C_inv)
    w_nMF = np.dot(A_inv,B)
    #--------------------------------------------------------------

    # TAP part    
    # solving Fi in equation: F(1-F)**2) = (1-m**2)sum_j W_nMF**2(1-m**2) ==> 0<F<1
    step = 0.001
    nloop = int(0.33/step)+2

    w2_nMF = w_nMF**2
    temp = np.empty(n) ; F = np.empty(n)
    for i in range(n):
       temp[i] = (1-m[i]**2)*np.sum(w2_nMF[i,:]*(1-m[:]**2))
    
       y=-1. ; iloop=0
       while y < 0 and iloop < nloop:
          x = iloop*step
          y = x*(1-x)**2-temp[i]
          iloop += 1

       F[i] = x
    
       #F[i]=np.sqrt(temp[i])
    
    # A_TAP matrix
    A_TAP = np.empty(n)
    for i in range(n):
       A_TAP[i] = A[i,i]*(1-F[i])
    A_TAP_inv = np.diag(1/A_TAP)
    
    w_TAP = np.dot(A_TAP_inv,B)

    return w_TAP

#=========================================================================================
"""---------------------------------------------------------------------------------------
Inferring interaction from data by exact Mean Field (eMF)
input: time series s
output: interaction w
"""
def emf(s,stop_criterion):
    n = s.shape[1]
    
    # nMF part: ---------------------------------------------------    
    m = np.mean(s,axis=0)
    # A matrix
    A = 1-m**2
    #A_inv = np.diag(1/A)
    A = np.diag(A)

    # equal-time correlation:
    ds = s - m
    C = np.cov(ds,rowvar=False,bias=True)
    C_inv = linalg.inv(C)
    
    # one-step-delayed correlation:
    s1 = s[1:]
    ds1 = s1 - np.mean(s1, axis=0)
    D = ft.cross_cov(ds1,ds[:-1])    
    ##------------------
    ## predict W_nMF:
    B = np.dot(D,C_inv)
    #w_nMF = np.dot(A_inv,B)
    
    #-------------------------------------------------------------------------------------
    fun1 = lambda x,H: (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)*np.tanh(H + x*np.sqrt(delta))
    fun2 = lambda x: (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)*(1-np.square(np.tanh(H + x*np.sqrt(delta))))
    
    w_eMF = np.empty((n,n))
    
    nloop = 100

    for i0 in range(n):
        cost = np.zeros(nloop+1) ; delta = 1.

        def integrand(H):
            y, err = quad(fun1, -np.inf, np.inf, args=(H,))
            return y - m[i0]
    
        for iloop in range(1,nloop):
            H = fsolve(integrand, 0.)
            H = float(H)
    
            a, err = quad(fun2, -np.inf, np.inf)
            a = float(a)
    
            if a !=0: 
                delta = (1/(a**2))* np.sum((B[i0,:]**2) * (1-m[:]**2))
                W_temp = B[i0,:]/a

            H_temp = np.dot(s[:-1,:], W_temp)
            cost[iloop] = np.mean((s1[:,i0] - np.tanh(H_temp))**2)
    
            if ((stop_criterion=='yes') and (cost[iloop] >= cost[iloop-1])): break

        w_eMF[i0,:] = W_temp[:]

    return w_eMF

#=========================================================================================
## update configurations of hidden variables according to interactions

def update_hidden(s,w,n):
    l,n2=np.shape(s)
    
    h0 = np.zeros(n2) # no external local fields
    
    h1 = np.empty(n2); p11 = np.empty(n2); p12 =np.empty(n2)
    h2=np.empty((n2, n2))
       
    # t = 0: ---------------------------    
    t = 0
    for i in range(n,n2):
        s[t,i] = 1.
        h2[:,i] = h0[i]+ np.sum(w[:,0:n2]*s[t,0:n2],axis=1)
        p1 = 1/np.prod(1+np.exp(-2*s[t+1,:]*h2[:,i]))
        p2 = 1/np.prod(1+np.exp(-2*s[t+1,:]*(h2[:,i]-2*w[:,i])))
        s[t,i] = ft.sign(p1/(p1+p2)-np.random.rand())

    # update s_hidden(t): t = 1 --> l-2:
    for t in range(1,l-1):
        # P(S_hidden(t)):
        h1[n:n2] = h0[n:n2]+np.sum(w[n:n2, :]*s[t-1, :], axis=1)
        p11[n:n2] = 1/(1+np.exp(-2*h1[n:n2])) # p(s =+1)
        p12[n:n2] = 1-p11[n:n2]                # p(s=-1)

        # P(S(t+1)):
        for i in range(n,n2):
            s[t,i] = 1.
            h2[:,i] = h0[:]+np.sum(w[:,0:n2]*s[t,0:n2],axis=1)
            p1 = p11[i]/np.prod(1+np.exp(-2*s[t+1,:]*h2[:,i]))
            p2 = p12[i]/np.prod(1+np.exp(-2*s[t+1,:]*(h2[:,i]-2*w[:,i])))
            s[t,i] = ft.sign(p1/(p1+p2)-np.random.rand())
                          
    # update s_hidden(t): t = l-1:
    h1[n:n2] = h0[n:n2]+np.sum(w[n:n2, :]*s[l-2, :], axis=1)
    p11[n:n2] = 1/(1+np.exp(-2*h1[n:n2]))     
    s[l-1,n:n2] = ft.sign_vec(p11[n:n2]-np.random.rand(n2-n))
        
    return s

#=========================================================================================
# inferring interactions with hidden variables

#-----------------------------------------------
def infer_hidden(s,nh,method):
    nrepeat = 100

    l,n = np.shape(s)
    sh = []
    if nh>0:
        sh = np.sign(np.random.rand(l,nh)-0.5)
        s = np.hstack((s,sh)) 
    
    cost_obs = np.empty(nrepeat)
    h0 = np.zeros(s.shape[1])
    for irepeat in range(nrepeat):
               
        if method == 'nmf': w = nmf(s)         
        if method == 'tap': w = tap(s)         
        if method == 'emf': w = emf(s,stop_criterion='yes')
        if method == 'mle': w = mle(s,1.,stop_criterion='yes')
        if method == 'fem': w = fem(s) 
                
        if nh>0:
            s = update_hidden(s,w,n)            
        h = np.matmul(s[:-1,:],w.T)
        cost_obs[irepeat] = np.mean((s[1:,:n] - np.tanh(h[:,:n]))**2)
        #print(irepeat,cost_obs[irepeat])
        
    return cost_obs,w,s[:,n:]
#-----------------------------------------------

"""---------------------------------------------------------------------------------------
2018.08.16: find coordinate of hidden variables, to verify that we can predict 
hidden configuration correctly
"""
def hidden_coordinate(w0,s0,w,sh):
    n2 = w0.shape[0]
    l,nh = sh.shape
    n = n2 - nh
    
    #sh0 = s0[:,n:]
    
    i_sign = np.ones(n2)
    i_tab = np.linspace(0,n2-1,n2).astype(int)
    wfinal = np.empty((n2,n2))
    cost1 = np.empty((n2,n2))
    cost2 = np.empty((n2,n2))
    
    # discrepancy
    for i in range(n,n2):
        for j in range(n,n2):
            cost1[i,j] = np.mean((w0[:n,i]-w[:n,j])**2) + np.mean((w0[i,:n]-w[j,:n])**2)
            cost2[i,j] = np.mean((w0[:n,i]+w[:n,j])**2) + np.mean((w0[i,:n]+w[j,:n])**2)
    
    # i cordinate
    for i in range(n,n2):
        j1 = np.argmin(cost1[i,n:n2]) + n
        j2 = np.argmin(cost2[i,n:n2]) + n
        
        if cost1[i,j1]<cost2[i,j2]:
            i_tab[i] = j1
            i_sign[i] = 1
        
            cost1[:,j1] = 100.
            cost2[:,j1] = 100.  # avoid double selection
        else:
            i_tab[i] = j2
            i_sign[i] = -1
            cost1[:,j2] = 100.
            cost2[:,j2] = 100. # avoid double selection
        
        #print('i_tab',i,i_tab[i],i_sign[i])
    
        i_tab = i_tab.astype(int)
        i_sign = i_sign.astype(int)
    
    # final w
    for i in range(n2):
        for j in range(n2):
            wfinal[i,j] = w[i_tab[i],i_tab[j]]*i_sign[i]*i_sign[j]
    
    # final sh
    shfinal = np.empty((l,nh))
    for i in range(n,n2):
        shfinal[:,i-n] = sh[:,i_tab[i]-n]*i_sign[i]
        
    #sh_accuracy = 1 - np.mean(np.abs(sh0-shfinal)/2.)
    #MSE = np.mean((w0 - wfinal)**2)

    #plt.plot([-1,1],[-1,1],'r')
    #plt.scatter(w0,wfinal)
       
    #print(MSE,sh_accuracy)

    return wfinal,shfinal
