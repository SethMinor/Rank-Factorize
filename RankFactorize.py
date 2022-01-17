# -*- coding: utf-8 -*-

"""
Welcome to

Seth Minor's Incredibly Ineffeicient Rank Factorization Algorithm
(or SMIIRFA, for short)

Enjoy your stay!
"""


import numpy as np


"""
The GS algorithm immediately below was developed by Dr. Eric McAlister
of Fort Lewis College.
"""

def gram_schmidt(data):
    gs_basis = []
    for i in range(len(data)):
        q_tilde = data[i]
        for j in range(len(gs_basis)):
            q_tilde = q_tilde - gs_basis[j]@data[i]*gs_basis[j]
        if np.linalg.norm(q_tilde)<=1e-10:
            print('Vectors are linearly dependent.')
            print('GS algorithm terminates at iteration' , i+1)
            return gs_basis
        else:
            q_tilde = q_tilde/(np.linalg.norm(q_tilde))
            gs_basis.append(q_tilde)
    return gs_basis

def QR_factorization(data):
    Q = np.array(gram_schmidt(data.T)).T
    R = Q.T@data
    return Q,R

def back_subst(R,b):
    n = R.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = b[i]
        for j in range(i+1,n):
            x[i] = x[i] - R[i,j]*x[j]
        x[i] = x[i]/R[i,i]
    return x

# Q,R = QR_factorization(A)


# The rank factorization function

# Enter an m x n matrix, A
# We'll call A's m-vector columns a1,a2,...an
# For example,

A = np.array([[1,7,-8,15,15,-30,-23,-27],
              [2,4,0,10,4,-2,-10,-8],
              [7,9,1,25,8,1,-24,-12],
              [0,3,-4,6,7,-15,-10,-13],
              [1,-1,-1,-1,0,-1,0,4],
              [3,1,6,5,-5,20,1,9]])
print("A is")
print(A)

(m,n) = np.shape(A)
print("A is a ",m,"x",n," matrix")
print("")


def columnVector(M,i):
    a = np.zeros(m)
    for j in range(m):
        a[j] = M[j][i-1]
    return a

print("To test the columnVector function, a1 is")
print(columnVector(A,1))
print("")


def columnSet(M,p):
    for i in range(p):
        counter = i + 1
        if (counter == 1):
            S = columnVector(M,1)
        elif (counter != 1):
            S = np.vstack((S,columnVector(M,counter)))
    return S 

print("To test the columnSet function, {columns(A_5)} is")
print(columnSet(A,5))
print("")


def columnSetStack(M,i,N,j):
    b = columnSet(M,i)
    a = columnVector(N,j)
    S = np.vstack((b,a))
    return S
    
print("To test the columSetStack function, {columns(A_5),a8} is")
print(columnSetStack(A,5,A,8))
print("")

  
def unitVector(m,p):
    e = np.zeros(m)
    e[p-1]=1
    return e

print("To test the unitVector function in R^m, e3 is")
print(unitVector(m,3))
print("")

def solve(M,b):
    # solving Mx = b
    Q,R = QR_factorization(M)
    x = back_subst(R,Q.T@b)
    return x

Atest = np.array([[5,2],[2,3]])
btest = np.array([2,4])
print("To test the solve function, Ax = b, ")
print("where A = ")
print(Atest)
print(" and b = ",btest,",")
print("we have a solution x for x =",solve(Atest,btest))
print("")


def gram_schmidtChecker(data):
    gs_basis = []
    for i in range(len(data)):
        q_tilde = data[i]
        for j in range(len(gs_basis)):
            q_tilde = q_tilde - gs_basis[j]@data[i]*gs_basis[j]
        if np.linalg.norm(q_tilde)<=1e-10:
            dependence = 1
            return dependence
        else:
            q_tilde = q_tilde/(np.linalg.norm(q_tilde))
            gs_basis.append(q_tilde)
            dependence = 0
    return dependence

print("To test the linear-dependence checking function (the modified GS algorithm),\
 we want to make sure that GSchecker(A)=1 (dependent). After running it (where 0 =\
  independent, 1 = dependent), we get")
print("GSchecker(A) =",gram_schmidtChecker(A))
print("")
print("")

# these print statements are helpful for debugging

def RankFactorize(A):
    
    print("Calculating the rank factorization of A...")
    print("")
    
    setback = 0
    
    for i in range(n):
        
        loop = i + 1
        print("Loop",loop)
        
        if (loop == 1):
            
            C = columnVector(A,1)
            C = C[:,np.newaxis]
            print("C is")
            print(C)
            
            R = unitVector(m,1)
            R = R[:,np.newaxis]
            print("R is")
            print(R)
            
            print("")
            
        elif (loop != 1):
            
            numbaOfCols = i - setback
            print("C_",i,"had",numbaOfCols,"column(s)...")
            
            a = columnVector(A,loop)
            a = a[:,np.newaxis]
            print("a_",loop," is")
            print(a)
            
            if (m >= loop):
                e = unitVector(m,loop)
                e = e[:,np.newaxis]
            elif (m < loop):
                e = unitVector(m,m)
                e = e[:,np.newaxis]
            print("e_",loop," is")
            print(e)
            
            colC = columnSet(C,i-setback)
            print("The set of columns, {columns(C_",i,")}, is")
            print(colC)
            
            S = columnSetStack(C,i-setback,A,loop)
            print("The column stack, S_",loop,"= {columns(C_",loop-1,"), a_",loop,"}, is")
            print(S)
            
            d = gram_schmidtChecker(S)
            
            if (d == 0):
                
                print("S_",loop," is linearly independent")
                
                C = np.hstack((C,a))
                print("C_",loop," is")
                print(C)
                
                R = np.hstack((R,e))
                print("R_",loop," is")
                print(R)
                
                print("")
                
            elif (d == 1):
                
                print("S_",loop," is linearly dependent")
                
                # C stays the same
                print("C_",loop," is")
                print(C)
                
                print("Figuring out how to build a_",loop," from vectors in S_",i,"...")
                print("Solving C_",loop,"x = a_",loop,"...")
                coeffs = solve(C,a)
                
                x = np.zeros(m)
                for i in range(len(x)):
                    if i in range(len(coeffs)):
                        x[i]=coeffs[i]
                
                x = x[:,np.newaxis]
                print("Our weight vector, x, of size",numbaOfCols,"is")
                print(coeffs)
                
                R = np.hstack((R,x))
                print("R_",loop," is")
                print(R)
                
                setback = setback + 1
                
                print("")
                
    # R needs to be 3x8, not 6x8
    Rstar = np.empty((3,8))
    
    for r in range(3):
        for c in range(8):
            if (np.absolute(R[r][c]) < (2e-14)):
                R[r][c] = 0
            Rstar[r][c] = R[r][c]
                
    return C,Rstar
    
    
C,R = RankFactorize(A)

print("The rank factorization of A is given by A = CR, where C is")
print(C)
print("and R is")
print(R)
print("")
    
print("----------------------------------------------------")
print("")