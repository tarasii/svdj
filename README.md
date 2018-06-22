# SVDJ
Singular value decomposition using Jacobi algorithm for python.

Was adapted from matlab Nicolas Le Bihan and Stephen J. Sangwine function

http://freesourcecode.net/matlabprojects/5649/sourcecode/svdj.m#.WyokgfZuKUk

# Usage:
```
from svdj import svdj
u, s, v = svdj(a)
```

# Feature
Result U matrix of SVDJ don't match standart numpy SVD result U matrix while S and V are close

# Implementation
Matlab:        python:

 ```A*B``` => ```np.dot(a, b)``` for matrix multiplication np.dpt instead * 
 
 ```A'```  => ```a.conj().T``` for complex numbers A' consists not only A.T but also complex elements rotation
 
 ```A*A'```=> ```np.dot(a, a.conj().T)```
 
To get back A fom U, S, V on python:
```
u, s, v = svdj(a)
a = np.dot(u * s, v.conj().T)
```
S product of SVD have to be diagonal matrix, but algorithm returns simple vector s acording to standart ```numpy.linalg.svd()``` function, to get classical S product diagonal matrix have to ```np.diag(s)```

# Example :
```
A =  
-2125.+1467.j   964.-7903.j
 2818.-2029.j -9697.-6936.j
 1956.-7418.j   703.  -49.j 

Usvdj =
 0.10410349-0.5258483j  -0.35280053-0.01524117j
-0.70048792-0.46359275j  0.00734537-0.22968042j
-0.04557669+0.07047861j  0.24160253-0.87413995j
 
Usvd=
 0.23043339+0.48399838j  0.35280053+0.01524117j
 0.83872343-0.04630936j -0.00734537+0.22968042j
-0.00551535-0.08374993j -0.24160253+0.87413995j

S =
14603.77837422   8386.37085971

V =
-0.18063428-0.13443222j  0.97431988+0.j        
 0.97431988+0.j          0.18063428-0.13443222j
```
