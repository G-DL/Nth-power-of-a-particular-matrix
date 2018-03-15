######################################################################################

"""This is a solution to a problem on linear algebra on how to compute the N-th power of a specific
2x2 matrix, by using its eigenvalues.

The problem was posed by 3blue1brown in its excellent video series on linear algebra. 
The original text of the problem can be found here: https://youtu.be/PFDu9oVAE-g?t=16m29s
"""

######################################################################################

"""
The program computes the N-th power of the matrix A = [0,1]
                                                      [1,1]

This particular matrix describes the linear transformation that moves
the unit vectors i and j in 2D space from

i = [1] and j = [0]
    [0]         [1]

to i^ = [0] and j^ = [1]
        [1]          [1]

Geometrically, the matrix A describes a flip in the orientation of each surface in the original vector space, 
and a skewing of the same surface, which however preserves the absolute value of the surface area.
The transformation is executed by rotating the unit vector in the first dimension by 90 degrees, 
and by skewing the unit vector in the second dimension by one unit towards the positive direction 
of the first dimension.

Graphical representation of the linear transformation A:

        BEFORE                             AFTER
  ^                                   ^
  |                                   |  x i^+j^ 
  |                                   |  
  |                                   |  
 jx  x i+j                         i^ x  x j^
  |                                   | /
  |                                   |/ 
 -O--x------------->                 -O--|------------->
  |  i                                |                
 
Note how the surface delimited by [O, i, i+j, j] has a positive surface area, while the transformed 
surface, delimited by [O, i^, i^+j^, j^], has a negative surface area.

In order to calculate the power of matrix A, the coordinate system is changed to the one in which
the eigenvectors of A are the unit vectors. The change of coordinate is performed by means of the matrix 
C, whose columns are the eigenvectors of A. The two eigenvectors of A are provided in the text of the problem, 
and are:
v0 = [2        ]  and v1 = [2        ]
     [1+sqrt(5)]           [1-sqrt(5)]

The matrix C which changes the basis thus possesses the value:
C = [v0,v1] or similarly C = [    2                2    ]
                             [1+sqrt(5)        1-sqrt(5)]

Conversion from the old vector space to the new one is performed by calculating the dot product of
(C^-1)(A)(C), which is labeled as matrix D. The resulting matrix D is diagonal, and consists of the 
eigenvalues of A distributed along its non-zero entries.

Each non-zero entry of D is then elevated to the N-th power, and its result is stored in the matrix 
called D_power. This computation is assigned manually, and performed element by element, to stress 
to the reader the fact that D is diagonal.

Once the power of D has been computed, its result is then changed back to the original vector space by inverting 
the order in which the C matrix and its inverse C^-1 are applied. Thus:
A^N = (C) (D) (C^-1)

To sum up, then, this program computes A^N as: 

A^N = (C) ((C^-1) (A) (C))^N (C^-1)

"""

######################################################################################

# Import dependencies
import numpy as np
import math

np.set_printoptions(suppress=True) # Suppresses scientific notation

######################################################################################

"""Data given by the text of the problem"""

#The A matrix
A = np.array([[0,1],
              [1,1]])

#The eigenvectors of A. Note that they are expressed as row vectors, 
#and will thus need to be transformed later to column vectors.
v0 = [2,
      1 + math.sqrt(5)]
v1 = [2,
      1 - math.sqrt(5)]

power = 10 #Modify this parameter to change the power of A that you want to compute

######################################################################################

"""Computing A^N"""
#Defining C, the matrix that describes the change of bases, and its inverse.
#C is also transformed, to reflect how the eigenvectors were provided in row form, not column form

C = np.array([v0,v1]).T
C_I = np.linalg.inv(C)

#Compute the diagonal matrix D = (C^-1)(A)(C)
#Calculation is performed from left to right per convention

D = np.dot(C_I,A)
D = np.dot(D,  C)

#Extracts the two eigenvalues of A, e0 and e1. These eigenvalues are located in the non-zero entries
#of D. The eigenvalues are then elevated to the desired power

e0 = D[0][0] ** power   
e1 = D[1][1] ** power

#The matrix D_power, corresponding to D^N, is filled with the eigenvalues of A elevated to the
#power of N

D_power = np.array([[e0,0],
                    [0,e1]])

#The bases are changed back to the original ones, by reversing the order in which
#the C matrix and its inverse are applied.

A_power = np.dot(C,  D_power)
A_power = np.dot(A_power,C_I)


######################################################################################

"""Outputs the result"""

print("A: \n", A)
print("\nN: \n",power)
print("\nA^N: \n", A_power)