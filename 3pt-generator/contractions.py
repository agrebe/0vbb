#!/bin/python3

import numpy as np

# The point of this code is to generate C code that will itself compute the contractions
# Each contraction will consist of u correlators (of the form <u1 uA>) and d correlators (in similar form)
# Each contraction will also contain a sign for each of the u and d correlators
# The u/d correlators will be stored as the numerical index (1, 2, 3, 4) that pairs with each of A, B, C, and D
# The first element in the correlator array will be the sign (+/-) associated with it

u_correlators = np.array([[ 1, 1, 2, 3, 4],  # overall sign is +1, u1 pairs with uA, ..., u4 pairs with uD
                          [-1, 1, 3, 2, 4],  # overall sign is -1, u1 pairs with uA, u3 pairs with uB, ...
                          [ 1, 1, 4, 2, 3],  
                          [ 1, 3, 4, 1, 2],
                          [-1, 2, 4, 1, 3],
                          [ 1, 2, 3, 1, 4]], dtype='int')

# For this case, we also need to add the case where the final two elements are swapped (and the sign is flipped)
for i in range(6):
  row = np.copy(u_correlators[i,:])
  temp = row[3]
  row[3] = row[4]
  row[4] = temp
  row[0] *= -1
  u_correlators = np.vstack((u_correlators, row))

# For the d correlators, typing this out by hand is error prone, so we automate this
# This is literally just the 4-index epsilon tensor
# This is probably inefficient but gets the right answer
d_correlators = np.zeros((24,5), dtype='int')
d_counter = 0
for a in range(4):
  for b in range(4):
    for c in range(4):
      for d in range(4):
        M = np.zeros((4,4))
        M[0,a] = M[1,b] = M[2,c] = M[3,d] = 1
        det = np.linalg.det(M)
        if (det < 0): d_correlators[d_counter] = [-1, a+1, b+1, c+1, d+1]
        if (det > 0): d_correlators[d_counter] = [1, a+1, b+1, c+1, d+1]
        if (det == 0): d_counter -= 1
        d_counter += 1

# quarks whose spin indices pair
# This comes out of the algebra
pairings = [["u1", "u3"],
            ["u2", "d1"],
            ["u4", "d2"],
            ["d3", "uA"],
            ["d4", "uB"],
            ["uC", "dB"],
            ["uD", "dD"],
            ["dA", "dC"]]


# letters in alphabet
letters = ["A", "B", "C", "D"]

# convert contraction (i, j) to an array
# The array will have entries like ["u1", "uA"] for <u1 uA>
def corr_to_array(i, j):
  sign = u_correlators[i,0] * d_correlators[j,0]
  array = [["", ""],
           ["", ""],
           ["", ""],
           ["", ""],
           ["", ""],
           ["", ""],
           ["", ""],
           ["", ""]]
  for a in range(4):
    array[a][0] = "u" + str(u_correlators[i,a+1])
    array[a][1] = "u" + letters[a]
  for a in range(4):
    array[a+4][0] = "d" + str(d_correlators[j,a+1])
    array[a+4][1] = "d" + letters[a]
  return sign, array

# dictionary to store the color index, source, and sink associated with each of the quarks
# source/sink letters = "m" (for t_-), "x" (for t_x), "p" (for t_+)
# format: { q1 : ["initial gammas", "source/sink letter", "color index" "sink gammas"] }
index_lookup = {"u1" : ["ccs * g5s * ", "p", "a", ""],
                "u2" : ["", "p", "b", ""],
                "u3" : ["", "p", "d", ""],
                "u4" : ["", "p", "e", ""],
                "d1" : ["ccs * g5s * ", "p", "c", ""],
                "d2" : ["ccs * g5s * ", "p", "f", ""],
                "d3" : ["", "x", "m", ""],  # for scalar, unprimed operator
                "d4" : ["", "x", "n", ""],
                "uA" : ["", "x", "m", ""],
                "uB" : ["", "x", "n", ""],
                "uC" : ["", "m", "i", ""],
                "uD" : ["", "m", "l", ""],
                "dA" : ["", "m", "g", ""],
                "dB" : ["", "m", "h", " * ccs * g5s"],
                "dC" : ["", "m", "j", " * ccs * g5s"],
                "dD" : ["", "m", "k", " * ccs * g5s"]}

def corr_to_contraction(i, j):
  sign, array = corr_to_array(i, j)
  string = "tmp "
  if (sign == 1):
    string = string + "+"
  else:
    string = string + "-"
  string = string + "= "
  while len(array) > 0:
    string = string + "Trace("
    first_index = array[0][0]
    next_index = None
    index0, index1 = array[0][0], array[0][1]
    while next_index != first_index:
      if (string[(len(string)-6):] != "Trace("): string = string + " * "
      # print the numbered index first, transposing if necessary
      indexA = index1
      indexB = index0
      transposed = True
      if index0[1] == "1" or index0[1] == "2" or index0[1] == "3" or index0[1] == "4":
        indexA = index0
        indexB = index1
        transposed = False
      #string = string + " <" + indexA + " " + indexB + ">"
      #if transposed: string = string + "SpinTranspose"
      newterm = index_lookup[indexA][0] + "S_t" + index_lookup[indexB][1] + \
          index_lookup[indexA][1] + "[3*" + index_lookup[indexA][2] + "+" + \
          index_lookup[indexB][2] + "]" + index_lookup[indexB][3]
      if transposed:
        parts = newterm.split(" * ")
        newterm = ""
        for i in reversed(parts): newterm = newterm + i.replace("[", "_T[") + ".transpose()" + " * "
        newterm = newterm[:-3] # remove final " * " at end of string
      string = string + newterm
      # do two lookups
      # first lookup: determine next index in pairings table
      for a in range(8):
        if (index1 == pairings[a][0]): next_index = pairings[a][1]
        if (index1 == pairings[a][1]): next_index = pairings[a][0]
      # second lookup: get index0 = next_index in array
      index0 = next_index
      for a in range(len(array)):
        if (next_index == array[a][0]): index1 = array[a][1]
        if (next_index == array[a][1]): index1 = array[a][0]
        if (next_index == array[a][0] or next_index == array[a][1]): del array[a]; break
    string = string + ")"
    if (len(array) > 0): string = string + " * "
    # clean up the string
    # use the identities g5 = g5.T and C = -C.T
    string = string.replace("g5s.transpose()", "g5s").replace("ccs.transpose()", "-ccs").replace("].transpose()", "]")
    total_sign = (-1)**string.count("-")
    string = string.replace("-=", "+=")
    string = string.replace("-", "") 
    if (total_sign == -1): string = string.replace("+=", "-=")
  return string + ";"

for i in range(12):
  for j in range(24):
    print(corr_to_contraction(i, j))
