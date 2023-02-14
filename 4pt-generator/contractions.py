#!/bin/python3

import numpy as np

# The point of this code is to generate C code that will itself compute the contractions
# Each contraction will consist of u correlators (of the form <u1 uA>) and d correlators (in similar form)
# Each contraction will also contain a sign for each of the u and d correlators
# The u/d correlators will be stored as the numerical index (1, 2, 3, 4) that pairs with each of A, B, C, and D
# The first element in the correlator array will be the sign (+/-) associated with it

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

# u correlators are the same as the d ones here (no special symmetries to apply)
u_correlators = np.copy(d_correlators)

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
# source/sink letters = "x" (for source), "w" (for sink), "y" (for t_1), "z" (for t_2)
# format: { q1 : ["initial gammas", "source/sink letter", "color index" "sink gammas"] }
index_lookup = {"u1" : ["ccs * g5s * ", "w", "a", ""],
                "u2" : ["", "w", "b", ""],
                "u3" : ["", "w", "d", ""],
                "u4" : ["", "w", "e", ""],
                "d1" : ["ccs * g5s * ", "w", "c", ""],
                "d2" : ["ccs * g5s * ", "w", "f", ""],
                "d3" : ["gmu * pls * ", "y", "m", ""],  # for scalar, unprimed operator
                "d4" : ["gnu * pls * ", "z", "n", ""],
                "uA" : ["", "y", "m", ""],
                "uB" : ["", "z", "n", ""],
                "uC" : ["", "x", "i", ""],
                "uD" : ["", "x", "l", ""],
                "dA" : ["", "x", "g", ""],
                "dB" : ["", "x", "h", " * ccs * g5s"],
                "dC" : ["", "x", "j", " * ccs * g5s"],
                "dD" : ["", "x", "k", " * ccs * g5s"]}

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
      newterm = index_lookup[indexA][0] + "Sl_" + index_lookup[indexB][1] + \
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

# TODO: Understand why the last 288 entries sum to the same as the first 288
# This must be a symmetry of exchange of quarks but we should make sure we understand this
# This allows us to limit the range of i to <12 and then just multiply the final answer by 2
for i in range(12):
  for j in range(24):
    print(corr_to_contraction(i, j))
