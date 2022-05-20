# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:35:28 2022

@author: Vincent
"""

import math

class Node:
    
    def __init__(self, position, idx):
        
        self.position = position # position in x-y or x-y-z
        self.idx = idx  # ID of the node
        self.left = None
        self.right = None
       
class KD_tree:

    def __init__(self, dim):
        self.root = None
        self.dim = dim  # Dimension of a point, e.g., x-y or x-y-z

    def _dist(self, point1, point2):
        '''
        A helper function that calculates the distance between two nodes
        '''
        dist = 0
        for p1, p2 in zip(point1, point2):
               dist += (p1 - p2) ** 2
        return math.sqrt(dist)

    def _in_box(self, target, point, tol):
        '''
        Check if a point lies within the the 'box' centered with the target
        '''
        is_in_box = True
        for i in range(self.dim):
            is_in_box = is_in_box and (target[i] - tol <= point[i] <= target[i] + tol)

        return is_in_box   


    def _insert_helper(self, node, depth, point, idx):
        '''
        A helper function that recursively inserts a node (point)
        '''
        if not self.root:  # First node to be inserted ( that becomes root)
               self.root = Node(point, idx)

        if not node:  # A leaf node
               return Node(point, idx)

        idx_dim = depth % self.dim # Decide which dimension to split
        cur_pos = node.position[idx_dim]

        if point[idx_dim] < cur_pos: # Recursive splitting
                  node.left = self._insert_helper(node.left, depth + 1, point, idx)
        else:
                  node.right = self._insert_helper(node.right, depth + 1, point, idx)

        return node 

    def _search_helper(self, node, target, depth, tol, rslt):
        '''
        A helper function that recursively searches a point with distance tol to a target
        '''

        if not node:
               return None

        pos = node.position

        if self._in_box(target, pos, tol):
            if self._dist(target, pos) < tol:
                rslt.append(node.idx)

        idx_dim = depth % self.dim
        cur_pos = pos[idx_dim] 
        cur_tgt = target[idx_dim]

        if cur_tgt - tol < cur_pos:
                  self._search_helper(node.left, target, depth + 1, tol, rslt)

        if  cur_tgt + tol > cur_pos:
                  self._search_helper(node.right, target, depth + 1, tol, rslt)

    def insert(self, point, idx):

        self._insert_helper(self.root, 0, point, idx) 

    def search(self, target, tol):

        rslt = []
        self._search_helper(self.root, target, 0, tol, rslt)
        return rslt       
    
if __name__ == "__main__":

    kd_tree = KD_tree(2) # Initialize the 2D kD_tree
    
    # Insert points to the kd_tree
    points =[[3, 6], [17, 15], [13, 15], [6, 12], [9, 1], [2, 7], [10, 19]] 
    for idx, point in enumerate(points):
         kd_tree.insert(point, idx)
         
    ans = kd_tree.search([12, 14], 5)    
     
    print(ans)