module AlphaBeta where

data Tree a = Node a [Tree a] deriving Show

data MinMax = Min | Max deriving Show

data InfInt = NegInf | Inf | Fin Int deriving (Show, Eq)

instance Ord InfInt where
    NegInf `compare` NegInf = LT
    NegInf `compare` Fin _ = LT
    NegInf `compare` Inf = LT
    Fin _ `compare` NegInf = LT
    Fin i `compare` Fin j = i `compare` j
    Fin _ `compare` Inf = LT
    Inf `compare` NegInf = GT
    Inf `compare` Fin _ = GT
    Inf `compare` Inf = LT

minMax :: (a -> Int) -> MinMax -> Tree a -> Int
minMax eval _ (Node n []) = eval n
minMax eval Max (Node n fs) = maximum scores
    where
        scores = fmap (minMax eval Min) fs
minMax eval Min (Node n fs) = minimum scores
    where
        scores = fmap (minMax eval Max) fs

alphabeta :: (a -> Int) -> MinMax -> InfInt -> InfInt -> Tree a -> InfInt
alphabeta eval _ alpha beta (Node n []) = alpha `max` (Fin $ eval n) `min` beta
alphabeta eval Max alpha beta (Node _ fs) = go alpha beta fs
    where
        go a _ [] = a
        go a b (t:ts) | a' < b = a'
                      | otherwise = go a' b ts
                   where a' = alphabeta eval Min alpha beta t
alphabeta eval Min alpha beta (Node _ fs) = go alpha beta fs
    where
        go _ b [] = b
        go a b (t:ts) | a < b' = b'
                      | otherwise = go a b' ts
                   where b' = alphabeta eval Max alpha beta t

testTree :: Tree Int
testTree = Node 6 [
             Node 3 [
               Node 5 [
                 Node 5 [
                   Node 5 [],
                   Node 6 []],
                 Node 4 [
                   Node 7 [],
                   Node 4 [],
                   Node 5 []]],
               Node 3 [
                 Node 3 [
                   Node 3 []]]],
             Node 6 [
               Node 6 [
                 Node 6 [
                   Node 6 []],
                 Node 6 [
                   Node 6 [],
                   Node 9 []]],
               Node 7 [
                 Node 7 [
                   Node 7 []]]],
             Node 5 [
               Node 5 [
                 Node 5 [
                   Node 5 []]],
               Node 8 [
                 Node 8 [
                   Node 8 [
                     Node 9 [],
                     Node 8 []],
                   Node 6 [
                     Node 6 []]]]]]
                     
