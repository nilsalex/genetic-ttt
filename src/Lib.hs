module Lib where

import qualified Numeric.LinearAlgebra as Matrix
import qualified Numeric.LinearAlgebra.Data as HM

import Control.Monad.Random
import Control.Monad.Random.Class

import Data.List (foldl', sort)

data Player = X | O deriving (Show, Ord, Eq, Enum, Read)

type Grid   = (Player, [Maybe Player])
data Result = Undecided | Tie | Win Player deriving Show

empty :: Grid
empty = (X, take 9 $ repeat Nothing)

chunksOf :: Int -> [a] -> [[a]]
chunksOf n [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

rows :: [a] -> [[a]]
rows xs      = rs ++
               map (\i -> map (!!i) rs) [0,1,2] ++
               pure (zipWith (!!) rs [0,1,2]) ++
               pure (zipWith (!!) rs [2,1,0])
    where
        rs = chunksOf 3 xs
                

allSame :: Eq a => [a] -> (Bool, Maybe a)
allSame []     = (True, Nothing)
allSame (x:xs) = case all (==x) xs of
                   True  -> (True, Just x)
                   False -> (False, Nothing)

isNothing :: Maybe a -> Bool
isNothing Nothing = True
isNothing _       = False

gridResult :: Grid -> Result
gridResult (_, xs)
        | not (null winners) = Win $ head winners
        | undecided          = Undecided
        | otherwise          = Tie
    where
        rs = rows xs
        rresults = map rowResult rs
        winners = map (\(Just x) -> x) $
                  filter (not . isNothing) $
                  map (\res -> case res of
                                 Win p -> Just p
                                 _     -> Nothing)
                  rresults
        undecided = any (\res -> case res of
                                 Undecided -> True
                                 _         -> False)
                  rresults


rowResult :: [Maybe Player] -> Result
rowResult r = case any isNothing r of
                True  -> Undecided
                False -> case allSame r of
                           (True, Just (Just p)) -> Win p
                           _                     -> Tie

emptyFields :: [Maybe a] -> [Int]
emptyFields xs = map snd $
                 filter (isNothing . fst) $
                 zip xs [0..]

move :: Int -> Grid -> (Grid, Result)
move m (p, xs) = ((nextPlayer, nextGrid), res)
    where
        nextPlayer = case p of
                       X -> O
                       O -> X
        empties = emptyFields xs
        numEmpties = length empties
        m' = m `mod` numEmpties
        pos = empties !! m'
        nextGrid = take pos xs ++ [Just p] ++ drop (pos+1) xs
        res = gridResult (nextPlayer, nextGrid)

moveAI :: DenseData -> Grid -> (Grid, Result)
moveAI net g = move m g
    where
        l = fromGrid g
        m = predict (net, [relu, softmax]) l

battle :: DenseData -> DenseData -> Maybe Player
battle pX pO = battle' pX pO empty

battle' :: DenseData -> DenseData -> Grid -> Maybe Player
battle' pX pO g@(p, _) = case moveAI ai g of
                            (_, Tie)        -> Nothing
                            (_, Win w)      -> Just w
                            (g', Undecided) -> battle' pX pO g'
    where
        ai = case p of
                X -> pX
                O -> pO

beats :: DenseData -> DenseData -> Bool
beats p1 p2 = case w1 of
                Just X -> case w2 of
                            Nothing -> True
                            Just O  -> True
                            _       -> False
                _      -> False
    where
        w1 = battle p1 p2
        w2 = battle p2 p1

type Layer  = HM.Vector HM.R
type Bias   = HM.Vector HM.R
type Weight = HM.Matrix HM.R
type Activation = Layer -> Layer

type DenseData = [(Bias, Weight)]
type DenseNet  = (DenseData, [Activation])

randBias :: RandomGen g => Int -> Rand g Bias
randBias n = (pure . HM.fromList . take n) =<< getRandomRs (-1, 1)

randWeight :: RandomGen g => Int -> Int -> Rand g Weight
randWeight m n = (pure . (m HM.>< n)) =<< getRandomRs (-1, 1)

repeatM :: Monad m => Int -> m a -> m [a]
repeatM 0 m = pure []
repeatM n m =
    do
        x  <- m
        xs <- repeatM (n-1) m
        pure (x:xs)

randPlayer :: RandomGen g => Rand g DenseData
randPlayer =
  do
    b1 <- randBias 18
    b2 <- randBias 9
    w01 <- randWeight 18 18
    w12 <- randWeight 9 18
    return [(b1, w01), (b2, w12)]

fromGrid :: Grid -> Layer
fromGrid (_, xs) = HM.fromList $ lX ++ lO
    where
        lX = map (\x -> case x of
                          Just X -> 1
                          _      -> 0)
             xs
        lO = map (\x -> case x of
                          Just O -> 1
                          _      -> 0)
             xs

dense :: Bias -> Weight -> Activation -> Layer -> Layer
dense b w a l = a $ b + (w Matrix.#> l)

relu :: Activation
relu = HM.cmap relu'
    where
        relu' x = if x < 0 then 0 else x

softmax :: Activation
softmax l = HM.cmap (/s) exps
    where
        m    = HM.maxElement l
        exps = HM.cmap (\x -> exp (x - m)) l
        s    = sum $ HM.toList exps

runDenseNet :: DenseNet -> Layer -> Layer
runDenseNet (dat, acts) l =
    foldl' (\l' ((b, w), a) -> dense b w a l') l $ zip dat acts

predict :: DenseNet -> Layer -> Int
predict net = HM.maxIndex . runDenseNet net

someFunc :: IO ()
someFunc =
  do
    ps <- evalRandIO $ repeatM 1000 randPlayer
    let l = fmap (\x -> length $ filter id $ fmap (beats x) ps) ps
    let l' = sort l
    mapM_ print l'
