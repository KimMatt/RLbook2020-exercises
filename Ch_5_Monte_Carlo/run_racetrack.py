# racetrack.py
#
# implementation of racetrack problem from chapter 5
import pickle

from src.learner import Learner

if __name__ == "__main__":

    learner = Learner()

    for i in range(20000):
        print("ROUND {}".format(i))
        learner.play_round()

    pickle.dump(learner, open("./pickles/race_learner.p", "wb"))
