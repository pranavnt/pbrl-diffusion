from sac import SACAgent
import pickle

open_rollouts = pickle.load(open("./rollouts/rollouts_open.pkl", "rb"))

print(len(open_rollouts))
