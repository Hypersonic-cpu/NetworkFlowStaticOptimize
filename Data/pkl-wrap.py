import pickle

# Load object from .pkl file
with open('all_shortest_path.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

for elem in loaded_data:
    print(elem)