import train

# Pretrain a model with the output of a number of simulated annealing
# proceedures.

if __name__ == "__main__":
    args = train.get_args()

    with open('graphs.pkl', 'rb') as file:
        graphs = pickle.load(file)

    model = 
    preproc = PreInput(args)