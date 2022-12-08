import argparse
import glob
import matplotlib.pyplot as plt


def get_name_from_file(f):
    if 'DarkNet' in f:
        return 'DarkNet'
    elif 'bzip' in f:
        return 'bzip'
    elif 'ffmpeg' in f:
        return 'ffmpeg'
    elif 'livermorec' in f:
        return 'LivermoreC'
    elif 'freeimage' in f:
        return 'freeimage'
    assert False

def load_data_from(f):
    rewards = []
    with open(f) as f:
        for line in f.readlines():
            if line.startswith('Episode reward mean'):
                if 'nan' in line:
                    continue
                rewards.append(line.split(' ')[5])

        for i in range(len(rewards)):
            rewards[i] = float(rewards[i])

    return rewards

def load_annealer_data_from(f):
    best_score = 1000000000
    with open(f) as f:
        for line in f.readlines():
            if line.startswith("Computed IIs"):
                line = line.replace('[', '').replace(',', '').replace(']', '')
                scores = []
                score = 0
                for item in line.split(' ')[2:]:
                    if not item:
                        continue
                    try:
                        item = int(item)
                        # Need this crap to keep the scoring consistent
                        if item > 1000:
                            item = max(scores)
                        score += item
                        scores.append(item)
                    except:
                        print("Skipping item", item)
                        pass
                if score < best_score:
                    best_score = score
    return best_score

def plot_lines(lines, annealer_data, names):
    # names = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6", "Line 7", "Line 8", "Line 9"]
    colors = ['red', 'blue', 'orange', 'grey', 'green', 'purple', 'pink', 'black', 'brown']
    max_so_far = 0

    for i in range(len(lines)):
        data = lines[i]
        name = names[i]

        max_x = len(data)
        if max_x > max_so_far:
            max_so_far = max_x

        plt.plot(range(len(data)), data, label=get_name_from_file(name), color=colors[i])

    print("Annearler len is ")
    print(len(annealer_data))
    print(annealer_data)
    for i in range(len(annealer_data)):
        data = annealer_data[i]
        name = names[i]
        color = colors[i]
        if data > 1000000000:
            print("Skipping", name, "because annealer didn't give any decent results I guess")
            continue
        plt.plot([0, max_so_far], [-data, -data], color=color, linestyle='--')

    plt.xlabel('Epochs')
    plt.xlim([0, 100])
    plt.legend([get_name_from_file(n) for n in names], ncol=3, bbox_to_anchor=(0.05, 1.05))
    plt.ylabel('Architecture Performance (Average)')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.savefig('training_curves.eps')
    print("Saved fig")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_files')
    parser.add_argument('annealer_files')

    args = parser.parse_args()

    data = []
    files = []
    for f in glob.glob(args.output_files):
        files.append(f)
        data.append(load_data_from(f))

    annealer = []
    for f in glob.glob(args.annealer_files):
        print("Loading ", f)
        annealer.append(load_annealer_data_from(f))

    plot_lines(data, annealer, names=files)