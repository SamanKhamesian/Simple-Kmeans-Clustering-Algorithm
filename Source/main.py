from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import MinMaxScaler

from Source.kmeans import KMeans

N_SAMPLE = 200
NOISE_RATIO = 0.1


def normalize_data(x):
    normalizer = MinMaxScaler()
    return normalizer.fit_transform(x)


def plot_dataset(x, y, centers=None):
    clusters = set(y)
    if centers is None:
        centers = [[0, 0]]

    plt.xlabel('Feature 01')
    plt.ylabel('Feature 02')
    plt.title('Our Dataset')

    for c in clusters:
        plt.scatter(x=x[y == c, 0],
                    y=x[y == c, 1],
                    edgecolors='black',
                    label='Class ' + str(c + 1))

    for c in centers:
        plt.scatter(x=c[0],
                    y=c[1],
                    c='black',
                    s=75,
                    marker='x',
                    linewidths='10',
                    edgecolors='black'
                    )

    plt.legend()
    plt.plot()
    plt.show()


def make_dataset(choice):
    if choice == 1:
        x, y = make_moons(n_samples=N_SAMPLE, noise=NOISE_RATIO)
    elif choice == 2:
        x, y = make_circles(n_samples=N_SAMPLE, noise=NOISE_RATIO, random_state=True)
    elif choice == 3:
        x, y = make_blobs(n_samples=N_SAMPLE, n_features=2, centers=2, random_state=False, shuffle=True, cluster_std=0.5)
    elif choice == 4:
        x, y = make_blobs(n_samples=N_SAMPLE, n_features=2, centers=3, random_state=False, shuffle=True, cluster_std=0.5)
    elif choice == 5:
        x, y = make_blobs(n_samples=N_SAMPLE, n_features=2, centers=4, random_state=False, shuffle=True, cluster_std=0.5)
    else:
        x, y = make_moons(n_samples=N_SAMPLE, noise=NOISE_RATIO)

    return x, y


def show_menu():
    print('Here are our sample dataset :')
    choices = ['1- Moons', '2- Circles', '3- Two Blobs', '4- Three Blobs', '5- Four Blobs']
    for option in choices:
        print(option)
    selected = int(input('\nWhich dataset do you want to test? (Type the number) : ', ))
    return selected


if __name__ == '__main__':
    dataset_name = show_menu()
    X, Y = make_dataset(dataset_name)
    X = normalize_data(X)
    model = KMeans(k=len(set(Y)))
    model.fit(X)
    plot_dataset(X, model.get_labels(), model.get_centers())
