import matplotlib.pyplot as plt

class animator:
    def __init__(self, xlabel: str, ylabel: str) -> None:
        self.X = []
        self.Y = []
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def add(self, x: float, y: float):
        """add data to graph."""
        self.X.append(x)
        self.Y.append(y)


    def save(self, path: str):
        """save graph to path."""
        plt.clf()
        plt.plot(self.X, self.Y)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.yscale('log')
        plt.savefig(path)