import matplotlib.pyplot as plt

class animator:

    """a two-plot animator class."""

    def __init__(self, xlabel: str, y1label: str, y2label: str) -> None:
        self.X = []
        self.Y1 = []
        self.Y2 = []
        self.xlabel = xlabel
        self.y1label = y1label
        self.y2label = y2label


    
    def add(self, x: float, y1: float, y2: float):
        """add data to graph."""
        self.X.append(x)
        self.Y1.append(y1)
        self.Y2.append(y2)


    def save(self, path: str):
        """save graph to path."""
        plt.clf()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.set_xlabel(self.xlabel)
        ax1.set_ylabel(self.y1label, color='b')
        ax2.set_ylabel(self.y2label, color='r')
        l1, = ax1.plot(self.X, self.Y1, color='b')
        l2, = ax2.plot(self.X, self.Y2, color='r')
        plt.legend(handles=[l1, l2], labels=[self.y1label, self.y2label])
        plt.savefig(path)