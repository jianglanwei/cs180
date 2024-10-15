import matplotlib.pyplot as plt

class animator:
    """animator class that presents live graph."""
    def __init__(self, xlabel: str = None, ylabel: str = None) -> None:
        self.X = []
        self.Y = []
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def add(self, x: float, y: float, show_value: bool = False):
        self.X.append(x)
        self.Y.append(y)
        plt.clf()
        plt.plot(self.X, self.Y)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if show_value:
            for x, y in zip(self.X, self.Y):
                plt.text(x, y, '%.2f' % y, ha = 'center', va = 'bottom')
        plt.pause(0.05)
    
    def stay(self):
        plt.show()