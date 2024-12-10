import torch
from torch import nn
import matplotlib.pyplot as plt
from animator import animator
import torchsummary

NUM_INPUT_FREQ_LEVELS = 10
NUM_HIDDEN = 256
BATCH_SIZE = 10_000
NUM_STEPS = 2000
LR = 1e-2
SAVE_FREQ = 50

ROOT_DIR = ""
NET_DIR = "2d_nets/"
GEN_IMG_DIR = "pred_2d_imgs/"
IMG_PATH = "animal.jpg"

TRAIN_MODE = True


def sinusoidal_position(pos: torch.Tensor, L: int) -> torch.Tensor:
    """conduct Sinusoidal Positional Encoding (PE) on given position.
    - pos: original position wth shape [N, D].
    - L: highest frequency level.
    - output shape: [N, (2 * L + 1) * D]."""
    out_pos = torch.zeros((pos.shape[0], (2 * L + 1), pos.shape[1]))
    out_pos[:, 0] = pos
    weights = 2 ** torch.arange(L).reshape(1, -1, 1)
    out_pos[:, 1::2] = torch.sin(weights * torch.pi * pos.unsqueeze(1))
    out_pos[:, 2::2] = torch.cos(weights * torch.pi * pos.unsqueeze(1))
    out_pos = out_pos.reshape(pos.shape[0], -1)
    return out_pos

def psnr(mse: torch.Tensor):
    """calculate the Peak Signal-to-Noise Ratio (PSNR) given 
       the input Mean Square Error (MSE).
       - mse: Mean Square Error with shape [N].
       - output shape: [N]."""
    return -10 * torch.log10(mse)


class img_iter:
    """an image iterator class that returns random sinusoidal positions and 
       their desired color output."""
    def __init__(self, img, batch_size: int, num_steps: int):
        """img pixels should be inside the range of [0, 1]"""
        self.img = torch.tensor(img)
        self.num_steps = num_steps
        self.current_step = 0
        self.batch_size = batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """returns sinusoidal positional input and pixel color."""
        if self.current_step == self.num_steps: raise StopIteration
        rand_rows = torch.randint(0, self.img.shape[0], (self.batch_size,))
        rand_cols = torch.randint(0, self.img.shape[1], (self.batch_size,))
        values = self.img[rand_rows, rand_cols].squeeze()
        rand_pos = torch.stack((rand_rows / self.img.shape[0], 
                                rand_cols / self.img.shape[1]), dim=1) # TODO
        pe_pos = sinusoidal_position(rand_pos, NUM_INPUT_FREQ_LEVELS)
        self.current_step += 1
        return pe_pos.float(), values.float()

img = plt.imread(IMG_PATH)[:,:,:3]
img = img / img.max()

if TRAIN_MODE:
    net = nn.Sequential(
        nn.Linear(4*NUM_INPUT_FREQ_LEVELS+2, NUM_HIDDEN),
        nn.ReLU(),
        nn.Linear(NUM_HIDDEN, NUM_HIDDEN),
        nn.ReLU(),
        nn.Linear(NUM_HIDDEN, NUM_HIDDEN),
        nn.ReLU(),
        nn.Linear(NUM_HIDDEN, 3),
        nn.Sigmoid()
    ).cuda()
    train_iter = img_iter(img, BATCH_SIZE, NUM_STEPS)
    loss_func = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR) # type: ignore
    grf = animator(xlabel="Step", y1label="MSE", y2label="PSNR")
    torchsummary.summary(net, input_size=(BATCH_SIZE, 4*NUM_INPUT_FREQ_LEVELS+2))
    for step, (X, y) in enumerate(train_iter):
        X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        y_hat = net(X)
        loss = loss_func(y_hat, y).mean()
        loss.backward()
        optimizer.step()
        psnr_ = psnr(loss).item()
        grf.add(step, loss.item(), psnr_)
        grf.save(ROOT_DIR + "loss_graph.pdf")
        print(f"Step {step}: MSE: {loss.item(): .4f}; PSNR: {psnr_: .4f}.")
        if step % SAVE_FREQ == 0:
            torch.save(net, NET_DIR + f"net{step}.pth")
            torch.save(grf, ROOT_DIR + "loss_grf.pth")
            print(f"trained net saved at step {step}.")
else:
    MODEL_NO = 1950 # which trained net to use.
    with torch.no_grad():
        net = torch.load(NET_DIR + f"net{MODEL_NO}.pth")
        img = torch.Tensor(img)
        pred_img = torch.zeros(img.shape).cuda()
        for row in range(img.shape[0]): # predict img by row due to gpu memory limit.
            pos = torch.stack(((torch.ones(img.shape[1]) * row / img.shape[0]), 
                            (torch.arange(img.shape[1]) / img.shape[1])), dim=1)
            pe_pos = sinusoidal_position(pos, NUM_INPUT_FREQ_LEVELS).cuda()
            pred_values = net(pe_pos)
            pred_img[row] = pred_values
    plt.imsave(GEN_IMG_DIR + "pred_img.png", pred_img.cpu().numpy())