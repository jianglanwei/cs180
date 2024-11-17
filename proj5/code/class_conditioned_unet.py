from module import *


NUM_EPOCHS = 20 # number of training epochs.
LR = 1e-3 # initial learning rate.
BATCH_SIZE = 128 # training batch size.
NUM_INPUT = 1 # number of input layers. input size: [N, 1, 28, 28].
NUM_HIDDEN = 128 # number of hidden layers.
BETA1 = 1e-4 # lower bound for beta.
BETA2 = 0.02 # upper bound for beta.
NUM_TS = 300 # number of denoise steps. T reflects how noisy the image is.
BI_THRESHOLD = 0.3 # pixels with value under 0.3 are snapped to 0 in the output.
TRAIN_MODE = True # train UNet when true; if else, generate MNIST images using UNet.
DROPOUT_PROB = 0.1 # dropout probability.
GAMMA = 5. # for classifier free guidance.

def load_data(batch_size: int):
    """load data from MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                  download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def ddpm_schedule(beta1: float, beta2: float, num_ts: int) -> dict:
    """Constants for DDPM training and sampling."""
    assert beta1 < beta2 < 1.0, "Expect beta1 < beta2 < 1.0."
    betas = torch.linspace(beta1, beta2, steps=num_ts)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
    }

def one_hot_vec(labels: torch.Tensor, dropout_prob: float = 0) -> torch.Tensor:
    """get one-hot vector based on input labels.
       - input shape:  [N];
       - output shape: [N, 1, 1, 10];
       - when dropped, vector has an all-zero population."""
    one_hot = torch.zeros((len(labels), 1, 1, 10))
    one_hot[torch.arange(len(labels)), :, :, labels] = 1
    drop_indices = torch.randperm(labels.shape[0])[:int(labels.shape[0]*dropout_prob)]
    one_hot[drop_indices] = 0
    return one_hot

def add_noise(img: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor) -> tuple:
    """returns noisy image and pure noise."""
    noise = torch.randn_like(img)
    noisy_img = torch.sqrt(alpha_bars[t]) * img + torch.sqrt(1 - alpha_bars[t]) * noise
    return noisy_img, noise

def spread_imgs(imgs: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """spread 4d image tensor into a 2d image.
       input size: [N, 1, H, W]; 
       output size:[H * num_rows, W * num_cols]"""
    assert num_rows * num_cols == imgs.shape[0], "number of images doesn't match."
    chunks = torch.chunk(imgs.reshape(-1, imgs.shape[3]), num_rows*num_cols)
    rows = [torch.cat(chunks[row*num_cols:(row+1)*num_cols], dim=1) for row in range(num_rows)]
    out_img = torch.cat(rows, dim=0)
    return out_img

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

train_loader, test_loader = load_data(BATCH_SIZE)
ddpm_consts = ddpm_schedule(beta1=BETA1, beta2=BETA2, num_ts=NUM_TS+1)
if TRAIN_MODE: # train UNet with class conditioning
    net = ClassConditionalUNet(in_channels=1, num_hiddens=NUM_HIDDEN).cuda()
    net.apply(init_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR) # type: ignore
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    grf = animator(xlabel="step", ylabel="loss")
    alpha_bars = ddpm_consts['alpha_bars'].reshape(-1, 1, 1, 1).cuda()
    torchsummary.summary(net, input_size=[(1, 28, 28), (1, 1, 1), (1, 1, 10)])
    step = 0
    for epoch in range(NUM_EPOCHS):
        for img, label in train_loader:
            img = img.cuda()
            t = torch.randint(1, NUM_TS+1, (img.shape[0],)).cuda()
            noisy_img, noise = add_noise(img, t, alpha_bars)
            optimizer.zero_grad()
            t = (t/NUM_TS).reshape(-1, 1, 1, 1) # normalize t.
            c = one_hot_vec(label, dropout_prob=DROPOUT_PROB).cuda()
            pred_noise = net(noisy_img, t, c)
            loss = (noise - pred_noise) ** 2
            loss.mean().backward()
            optimizer.step()
            grf.add(step, loss.mean().item())
            grf.save("cs180/class_cond_loss_graph.pdf")
            print(f"epoch {epoch} step {step}: loss {loss.mean().item(): .4f}")
            step += 1
        torch.save(net, "cs180/class_cond_nets/class_cond_net" + str(epoch) + ".pth")
        torch.save(grf, "cs180/class_conditioned_grf.pth")
        scheduler.step()
else: # load trained UNet and generate MNIST image.
    net = torch.load("cs180/class_cond_nets/class_cond_net4.pth", weights_only=False).cpu()
    alphas = ddpm_consts['alphas']
    betas = ddpm_consts['betas']
    alpha_bars = ddpm_consts['alpha_bars']
    NUM_ROWS = 6 # number of rows of generated MNIST img.
    img = torch.randn((10*NUM_ROWS, 1, 28, 28))
    label = torch.arange(10).repeat(NUM_ROWS)
    for t in range(NUM_TS, 0, -1):
        z = torch.randn((10*NUM_ROWS, 1, 28, 28)) if t > 1 else 0
        t = t * torch.ones(10*NUM_ROWS, 1, 1, 1, dtype=torch.int)
        c = one_hot_vec(label)
        cond_pred_noise = net(img, t/NUM_TS, c).detach()
        uncond_pred_noise = net(img, t/NUM_TS, torch.zeros(c.shape)).detach()
        pred_noise = uncond_pred_noise + GAMMA * (cond_pred_noise - uncond_pred_noise)
        clean_img = (img - torch.sqrt(1 - alpha_bars[t]) * pred_noise) \
                  / torch.sqrt(alpha_bars[t])
        img = torch.sqrt(alpha_bars[t-1]) * betas[t] / (1 - alpha_bars[t]) * clean_img \
            + torch.sqrt(alphas[t]) * (1 - alpha_bars[t-1]) / (1 - alpha_bars[t]) * img \
            + torch.sqrt(betas[t]) * z
    spreaded_img = spread_imgs(img, num_rows=NUM_ROWS, num_cols=10).numpy()
    out_img = (spreaded_img > BI_THRESHOLD) * spreaded_img
    plt.imsave('cs180/class_cond_generated_img4.png', out_img, cmap='gray')