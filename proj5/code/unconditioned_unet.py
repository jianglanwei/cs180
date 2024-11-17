from module import *


NUM_EPOCHS = 5 # number of training epochs.
LR = 1e-4 # learning rate.
BATCH_SIZE = 256 # training batch size.
NUM_INPUT = 1 # number of input layers. input size: [N, 1, 28, 28].
NUM_HIDDEN = 256 # number of hidden layers.
TRAIN_MODE = True # train UNet when true; if else, load UNet to denoise test image.

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

def add_noise(img: torch.Tensor, alpha: float) -> torch.Tensor:
    """noised_image = original_image + alpha * pure_noise."""
    return img + torch.randn_like(img) * alpha

def spread_imgs(imgs: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """spread 4d image tensor into a 2d image.
       input size: [N, 1, H, W]; 
       output size:[H * num_rows, W * num_cols]"""
    assert num_rows * num_cols == imgs.shape[0], "number of images doesn't match."
    chunks = torch.chunk(imgs.reshape(-1, imgs.shape[3]), num_rows*num_cols)
    rows = [torch.cat(chunks[row*num_cols:(row+1)*num_cols], dim=1) for row in range(num_rows)]
    out_img = torch.cat(rows, dim=0)
    return out_img

train_loader, test_loader = load_data(BATCH_SIZE)
if TRAIN_MODE: # train UNet.
    net = UnconditionalUNet(in_channels=NUM_INPUT, num_hiddens=NUM_HIDDEN).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR) # type: ignore
    grf = animator(xlabel="step", ylabel="loss")
    torchsummary.summary(net, (1, 28, 28))
    step = 0
    for epoch in range(NUM_EPOCHS):
        for img, label in train_loader:
            img = img.cuda() # load image to gpu.
            noisy_img = add_noise(img, alpha=0.5)
            optimizer.zero_grad()
            pred_img = net(noisy_img)
            loss = (img - pred_img) ** 2
            loss.mean().backward()
            optimizer.step()
            grf.add(step, loss.mean().item())
            grf.save("media/uncond_loss_graph.pdf") # save loss curve to file.
            print(f"epoch {epoch} step {step}: loss {loss.mean().item(): .4f}")
            step += 1
        torch.save(net, "uncond_nets/uncond_net" + str(epoch) + ".pth")
else:
    net = torch.load("uncond_nets/uncond_net4.pth").cpu() # load UNet
    NUM_TEST_SAMPLES = 4 # number of test samples.
    for img, label in test_loader:
        img = img[:NUM_TEST_SAMPLES]
        noisy_img = add_noise(img, 0.5)
        pred_img = net(noisy_img).detach()
        out_img = spread_imgs(torch.cat((img, noisy_img, pred_img), dim=0), 
                              num_rows=3, num_cols=NUM_TEST_SAMPLES)
    plt.imsave("media/uncond_denoise.png", out_img, cmap='gray')