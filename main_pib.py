import os
import torch
import torch.optim as optim
import logging
from ssl_models import SimCLR, MoCo, BYOL, create_backbone
from pib_sample import generate_mini_batch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from argparse import ArgumentParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'), train=True, download=True, transform=transform)
    elif args.dataset_name == 'Image-100':
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'Image-100'), transform=transform)
    elif args.dataset_name == 'ImageNet':
        train_dataset = datasets.ImageNet(root=os.path.join(args.data_dir, 'ImageNet'), split='train', download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    backbone = create_backbone('resnet50', pretrained=True).to(device)
    
    if args.model_type == 'SimCLR':
        model = SimCLR(backbone, projection_dim=args.projection_dim).to(device)
    elif args.model_type == 'MoCo':
        model = MoCo(backbone, projection_dim=args.projection_dim).to(device)
    elif args.model_type == 'BYOL':
        model = BYOL(backbone, projection_dim=args.projection_dim).to(device)
    else:
        raise ValueError("Invalid model type. Choose between 'SimCLR', 'MoCo', 'BYOL'.")
    
    # Set up optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)

            mini_batch = generate_mini_batch(train_dataset, model, args.a, args.batch_size)
            x_plus, xlabel = zip(*mini_batch)

            x_plus = torch.stack(x_plus).to(device)
            xlabel = torch.stack(xlabel).to(device)

            if args.model_type == 'SimCLR':
                z_i = model(x)
                z_j = model(x_plus)
                loss = model.contrastive_loss(z_i, z_j)
            elif args.model_type == 'MoCo':
                z_q, z_k = model(x)
                loss = model.contrastive_loss(z_q, z_k)
            elif args.model_type == 'BYOL':
                z1, z2 = model(x, x_plus)
                loss = model.loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                logging.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0
        
        if args.model_type == 'BYOL' or args.model_type == 'MoCo':
            model.update_target_encoder()
        
        logging.info(f"End of Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    logging.info("Training Completed!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for SSL models with PIB sampling")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--model_type', type=str, choices=['SimCLR', 'MoCo', 'BYOL'], default='SimCLR', help='SSL model type')
    parser.add_argument('--projection_dim', type=int, default=128, help='Dimension of the projection head')
    parser.add_argument('--a', type=int, default=64, help='Mini-batch size for PIB sampling')
    parser.add_argument('--dataset_name', type=str, choices=['CIFAR10', 'Image-100', 'ImageNet'], default='CIFAR10', help='Dataset name')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size for model')
    parser.add_argument('--data_dir', type=str, default='./data/mltasa', help='Root directory for datasets')

    args = parser.parse_args()

    main(args)
