import os


 if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
    if rank == 0:
        print(f"{checkpoint['model']}")
    model.load_state_dict(checkpoint['model'])