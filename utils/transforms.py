from PIL import Image
import torchvision.transforms as tt

class NumpyToPIL:
    def __call__(self, frame):
        return Image.fromarray(frame).convert('RGB')

def get_transforms(args, split):
    transform_list = {
        'train': [
            NumpyToPIL(),
            tt.RandomResizedCrop(
                size=(args.img_size, args.img_size),
                scale=(0.9, 1.0),
                ratio=(9 / 10, 10 / 9)),
            tt.RandomHorizontalFlip(),
            tt.RandomApply([tt.RandomRotation(23)], p=0.8),
            tt.RandomApply([tt.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.125)], p=0.8),
            tt.ToTensor()
        ],
        'test':
        [
            NumpyToPIL(),
            tt.Resize((args.img_size, args.img_size)),
            tt.ToTensor()
        ]
    }

    return tt.Compose(transform_list[split])