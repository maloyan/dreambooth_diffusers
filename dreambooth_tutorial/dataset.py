from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if instance_image.mode != "RGB":
            instance_image = instance_image.convert("RGB")
        return {
            "instance_images": self.image_transforms(instance_image),
            "instance_prompt_ids": self.tokenizer(
                self.instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0],
        }


class DreamBoothImageDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
    ):
        self.dataset = load_dataset("maloyan/vqgan1024_reconstruction", split="train[:1000]")
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        image_256 = self.dataset[index]["image_256"]
        reconstruction_256 = self.dataset[index]["reconstruction_256"]

        return {
            "image_256": self.image_transforms(image_256),
            "reconstruction_256": self.image_transforms(reconstruction_256)
        }
