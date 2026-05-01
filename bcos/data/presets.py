import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment, ColorJitter
from torchvision.transforms.functional import InterpolationMode

import bcos.data.transforms as custom_transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# My Code:

class UCF101ClassificationPresetTrain:
    def __init__(
        self,
        crop_size=224,
        min_scale=256,
        max_scale=320,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        is_bcos=False,
        do_color_jitter=True,
        color_jitter_strength=0.2,
    ):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mean = mean
        self.std = std
        self.is_bcos = is_bcos
        self.do_color_jitter = do_color_jitter

        self.add_inv = custom_transforms.AddInverse()

        self.color_jitter = ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=min(0.05, color_jitter_strength / 4),
        )

    def _resize_short_side(self, video: torch.Tensor, short_side: int) -> torch.Tensor:
        """
        video: [T, C, H, W]
        returns: [T, C, H_new, W_new]
        """
        T, C, H, W = video.shape
        if H < W:
            new_h = short_side
            new_w = int(round(W * short_side / H))
        else:
            new_w = short_side
            new_h = int(round(H * short_side / W))

        return F.interpolate(
            video,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [T, H, W, C], uint8 or float-like
        returns: [C, T, H, W]
        """
        # to [0,1]
        video = video.float() / 255.0

        # [T, H, W, C] -> [T, C, H, W]
        video = video.permute(0, 3, 1, 2)

        # random resize
        short_side = torch.randint(self.min_scale, self.max_scale + 1, (1,)).item()
        video = self._resize_short_side(video, short_side)

        # random crop
        T, C, H, W = video.shape
        crop_size = self.crop_size

        if H < crop_size or W < crop_size:
            video = F.interpolate(
                video,
                size=(max(H, crop_size), max(W, crop_size)),
                mode="bilinear",
                align_corners=False,
            )
            T, C, H, W = video.shape

        top = torch.randint(0, H - crop_size + 1, (1,)).item()
        left = torch.randint(0, W - crop_size + 1, (1,)).item()
        video = video[:, :, top:top + crop_size, left:left + crop_size]

        # random horizontal flip
        if torch.rand(1).item() < 0.5:
            video = torch.flip(video, dims=[3])

        # color jitter
        if self.do_color_jitter and torch.rand(1).item() < 0.8:
            video = self.color_jitter(video)

        # clamp after jitter
        video = video.clamp(0.0, 1.0)

        # B-cos channel inv added
        # For B-Cos models, normalisation is done in model, instead of here
        if self.is_bcos:
            video = self.add_inv(video)  # expected: [T, 6, H, W]
        else:
            mean = torch.tensor(self.mean, device=video.device).view(1, -1, 1, 1)
            std = torch.tensor(self.std, device=video.device).view(1, -1, 1, 1)
            video = (video - mean) / std

        # [T, C, H, W] -> [C, T, H, W]
        video = video.permute(1, 0, 2, 3).contiguous()
        return video


class UCF101ClassificationPresetEval:
    def __init__(
        self,
        crop_size=224,
        resize_short_side=256,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        is_bcos=False,
    ):
        self.crop_size = crop_size
        self.resize_short_side = resize_short_side
        self.mean = mean
        self.std = std
        self.is_bcos = is_bcos
        self.add_inv = custom_transforms.AddInverse()

    def _resize_short_side(self, video: torch.Tensor, short_side: int) -> torch.Tensor:
        T, C, H, W = video.shape
        if H < W:
            new_h = short_side
            new_w = int(round(W * short_side / H))
        else:
            new_w = short_side
            new_h = int(round(H * short_side / W))

        return F.interpolate(
            video,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [T, H, W, C]
        returns: [C, T, H, W]
        """
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

        video = self._resize_short_side(video, self.resize_short_side)

        T, C, H, W = video.shape
        crop_size = self.crop_size
        top = max((H - crop_size) // 2, 0)
        left = max((W - crop_size) // 2, 0)
        video = video[:, :, top:top + crop_size, left:left + crop_size]

        video = video.clamp(0.0, 1.0)
        if self.is_bcos:
            video = self.add_inv(video)
        else:
            mean = torch.tensor(self.mean, device=video.device).view(1, -1, 1, 1)
            std = torch.tensor(self.std, device=video.device).view(1, -1, 1, 1)
            video = (video - mean) / std

        video = video.permute(1, 0, 2, 3).contiguous()
        return video

# Taken from existing B-Cosification Repo - https://github.com/shrebox/B-cosification
# Used for replicating 2D B-Cosification Results:

class ImageNetClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        is_bcos=False,
        normalise_function=transforms.Normalize
    ):
        self.args = get_args_dict(ignore=["mean", "std"])
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(
                    autoaugment.RandAugment(
                        interpolation=interpolation, magnitude=ra_magnitude
                    )
                )
            elif auto_augment_policy == "ta_wide":
                trans.append(
                    autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            elif auto_augment_policy == "augmix":
                trans.append(
                    autoaugment.AugMix(
                        interpolation=interpolation, severity=augmix_severity
                    )
                )
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation
                    )
                )
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        if not is_bcos:
            trans.append(normalise_function(mean=mean, std=std))
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        if is_bcos:
            trans.append(custom_transforms.AddInverse())

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms

    def __to_config__(self):
        result = dict(
            transform=repr(self),
            **self.args,
        )
        result["interpolation"] = str(result["interpolation"])
        return result


class ImageNetClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        is_bcos=False,
        normalise_function=transforms.Normalize
    ):
        self.args = get_args_dict(ignore=["mean", "std"])
        trans = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            custom_transforms.AddInverse()
            if is_bcos
            else normalise_function(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    @property
    def resize(self):
        return self.transforms.transforms[0]

    @property
    def center_crop(self):
        return self.transforms.transforms[1]

    def no_scale(self, img):
        x = img
        for t in self.transforms.transforms[2:]:
            x = t(x)
        return x

    # this is intended for when using the pretrained models
    def transform_with_options(self, img, center_crop=True, resize=True):
        x = img
        if resize:
            x = self.resize(x)
        if center_crop:
            x = self.center_crop(x)
        x = self.no_scale(x)
        return x

    def with_args(self, **kwargs):
        args = self.args.copy()
        args.update(kwargs)
        return self.__class__(**args)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms

    def __to_config__(self):
        result = dict(
            transform=repr(self),
            **self.args,
        )
        result["interpolation"] = str(result["interpolation"])
        return result


def get_args_dict(ignore: "tuple | list" = tuple()):
    """Helper for saving args easily."""
    import inspect

    frame = inspect.currentframe().f_back
    av = inspect.getargvalues(frame)
    ignore = tuple(ignore) + ("self", "cls")
    return {arg: av.locals[arg] for arg in av.args if arg not in ignore}