import sys

import torch
from torchvision.transforms import autoaugment, transforms, ColorJitter
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import bcos.data.transforms as custom_transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class CLIPBcosImageNetClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
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
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))
        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms.transforms)})"

    def __rich_repr__(self):
        # https://rich.readthedocs.io/en/stable/pretty.html
        yield "transforms", self.transforms.transforms

    def __to_config__(self):
        """See bcos.experiments.utils.sanitize_config for details."""
        result = dict(
            transform=repr(self),
            **self.args,
        )
        result["interpolation"] = str(result["interpolation"])
        return result


class CLIPBcosImageNetClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        interpolation=InterpolationMode.BILINEAR,
    ):
        self.args = get_args_dict(ignore=["mean", "std"])
        trans = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
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
        """See bcos.experiments.utils.sanitize_config for details."""
        result = dict(
            transform=repr(self),
            **self.args,
        )
        result["interpolation"] = str(result["interpolation"])
        return result

    
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
        # https://rich.readthedocs.io/en/stable/pretty.html
        yield "transforms", self.transforms.transforms

    def __to_config__(self):
        """See bcos.experiments.utils.sanitize_config for details."""
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
        """See bcos.experiments.utils.sanitize_config for details."""
        result = dict(
            transform=repr(self),
            **self.args,
        )
        result["interpolation"] = str(result["interpolation"])
        return result

class UCF101ClassificationPresetTrain:
    def __init__(
        self,
        crop_size=224,
        min_scale=256,          # short side lower bound
        max_scale=320,          # short side upper bound
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

        # torchvision ColorJitter works on tensors shaped [..., C, H, W]
        # so [T, C, H, W] is fine, and it applies the same sampled params
        # across the whole clip in one call.
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

        # 1) random resize on short side
        short_side = torch.randint(self.min_scale, self.max_scale + 1, (1,)).item()
        video = self._resize_short_side(video, short_side)

        # 2) random crop, same crop for all frames
        T, C, H, W = video.shape
        crop_size = self.crop_size

        if H < crop_size or W < crop_size:
            # safety fallback
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

        # 3) random horizontal flip, same for all frames
        if torch.rand(1).item() < 0.5:
            video = torch.flip(video, dims=[3])

        # 4) light color jitter, same sampled params for the whole clip
        if self.do_color_jitter and torch.rand(1).item() < 0.8:
            video = self.color_jitter(video)

        # clamp after jitter
        video = video.clamp(0.0, 1.0)

        # 5) B-cos channel expansion if needed
        if self.is_bcos:
            video = self.add_inv(video)  # expected: [T, 6, H, W]
        else:
            # standard path: normalize here if you are NOT normalizing in-model
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

        print("Video shape: ", video.shape)
        print(video[0].max())
        print(video[0].mean())
        if self.is_bcos:
            video = self.add_inv(video)
            print("Video shape: ", video.shape)
            print(video[0].max())
            print(video[0].mean())
            sys.exit()
        else:
            mean = torch.tensor(self.mean, device=video.device).view(1, -1, 1, 1)
            std = torch.tensor(self.std, device=video.device).view(1, -1, 1, 1)
            video = (video - mean) / std

        video = video.permute(1, 0, 2, 3).contiguous()
        return video


CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)


class CIFAR10ClassificationPresetTrain:
    def __init__(
        self,
        *,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        is_bcos=False,
        normalise_function=transforms.Normalize
    ):
        trans = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            custom_transforms.AddInverse()
            if is_bcos
            else normalise_function(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms


class CIFAR10ClassificationPresetTest:
    def __init__(
        self,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        is_bcos=False,
        normalise_function=transforms.Normalize
    ):
        trans = [
            transforms.ToTensor(),
            custom_transforms.AddInverse()
            if is_bcos
            else normalise_function(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms


def get_args_dict(ignore: "tuple | list" = tuple()):
    """Helper for saving args easily."""
    import inspect

    frame = inspect.currentframe().f_back
    av = inspect.getargvalues(frame)
    ignore = tuple(ignore) + ("self", "cls")
    return {arg: av.locals[arg] for arg in av.args if arg not in ignore}

class VOCClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),
            custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms

class VOCClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            # transforms.Resize((224, 224)), # For reproducing Model Guiding Paper
            custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms
