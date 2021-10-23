__author__ = 'ltnghia'

import os
import cv2
import numpy as np
from tqdm import tqdm
import imgaug.augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)


def augmentation_lvl1():
    seq = iaa.SomeOf((1, 2), [
        iaa.SomeOf((1, 3), [
            # Hue
            iaa.OneOf([
                iaa.AddToHue((-30, 30)),
                iaa.Add(value=(-30, 30), per_channel=0.5),
                iaa.Multiply(mul=(0.7, 1.3), per_channel=0.5),
            ]),
            # Saturation
            iaa.OneOf([
                iaa.MultiplySaturation((0.7, 1.3)),
                iaa.AddToSaturation((-30, 30)),
                iaa.Grayscale(alpha=(0.3, 0.7)),
            ]),
            # Brightness
            iaa.OneOf([
                iaa.MultiplyBrightness((0.7, 1.3)),
                iaa.AddToBrightness((-30, 30)),
            ]),
            # Contrast
            iaa.OneOf([
                iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=0.5),
                iaa.LogContrast(gain=(0.6, 1.4), per_channel=0.5),
                iaa.LinearContrast(alpha=(0.5, 2.0), per_channel=0.5),
            ]),
        ], random_order=True),

        iaa.OneOf([
            iaa.MultiplyElementwise(mul=(0.8, 1.2), per_channel=0.5),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0, 20), per_channel=0.5),
            iaa.GaussianBlur(sigma=(0.5, 2.0)),
            iaa.AverageBlur(k=(1, 5)),
            iaa.MedianBlur(k=(1, 5)),
            iaa.Sharpen(alpha=(0.1, 0.4), lightness=(0.75, 1.5)),
        ]),

        iaa.OneOf([
            iaa.imgcorruptlike.JpegCompression(severity=(1, 4)),
            iaa.BlendAlpha((0.4, 0.7), foreground=iaa.imgcorruptlike.Frost(severity=(1, 1)), per_channel=True),
            iaa.BlendAlpha((0.4, 0.7), foreground=iaa.Clouds()),
            iaa.BlendAlpha((0.4, 0.6), foreground=iaa.Fog()),
            iaa.imgcorruptlike.Snow(severity=(1, 1)),
            iaa.Rain(speed=(0.1, 0.3), nb_iterations=(1, 1), drop_size=(0.01, 0.02)),
            iaa.Snowflakes(flake_size=(0.4, 0.7), speed=(0.001, 0.03), density=(0.004, 0.05),
                           density_uniformity=(0.3, 0.7)),
        ]),
    ])
    return seq


def augmentation_lvl2():
    seq = iaa.OneOf([
        iaa.Sequential([
            # ===================
            # Color
            iaa.SomeOf((1, 3),[
                # Hue
                iaa.OneOf([
                    iaa.MultiplyHue((0.5, 2.0)),
                    iaa.AddToHue((-30, 30)),
                    iaa.ChangeColorTemperature((1000, 10000)),
                    iaa.Add(value=(-50, 50), per_channel=0.5),
                    iaa.Multiply(mul=(0.5, 1.5), per_channel=0.5),
                    iaa.pillike.EnhanceColor(factor=(0.0, 3.0)),
                    iaa.pillike.Equalize(),
                ]),
                # Saturation
                iaa.OneOf([
                    iaa.MultiplySaturation((0.5, 1.5)),
                    iaa.AddToSaturation((-50, 50)),
                    iaa.imgcorruptlike.Saturate(severity=(1, 3)),
                    iaa.RemoveSaturation(),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                ]),
                # Brightness
                iaa.OneOf([
                    iaa.MultiplyBrightness((0.5, 1.5)),
                    iaa.AddToBrightness((-50, 50)),
                    iaa.pillike.EnhanceBrightness(factor=(0.5, 1.5)),
                    iaa.imgcorruptlike.Brightness(severity=(1, 3)),
                ]),
                # Contrast
                iaa.OneOf([
                    iaa.GammaContrast(gamma=(0.5, 2.0), per_channel=0.5),
                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=0.5),
                    iaa.LogContrast(gain=(0.6, 1.4), per_channel=0.5),
                    iaa.LinearContrast(alpha=(0.5, 2.0), per_channel=0.5),
                    iaa.pillike.EnhanceContrast(factor=(0.5, 1.5)),
                    iaa.pillike.Autocontrast(cutoff=(10, 20), per_channel=0.5),
                    # iaa.AllChannelsCLAHE(clip_limit=(1, 3), per_channel=0.5), # medium
                    # iaa.imgcorruptlike.Contrast(severity=(1, 3)), # for difficult
                ]),
            ], random_order=True),
            iaa.OneOf([
                # ===================
                # Blur
                iaa.OneOf([
                    iaa.OneOf([
                        iaa.GaussianBlur(sigma=(0.5, 2.0)),
                        iaa.AverageBlur(k=(1, 7)),
                        iaa.MedianBlur(k=(3, 7)),
                        iaa.imgcorruptlike.GaussianBlur(severity=(1, 2)),
                        iaa.pillike.FilterBlur(),
                    ]),
                    iaa.OneOf([
                        iaa.imgcorruptlike.MotionBlur(severity=(1, 2)),
                        iaa.MotionBlur(k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0)),
                        iaa.Alpha((0.01, 0.05), iaa.Canny(alpha=(0, 1)), iaa.MedianBlur((1, 5))),
                        # iaa.imgcorruptlike.GlassBlur(severity=(1, 2)),  # medium
                        # iaa.imgcorruptlike.DefocusBlur(severity=(1, 2)),  # medium
                        # iaa.imgcorruptlike.ZoomBlur(severity=(1, 1))  # hard
                    ]),
                ]),
                # ===================
                # sharpen and edge
                iaa.OneOf([
                    iaa.Sharpen(alpha=(0.1, 0.4), lightness=(0.75, 1.5)),
                    iaa.Canny(alpha=(0.05, 0.2),
                              colorizer=iaa.RandomColorsBinaryImageColorizer(color_true=255, color_false=0)),
                    iaa.Canny(alpha=(0.05, 0.2), sobel_kernel_size=[3, 7]),
                    iaa.Alpha(factor=(0.05, 0.2), first=iaa.Canny(alpha=1), second=iaa.MedianBlur(k=(1, 3)),
                              per_channel=0.5),
                    iaa.BlendAlpha((0.3, 0.6), foreground=iaa.Emboss(alpha=(0.5, 1.0), strength=(0.5, 1.0))),
                    iaa.BlendAlpha((0.01, 0.3), foreground=iaa.pillike.FilterContour()),
                ]),
            ]),
        ]),
        iaa.SomeOf((1, 2),[
            # ===================
            # image corrupt
            iaa.OneOf([
                # ===================
                # Noise
                iaa.OneOf([
                    iaa.MultiplyElementwise(mul=(0.8, 1.2), per_channel=0.5),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(10, 20), per_channel=0.5),
                    iaa.AdditiveLaplaceNoise(loc=0, scale=(10, 20), per_channel=0.5),
                    iaa.AdditivePoissonNoise(lam=(5, 15), per_channel=0.5),
                    iaa.imgcorruptlike.ImpulseNoise(severity=(1, 2)),
                    iaa.Dropout(p=(0.05, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(p=(0.04, 0.08), size_percent=(0.05, 0.1), per_channel=0.5),
                    # iaa.imgcorruptlike.GaussianNoise(severity=(1, 2)),  # medium
                    # iaa.imgcorruptlike.SpeckleNoise(severity=(1, 2)),  # medium
                    # iaa.imgcorruptlike.ShotNoise(severity=(1, 2)),  # hard
                ]),
                iaa.OneOf([
                    iaa.BlendAlpha((0.4, 0.6), foreground=iaa.ElasticTransformation(alpha=(1, 4), sigma=(0.05, 0.1))),
                    # pixels move locally around (with random strengths)
                    iaa.imgcorruptlike.JpegCompression(severity=(1, 5)),
                    iaa.imgcorruptlike.Pixelate(severity=(2, 4)),
                    iaa.BlendAlpha((0.4, 0.8), foreground=iaa.imgcorruptlike.ElasticTransform(severity=(1, 1))),
                    iaa.UniformColorQuantization(n_colors=(2, 16)),
                ]),
                iaa.OneOf([
                    iaa.Superpixels(p_replace=(0, 1.0), n_segments=(5000, 8000)),  # hard
                    iaa.AveragePooling(kernel_size=(2, 5)),
                    iaa.MaxPooling(kernel_size=(2, 5)),
                    iaa.MinPooling(kernel_size=(2, 5)),
                    iaa.MedianPooling(kernel_size=(2, 5)),
                ]),
                # iaa.OneOf([ # hard
                #     iaa.Invert(p=0.5, per_channel=0.5),
                #     iaa.Solarize(p=0.5, threshold=(0, 255), per_channel=0.5),
                # ]),
            ]),
            # ===================
            # weather
            iaa.OneOf([
                iaa.OneOf([
                    iaa.BlendAlpha((0.5, 0.7), foreground=iaa.imgcorruptlike.Fog(severity=(1, 2)), per_channel=True),
                    iaa.BlendAlpha((0.8, 0.9), foreground=iaa.imgcorruptlike.Frost(severity=(1, 1)), per_channel=True),
                    iaa.BlendAlpha((0.6, 0.8), foreground=iaa.Clouds()),
                    iaa.BlendAlpha((0.5, 0.7), foreground=iaa.Fog()),
                ]),
                iaa.OneOf([
                    iaa.imgcorruptlike.Snow(severity=(1, 1)),
                    iaa.imgcorruptlike.Spatter(severity=(1, 2)),
                    iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03), density=(0.005, 0.075),
                                   density_uniformity=(0.3, 0.9)),
                    iaa.Rain(speed=(0.1, 0.3), nb_iterations=(1, 3), drop_size=(0.01, 0.02)),
                ]),
            ]),
        ], random_order=True),
    ])
    return seq


def augmentation_lvl3():
    seq = iaa.Sequential([
        iaa.SomeOf((1, 2),[
            # ===================
            # Color
            iaa.SomeOf((1, 3),[
                # Hue
                iaa.OneOf([
                    iaa.MultiplyHue((0.5, 1.5)),
                    iaa.AddToHue((-50, 50)),
                    iaa.ChangeColorTemperature((1000, 10000)),
                    iaa.Add(value=(-50, 50), per_channel=0.5),
                    iaa.Multiply(mul=(0.5, 1.5), per_channel=0.5),
                    iaa.pillike.EnhanceColor(factor=(1.0, 3.0)),
                ]),
                # Saturation
                iaa.OneOf([
                    iaa.MultiplySaturation((0.5, 1.5)),
                    iaa.AddToSaturation((-50, 50)),
                    iaa.imgcorruptlike.Saturate(severity=(1, 3)),
                    iaa.RemoveSaturation(),
                    iaa.Grayscale(alpha=(0.3, 1.0)),
                ]),
                # Brightness
                iaa.OneOf([
                    iaa.MultiplyBrightness((0.5, 1.5)),
                    iaa.AddToBrightness((-50, 50)),
                    iaa.pillike.EnhanceBrightness(factor=(0.5, 1.5)),
                    iaa.imgcorruptlike.Brightness(severity=(1, 3)),
                ]),
                # Contrast
                iaa.OneOf([
                    iaa.GammaContrast(gamma=(0.5, 2.0), per_channel=0.5),
                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=0.5),
                    iaa.LogContrast(gain=(0.6, 1.4), per_channel=0.5),
                    iaa.LinearContrast(alpha=(0.5, 2.0), per_channel=0.5),
                    iaa.pillike.EnhanceContrast(factor=(0.5, 1.5)),
                    iaa.pillike.Autocontrast(cutoff=(10, 20), per_channel=0.5),
                    iaa.AllChannelsCLAHE(clip_limit=(1, 3), per_channel=0.5),  # medium
                    # iaa.imgcorruptlike.Contrast(severity=(1, 3)),  # for difficult
                ]),
            ], random_order=True),

            iaa.OneOf([
                # ===================
                # Blur
                iaa.OneOf([
                    iaa.OneOf([
                        iaa.GaussianBlur(sigma=(0.5, 2.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 9)),
                        iaa.imgcorruptlike.GaussianBlur(severity=(1, 2)),
                        iaa.pillike.FilterBlur(),
                    ]),
                    iaa.OneOf([
                        iaa.imgcorruptlike.MotionBlur(severity=(1, 2)),
                        iaa.MotionBlur(k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0)),
                        iaa.Alpha((0.01, 0.05), iaa.Canny(alpha=(0, 1)), iaa.MedianBlur((1, 5))),
                        iaa.imgcorruptlike.GlassBlur(severity=(1, 2)),  # medium
                        iaa.imgcorruptlike.DefocusBlur(severity=(1, 2)),  # medium
                        # iaa.imgcorruptlike.ZoomBlur(severity=(1, 1))  # hard
                    ]),
                ]),
                # ===================
                # sharpen and edge
                iaa.OneOf([
                    iaa.Sharpen(alpha=(0.1, 0.4), lightness=(0.75, 1.5)),
                    iaa.Canny(alpha=(0.05, 0.2),
                              colorizer=iaa.RandomColorsBinaryImageColorizer(color_true=255, color_false=0)),
                    iaa.Canny(alpha=(0.05, 0.2), sobel_kernel_size=[3, 7]),
                    iaa.Alpha(factor=(0.05, 0.2), first=iaa.Canny(alpha=1), second=iaa.MedianBlur(k=(1, 3)),
                              per_channel=0.5),
                    iaa.BlendAlpha((0.3, 0.6), foreground=iaa.Emboss(alpha=(0.5, 1.0), strength=(0.5, 1.0))),
                    iaa.BlendAlpha((0.01, 0.3), foreground=iaa.pillike.FilterContour()),
                ]),
            ]),
        ], random_order=True),

        iaa.SomeOf((0, 2),[
            # ===================
            # image corrupt
            iaa.OneOf([
                iaa.OneOf([
                    iaa.MultiplyElementwise(mul=(0.1, 1.5), per_channel=0.5),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(5, 20), per_channel=0.5),
                    iaa.AdditiveLaplaceNoise(loc=0, scale=(5, 20), per_channel=0.5),
                    iaa.AdditivePoissonNoise(lam=(5, 15), per_channel=0.5),
                    iaa.imgcorruptlike.ImpulseNoise(severity=(1, 3)),
                    iaa.Dropout(p=(0.01, 0.1), per_channel=0.5),
                    iaa.imgcorruptlike.GaussianNoise(severity=(1, 2)),  # medium
                    iaa.imgcorruptlike.SpeckleNoise(severity=(1, 2)),  # medium
                    # iaa.imgcorruptlike.ShotNoise(severity=(1, 2)),  # hard
                ]),
                iaa.OneOf([
                    iaa.BlendAlpha((0.4, 0.6), foreground=iaa.ElasticTransformation(alpha=(1, 4), sigma=(0.05, 0.1))),
                    # pixels move locally around (with random strengths)
                    iaa.imgcorruptlike.JpegCompression(severity=(1, 5)),
                    iaa.imgcorruptlike.Pixelate(severity=(2, 4)),
                    iaa.BlendAlpha((0.4, 0.8), foreground=iaa.imgcorruptlike.ElasticTransform(severity=(1, 1))),
                    iaa.UniformColorQuantization(n_colors=(2, 16)),
                ]),
                iaa.OneOf([
                    iaa.CoarseDropout(p=(0.04, 0.08), size_percent=(0.05, 0.1), per_channel=0.5),
                ]),
                iaa.OneOf([
                    iaa.Superpixels(p_replace=(0, 1.0), n_segments=(5000, 8000)),  # hard
                    iaa.AveragePooling(kernel_size=(2, 4)),
                    iaa.MaxPooling(kernel_size=(2, 3)),
                    iaa.MinPooling(kernel_size=(2, 3)),
                    iaa.MedianPooling(kernel_size=(2, 3)),
                ]),
                # iaa.OneOf([ # hard
                #     iaa.Invert(p=0.5, per_channel=0.5),
                #     iaa.Solarize(p=0.5, threshold=(0, 255), per_channel=0.5),
                # ]),
            ]),
            # ===================
            # weather
            iaa.OneOf([
                iaa.OneOf([
                    iaa.BlendAlpha((0.6, 0.8), foreground=iaa.imgcorruptlike.Fog(severity=(1, 2)), per_channel=True),
                    iaa.BlendAlpha((0.8, 0.9), foreground=iaa.imgcorruptlike.Frost(severity=(1, 1)), per_channel=True),
                    iaa.BlendAlpha((0.6, 0.8), foreground=iaa.Clouds()),
                    iaa.BlendAlpha((0.6, 0.8), foreground=iaa.Fog()),
                ]),
                iaa.OneOf([
                    iaa.imgcorruptlike.Snow(severity=(1, 2)),
                    iaa.imgcorruptlike.Spatter(severity=(1, 2)),
                    iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03), density=(0.005, 0.075),
                                   density_uniformity=(0.3, 0.9)),
                    iaa.Rain(speed=(0.1, 0.3), nb_iterations=(1, 3), drop_size=(0.01, 0.02)),
                ]),
            ]),
        ], random_order=True),
    ])
    return seq


def augmentation_lvl4():
    seq = iaa.Sequential([
        iaa.SomeOf((1, 2),[
            # ===================
            # Color
            iaa.SomeOf((1, 3),[
                # Hue
                iaa.OneOf([
                    iaa.MultiplyHue((0.5, 1.5)),
                    iaa.AddToHue((-50, 50)),
                    iaa.ChangeColorTemperature((1000, 10000)),
                    iaa.Add(value=(-50, 50), per_channel=0.5),
                    iaa.Multiply(mul=(0.5, 1.5), per_channel=0.5),
                    iaa.pillike.EnhanceColor(factor=(1.0, 3.0)),
                ]),
                # Saturation
                iaa.OneOf([
                    iaa.MultiplySaturation((0.5, 1.5)),
                    iaa.AddToSaturation((-50, 50)),
                    iaa.imgcorruptlike.Saturate(severity=(1, 3)),
                    iaa.RemoveSaturation(),
                    iaa.Grayscale(alpha=(0.3, 1.0)),
                ]),
                # Brightness
                iaa.OneOf([
                    iaa.MultiplyBrightness((0.5, 1.5)),
                    iaa.AddToBrightness((-50, 50)),
                    iaa.pillike.EnhanceBrightness(factor=(0.5, 1.5)),
                    iaa.imgcorruptlike.Brightness(severity=(1, 3)),
                ]),
                # Contrast
                iaa.OneOf([
                    iaa.GammaContrast(gamma=(0.5, 2.0), per_channel=0.5),
                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=0.5),
                    iaa.LogContrast(gain=(0.6, 1.4), per_channel=0.5),
                    iaa.LinearContrast(alpha=(0.5, 2.0), per_channel=0.5),
                    iaa.pillike.EnhanceContrast(factor=(0.5, 1.5)),
                    iaa.pillike.Autocontrast(cutoff=(10, 20), per_channel=0.5),
                    iaa.AllChannelsCLAHE(clip_limit=(1, 3), per_channel=0.5),  # medium
                    iaa.imgcorruptlike.Contrast(severity=(1, 3)),  # for difficult
                ]),
            ], random_order=True),

            iaa.OneOf([
                # ===================
                # Blur
                iaa.OneOf([
                    iaa.OneOf([
                        iaa.GaussianBlur(sigma=(0.5, 2.0)),
                        iaa.AverageBlur(k=(3, 9)),
                        iaa.MedianBlur(k=(3, 9)),
                        iaa.imgcorruptlike.GaussianBlur(severity=(1, 2)),
                        iaa.pillike.FilterBlur(),
                    ]),
                    iaa.OneOf([
                        iaa.imgcorruptlike.MotionBlur(severity=(1, 2)),
                        iaa.MotionBlur(k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0)),
                        iaa.Alpha((0.01, 0.05), iaa.Canny(alpha=(0, 1)), iaa.MedianBlur((1, 5))),
                        iaa.imgcorruptlike.GlassBlur(severity=(1, 2)),  # medium
                        iaa.imgcorruptlike.DefocusBlur(severity=(1, 2)),  # medium
                        iaa.imgcorruptlike.ZoomBlur(severity=(1, 1))  # hard
                    ]),
                ]),
                # ===================
                # sharpen and edge
                iaa.OneOf([
                    iaa.Sharpen(alpha=(0.1, 0.4), lightness=(0.75, 1.5)),
                    iaa.Canny(alpha=(0.05, 0.2),
                              colorizer=iaa.RandomColorsBinaryImageColorizer(color_true=255, color_false=0)),
                    iaa.Canny(alpha=(0.05, 0.2), sobel_kernel_size=[3, 7]),
                    iaa.Alpha(factor=(0.05, 0.2), first=iaa.Canny(alpha=1), second=iaa.MedianBlur(k=(1, 3)),
                              per_channel=0.5),
                    iaa.BlendAlpha((0.3, 0.6), foreground=iaa.Emboss(alpha=(0.5, 1.0), strength=(0.5, 1.0))),
                    iaa.BlendAlpha((0.01, 0.3), foreground=iaa.pillike.FilterContour()),
                ]),
            ]),
        ], random_order=True),

        iaa.Sequential([
            # ===================
            # image corrupt
            iaa.OneOf([
                iaa.OneOf([
                    iaa.MultiplyElementwise(mul=(0.5, 1.5), per_channel=0.5),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(10, 20), per_channel=0.5),
                    iaa.AdditiveLaplaceNoise(loc=0, scale=(10, 20), per_channel=0.5),
                    iaa.AdditivePoissonNoise(lam=(5, 15), per_channel=0.5),
                    iaa.imgcorruptlike.ImpulseNoise(severity=(2, 3)),
                    iaa.Dropout(p=(0.05, 0.1), per_channel=0.5),
                    iaa.imgcorruptlike.GaussianNoise(severity=(2, 3)),  # medium
                    iaa.imgcorruptlike.SpeckleNoise(severity=(2, 3)),  # medium
                    iaa.imgcorruptlike.ShotNoise(severity=(1, 2)),  # hard
                    iaa.CoarseDropout(p=(0.04, 0.08), size_percent=(0.05, 0.1), per_channel=0.5),
                ]),
                iaa.OneOf([
                    iaa.BlendAlpha((0.4, 0.6), foreground=iaa.ElasticTransformation(alpha=(1, 4), sigma=(0.05, 0.1))),
                    # pixels move locally around (with random strengths)
                    iaa.imgcorruptlike.JpegCompression(severity=(1, 5)),
                    iaa.imgcorruptlike.Pixelate(severity=(2, 4)),
                    iaa.BlendAlpha((0.4, 0.8), foreground=iaa.imgcorruptlike.ElasticTransform(severity=(1, 1))),
                    iaa.UniformColorQuantization(n_colors=(2, 16)),
                ]),
                iaa.OneOf([
                    iaa.Superpixels(p_replace=(0, 1.0), n_segments=(5000, 8000)),  # hard
                    iaa.AveragePooling(kernel_size=(2, 4)),
                    iaa.MaxPooling(kernel_size=(2, 5)),
                    iaa.MinPooling(kernel_size=(2, 5)),
                    iaa.MedianPooling(kernel_size=(2, 5)),
                ]),
                iaa.OneOf([  # hard
                    iaa.Invert(p=0.5, per_channel=0.5),
                    iaa.Solarize(p=0.5, threshold=(0, 255), per_channel=0.5),
                ]),
            ]),
            # ===================
            # weather
            iaa.OneOf([
                iaa.OneOf([
                    iaa.BlendAlpha((0.6, 0.9), foreground=iaa.imgcorruptlike.Fog(severity=(1, 2)), per_channel=True),
                    iaa.BlendAlpha((0.8, 0.9), foreground=iaa.imgcorruptlike.Frost(severity=(1, 1)), per_channel=True),
                    iaa.BlendAlpha((0.6, 0.9), foreground=iaa.Clouds()),
                    iaa.BlendAlpha((0.6, 0.9), foreground=iaa.Fog()),
                ]),
                iaa.OneOf([
                    iaa.imgcorruptlike.Snow(severity=(1, 2)),
                    iaa.imgcorruptlike.Spatter(severity=(1, 2)),
                    iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03), density=(0.005, 0.075),
                                   density_uniformity=(0.3, 0.9)),
                    iaa.Rain(speed=(0.1, 0.3), nb_iterations=(1, 3), drop_size=(0.01, 0.02)),
                ]),
            ]),
        ], random_order=True),
    ])
    return seq

##############


def augment_dataset(dir_input, dir_output, seq):
    os.makedirs(dir_output, exist_ok=True)
    files = os.listdir(dir_input)
    files = [file for file in files if file[0] != '.']
    files.sort()
    for file in tqdm(files):
        filename = os.path.splitext(os.path.basename(file))[0]
        image = cv2.imread(os.path.join(dir_input, file))
        image = image[:, :, [2, 1, 0]]
        image = np.expand_dims(image, axis=0)
        image_aug = seq(images=image)
        image_aug = np.squeeze(image_aug, axis=0)
        image_aug = image_aug[:, :, [2, 1, 0]]
        cv2.imwrite(os.path.join(dir_output, filename + '.jpg'), image_aug, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    seq = augmentation_lvl1()
    augment_dataset(dir_input='/Users/ltnghia/Workplace/Projects/DeepFake/OpenForensics/Release/V.1.0/Visualization/Images/Val',
                    dir_output='/Users/ltnghia/Workplace/Projects/DeepFake/OpenForensics/Release/V.1.0/Visualization/Augmented_Images/Val',
                    seq=seq)
