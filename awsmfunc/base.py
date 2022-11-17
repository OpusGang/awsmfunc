import math
from enum import Enum
from functools import partial
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Union

import vapoursynth as vs
from vapoursynth import core
from vsutil import depth as vsuDepth
from vsutil import get_depth, join, scale_value, split

from rekt import rektlvls

from .types.dovi import HdrMeasurement
from .types.misc import ST2084_PEAK_LUMINANCE, st2084_eotf, st2084_inverse_eotf
from .types.placebo import PlaceboTonemapOpts

SUBTITLE_DEFAULT_STYLE: str = ("sans-serif,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
                               "0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1")


def Depth(clip: vs.VideoNode, bits: int, **kwargs) -> vs.VideoNode:
    """
    This is just a vsutil.depth wrapper that doesn't dither for high bit depth.
    The reason for this is that repeated error diffusion can be destructive, so if we assume most filtering is done at
    high bit depth (which it should), this should keep that from happening while still performing it before outputting.
    """
    dither_type = "error_diffusion" if bits < 16 else "none"

    return vsuDepth(clip, bits, dither_type=dither_type, **kwargs)


def ReplaceFrames(clipa: vs.VideoNode,
                  clipb: vs.VideoNode,
                  mappings: Optional[str] = None,
                  filename: Optional[Union[str, PathLike]] = None) -> vs.VideoNode:
    """
    ReplaceFramesSimple wrapper that uses the RemapFrames plugin and works for different length clips.
    https://github.com/Irrational-Encoding-Wizardry/Vapoursynth-RemapFrames
    :param clipa: Main clip.
    :param clipb: Filtered clip to splice into main clip.
    :param mappings: String of frames to be replaced, e.g. "[0 500] [1000 1500]".
    :param filename: File with frames to be replaced.
    :return: clipa with clipb spliced in according to specified frames.
    """
    try:
        return core.remap.Rfs(baseclip=clipa, sourceclip=clipb, mappings=mappings, filename=filename)
    except vs.Error:

        # copy-pasted from fvsfunc, sadly
        def __fvsfunc_remap(clipa: vs.VideoNode, clipb: vs.VideoNode, mappings, filename) -> vs.VideoNode:
            import re
            if filename:
                with open(filename, 'r') as mf:
                    mappings += f'\n{mf.read()}'
            # Some people used this as separators and wondered why it wasn't working
            mappings = mappings.replace(',', ' ').replace(':', ' ')

            frames = re.findall(r'\d+(?!\d*\s*\d*\s*\d*\])', mappings)
            ranges = re.findall(r'\[\s*\d+\s+\d+\s*\]', mappings)
            maps = []

            for range_ in ranges:
                maps.append([int(x) for x in range_.strip('[ ]').split()])

            for frame in frames:
                maps.append([int(frame), int(frame)])

            maps = [x for y in maps for x in y]
            start, end = min(maps), max(maps)

            if (end - start) > len(clipb):
                raise ValueError("ReplaceFrames: mappings exceed clip length!")

            if len(clipb) < len(clipa):
                clipb = clipb + clipb.std.BlankClip(length=len(clipa) - len(clipb))

            elif len(clipb) > len(clipa):
                clipb = clipb.std.Trim(0, len(clipa) - 1)

            return clipa, clipb, mappings, filename

        clipa, clipb, mappings, filename = __fvsfunc_remap(clipa, clipb, mappings, filename)

        return core.remap.Rfs(baseclip=clipa, sourceclip=clipb, mappings=mappings, filename=filename)


def bbmod(clip: vs.VideoNode,
          top: int = 0,
          bottom: int = 0,
          left: int = 0,
          right: int = 0,
          thresh: Any = None,
          blur: int = 20,
          planes: Any = None,
          y: Optional[bool] = None,
          u: Optional[bool] = None,
          v: Optional[bool] = None,
          scale_thresh: Optional[bool] = None,
          cpass2: bool = False,
          csize: int = 2,
          scale_offsets: bool = True,
          cTop: Optional[int] = None,
          cBottom: Optional[int] = None,
          cLeft: Optional[int] = None,
          cRight: Optional[int] = None) -> vs.VideoNode:
    """
    bbmod helper for a significant speedup from cropping unnecessary pixels before processing.
    :param clip: Clip to be processed.
    :param top: Top rows to be processed.
    :param bottom: Bottom rows to be processed.
    :param left: Left columns to be processed.
    :param right: Right columns to be processed.
    :param thresh: Largest change allowed. Scale of 0-128, default is 128 (assuming 8-bit).
                   Specify a list for [luma, chroma] or [y, u, v].
    :param blur: Processing strength, lower values are more aggressive. Default is 20, not 999 like the old bbmod.
                 Specify a list for [luma, chroma] or [y, u, v].
    :param planes: Planes to process. Overwrites y, u, v. Defaults to all planes.
    :param y: Boolean whether luma plane is processed. Default is True.
    :param u: Boolean whether first chroma plane is processed. Default is True.
    :param v: Boolean whether second chroma plane is processed. Default is True.
    :param scale_thresh: Boolean whether thresh value is scaled from 8-bit to source bit depth.
                         If thresh <= 128, this defaults to True, else False.
    :param cpass2: Second, significantly stronger, chroma pass. If enabled, default for chroma blur is blur * 2 and
                   chroma thresh is thresh / 10.
    :param csize: Size to be cropped to. This might help maintain details at the cost of processing speed.
    :param scale_offsets: Whether scaling should take offsets into account in vsutil.scale_value.
                          If you don't know what this means, don't change it.  Thresh never uses scale_offsets.
    :param cTop: Legacy top.
    :param cBottom: Legacy bottom.
    :param cLeft: Legacy left.
    :param cRight: Legacy right.
    :return: Clip with color offsets fixed.
    """

    if clip.format.color_family not in (vs.YUV, vs.GRAY):
        raise ValueError("bbmod: only YUV and GRAY clips are supported")

    if cTop is not None:
        top = cTop

    if cBottom is not None:
        bottom = cBottom

    if cLeft is not None:
        left = cLeft

    if cRight is not None:
        right = cRight

    if planes is not None:
        if isinstance(planes, int):
            planes = [planes]

        y = bool(0 in planes)
        u = bool(1 in planes)
        v = bool(2 in planes)
    elif clip.format.color_family == vs.YUV:
        if y is None:
            y = True
        if u is None:
            u = True
        if v is None:
            v = True
    else:
        if y is None and not u and not v:
            y = True
            u = False
            v = False
        elif y is True:
            u = False
            v = False
        else:
            y = False
            u = True
            v = False

    depth = clip.format.bits_per_sample
    if thresh is None:
        thresh = scale_value(128, 8, depth, scale_offsets=scale_offsets)

    if scale_thresh is None:
        if thresh < 1:
            scale_thresh = False
        elif thresh < 129:
            scale_thresh = True
        else:
            scale_thresh = False

    filtered = clip

    c_left = max(left * csize, 4)
    c_right = max(right * csize, 4)
    c_top = max(top * csize, 4)
    c_bottom = max(bottom * csize, 4)

    f_width, f_height = filtered.width, filtered.height

    if left > 0 and right > 0:
        left_clip = filtered.std.Crop(left=0, right=f_width - c_left, top=0, bottom=0)
        middle_clip = filtered.std.Crop(left=c_left, right=c_right, top=0, bottom=0)
        right_clip = filtered.std.Crop(left=f_width - c_right, right=0, top=0, bottom=0)

        left_clip = bbmoda(left_clip,
                           cTop=0,
                           cBottom=0,
                           cLeft=left,
                           cRight=0,
                           thresh=thresh,
                           blur=blur,
                           y=y,
                           u=u,
                           v=v,
                           scale_thresh=scale_thresh,
                           cpass2=cpass2,
                           csize=csize,
                           scale_offsets=scale_offsets)
        right_clip = bbmoda(right_clip,
                            cTop=0,
                            cBottom=0,
                            cLeft=0,
                            cRight=right,
                            thresh=thresh,
                            blur=blur,
                            y=y,
                            u=u,
                            v=v,
                            scale_thresh=scale_thresh,
                            cpass2=cpass2,
                            csize=csize,
                            scale_offsets=scale_offsets)

        filtered = core.std.StackHorizontal(clips=[left_clip, middle_clip, right_clip])

    if left > 0 and right == 0:
        left_clip = filtered.std.Crop(left=0, right=f_width - c_left, top=0, bottom=0)
        middle_clip = filtered.std.Crop(left=c_left, right=0, top=0, bottom=0)

        left_clip = bbmoda(left_clip,
                           cTop=0,
                           cBottom=0,
                           cLeft=left,
                           cRight=0,
                           thresh=thresh,
                           blur=blur,
                           y=y,
                           u=u,
                           v=v,
                           scale_thresh=scale_thresh,
                           cpass2=cpass2,
                           csize=csize,
                           scale_offsets=scale_offsets)

        filtered = core.std.StackHorizontal(clips=[left_clip, middle_clip])

    if left == 0 and right > 0:
        right_clip = filtered.std.Crop(left=f_width - c_right, right=0, top=0, bottom=0)
        middle_clip = filtered.std.Crop(left=0, right=c_right, top=0, bottom=0)

        right_clip = bbmoda(right_clip,
                            cTop=0,
                            cBottom=0,
                            cLeft=0,
                            cRight=right,
                            thresh=thresh,
                            blur=blur,
                            y=y,
                            u=u,
                            v=v,
                            scale_thresh=scale_thresh,
                            cpass2=cpass2,
                            csize=csize,
                            scale_offsets=scale_offsets)

        filtered = core.std.StackHorizontal(clips=[middle_clip, right_clip])

    if top > 0 and bottom > 0:
        top_clip = filtered.std.Crop(left=0, right=0, top=0, bottom=f_height - c_top)
        middle_clip = filtered.std.Crop(left=0, right=0, top=c_top, bottom=c_bottom)
        bottom_clip = filtered.std.Crop(left=0, right=0, top=f_height - c_bottom, bottom=0)

        top_clip = bbmoda(top_clip,
                          cTop=top,
                          cBottom=0,
                          cLeft=0,
                          cRight=0,
                          thresh=thresh,
                          blur=blur,
                          y=y,
                          u=u,
                          v=v,
                          scale_thresh=scale_thresh,
                          cpass2=cpass2,
                          csize=csize,
                          scale_offsets=scale_offsets)
        bottom_clip = bbmoda(bottom_clip,
                             cTop=0,
                             cBottom=bottom,
                             cLeft=0,
                             cRight=0,
                             thresh=thresh,
                             blur=blur,
                             y=y,
                             u=u,
                             v=v,
                             scale_thresh=scale_thresh,
                             cpass2=cpass2,
                             csize=csize,
                             scale_offsets=scale_offsets)

        filtered = core.std.StackVertical(clips=[top_clip, middle_clip, bottom_clip])

    if top > 0 and bottom == 0:
        top_clip = filtered.std.Crop(left=0, right=0, top=0, bottom=f_height - c_top)
        middle_clip = filtered.std.Crop(left=0, right=0, top=c_top, bottom=0)

        top_clip = bbmoda(top_clip,
                          cTop=top,
                          cBottom=0,
                          cLeft=0,
                          cRight=0,
                          thresh=thresh,
                          blur=blur,
                          y=y,
                          u=u,
                          v=v,
                          scale_thresh=scale_thresh,
                          cpass2=cpass2,
                          csize=csize,
                          scale_offsets=scale_offsets)

        filtered = core.std.StackVertical(clips=[top_clip, middle_clip])

    if top == 0 and bottom > 0:
        bottom_clip = filtered.std.Crop(left=0, right=0, top=f_height - c_bottom, bottom=0)
        middle_clip = filtered.std.Crop(left=0, right=0, top=0, bottom=c_bottom)

        bottom_clip = bbmoda(bottom_clip,
                             cTop=0,
                             cBottom=bottom,
                             cLeft=0,
                             cRight=0,
                             thresh=thresh,
                             blur=blur,
                             y=y,
                             u=u,
                             v=v,
                             scale_thresh=scale_thresh,
                             cpass2=cpass2,
                             csize=csize,
                             scale_offsets=scale_offsets)

        filtered = core.std.StackVertical(clips=[middle_clip, bottom_clip])

    return filtered


def bbmoda(c: vs.VideoNode,
           cTop: int = 0,
           cBottom: int = 0,
           cLeft: int = 0,
           cRight: int = 0,
           thresh: Any = 128,
           blur: Any = 999,
           y: bool = True,
           u: bool = True,
           v: bool = True,
           scale_thresh: bool = True,
           cpass2: bool = False,
           csize: int = 2,
           scale_offsets: bool = True) -> vs.VideoNode:
    """
    bbmod, port from Avisynth's function, a mod of BalanceBorders
      The function changes the extreme pixels of the clip, to fix or attenuate dirty borders
      Any bit depth
      Inspired from BalanceBorders from
        https://github.com/WolframRhodium/muvsfunc/ and
        https://github.com/fdar0536/Vapoursynth-BalanceBorders/
    > Usage: bbmod(c, cTop, cBottom, cLeft, cRight, thresh, blur)
      * c: Input clip. The image area "in the middle" does not change during processing.
           The clip can be any format, which differs from Avisynth's equivalent.
      * cTop, cBottom, cLeft, cRight (int, 0-inf): The number of variable pixels on each side.
      * thresh (int, 0~128, default 128): Threshold of acceptable changes for local color matching in 8 bit scale.
        Recommended: 0~16 or 128
      * blur (int, 1~inf, default 999): Degree of blur for local color matching.
        Smaller values give a more accurate color match, larger values give a more accurate picture transfer.
        Recommended: 1~20 or 999
      Notes:
        1) At default values of thresh = 128 blur = 999:
           You will get a series of pixels that have been changed only by selecting
            the color for each row in its entirety, without local selection;
           The colors of neighboring pixels may be very different in some places
            but there will be no change in the nature of the picture.
           With thresh = 128 and blur = 1 you get almost the same rows of pixels, i.e.
            The colors between them will coincide completely, but the original pattern will be lost.
        2) Beware of using a large number of pixels to change in combination with a high level of "thresh",
           and a small "blur" that can lead to unwanted artifacts "in a clean place".
           For each function call, try to set as few pixels as possible to change
            and as low a threshold as possible "thresh" (when using blur 0..16).
    """
    funcName = "bbmoda"

    if not isinstance(c, vs.VideoNode):
        raise TypeError(funcName + ': \"c\" must be a clip!')

    if isinstance(thresh, (int, float)):
        # thresh needs to be lower for chroma for cpass2
        if cpass2:
            thresh = [thresh] + (2 * [round(thresh / 10)])
        else:
            thresh = 3 * [thresh]
    elif len(thresh) == 2:
        thresh.append(thresh[1])

    if scale_thresh:
        thresh[0] = scale_value(thresh[0], 8, c.format.bits_per_sample, scale_offsets=False)
        i = 1

        for _ in thresh[1:]:
            thresh[i] = scale_value(thresh[i], 8, c.format.bits_per_sample, scale_offsets=False, chroma=False)
            i += 1

    if isinstance(blur, int):
        # blur should also be higher
        if cpass2:
            blur = [blur] + 2 * [blur * 2]
        else:
            blur = 3 * [blur]
    elif len(blur) == 2:
        blur.append(blur[1])

    for _ in blur:
        if _ <= 0:
            raise ValueError(funcName + ': \'blur\' has an incorrect value! (0 ~ inf]')

    for _ in thresh:
        if _ <= 0 and c.format.sample_type == vs.INTEGER:
            raise ValueError(funcName + ': \'thresh\' has an incorrect value! (0 ~ inf]')

    def btb(c: vs.VideoNode, cTop: int, thresh: Any, blur: Any) -> vs.VideoNode:

        cWidth = c.width
        cHeight = c.height
        sw, sh = c.format.subsampling_w + 1, c.format.subsampling_h + 1
        cTop = min(cTop, cHeight - 1)

        blurWidth = [
            max(8, math.floor(cWidth / blur[0])),
            max(8, math.floor(cWidth / blur[1])),
            max(8, math.floor(cWidth / blur[2]))
        ]

        scale128 = str(scale_value(128, 8, c.format.bits_per_sample, scale_offsets=scale_offsets, chroma=True))
        uvexpr_ = "z y - x +"
        uvexpr = []

        for t in [1, 2]:
            uvexpr.append((f"{uvexpr_} x - {thresh[t]} > x {thresh[t]} + {uvexpr_} x - "
                           f"-{thresh[t]} < x {thresh[t]} - {uvexpr_} ? ?"))
        if c.format.sample_type == vs.INTEGER:
            exprchroma = f"x {scale128} - abs 2 *"
            expruv = f"z y / 8 min 0.4 max x {scale128} - * {scale128} + x - {scale128} +"
        else:
            exprchroma = "x abs 2 *"
            expruv = "z .5 + y .5 + / 8 min .4 max x .5 + * x - .5 +"

        scale16 = str(scale_value(16, 8, c.format.bits_per_sample, scale_offsets=scale_offsets))

        yexpr = f"z {scale16} - y {scale16} - / 8 min 0.4 max x {scale16} - * {scale16} +"
        yexpr = f"{yexpr} x - {thresh[0]} > x {thresh[0]} + {yexpr} x - -{thresh[0]} < x {thresh[0]} - {yexpr} ? ?"

        if y and u and v and blur[0] == blur[1] == blur[2] and thresh[0] == thresh[1] == thresh[2] and sw == sh == 1:
            c2 = core.resize.Point(c, cWidth * csize, cHeight * csize)
            last = core.std.CropAbs(c2, cWidth * csize, csize, 0, cTop * csize)
            last = core.resize.Point(last, cWidth * csize, cTop * csize)
            exprchroma = ["", exprchroma]

            if cpass2:
                referenceBlurChroma = last.std.Expr(exprchroma).resize.Bicubic(blurWidth[0] * csize,
                                                                               cTop * csize,
                                                                               filter_param_a=1,
                                                                               filter_param_b=0).resize.Bicubic(
                                                                                   cWidth * csize,
                                                                                   cTop * csize,
                                                                                   filter_param_a=1,
                                                                                   filter_param_b=0)

            referenceBlur = core.resize.Bicubic(last,
                                                blurWidth[0] * csize,
                                                cTop * csize,
                                                filter_param_a=1,
                                                filter_param_b=0)
            referenceBlur = referenceBlur.resize.Bicubic(cWidth * csize,
                                                         cTop * csize,
                                                         filter_param_a=1,
                                                         filter_param_b=0)

            original = core.std.CropAbs(c2, cWidth * csize, cTop * csize, 0, 0)

            last = core.resize.Bicubic(original, blurWidth[0] * csize, cTop * csize, filter_param_a=1, filter_param_b=0)

            originalBlur = last.resize.Bicubic(cWidth * csize, cTop * csize, filter_param_a=1, filter_param_b=0)

            if cpass2:
                originalBlurChroma = last.std.Expr(exprchroma).resize.Bicubic(blurWidth[0] * csize,
                                                                              cTop * csize,
                                                                              filter_param_a=1,
                                                                              filter_param_b=0)

                originalBlurChroma = originalBlurChroma.resize.Bicubic(cWidth * csize,
                                                                       cTop * csize,
                                                                       filter_param_a=1,
                                                                       filter_param_b=0)

                balancedChroma = core.std.Expr(clips=[original, originalBlurChroma, referenceBlurChroma],
                                               expr=["", expruv])
                balancedLuma = core.std.Expr(clips=[balancedChroma, originalBlur, referenceBlur],
                                             expr=[yexpr, uvexpr[0], uvexpr[1]])
            else:
                balancedLuma = core.std.Expr(clips=[original, originalBlur, referenceBlur],
                                             expr=[yexpr, uvexpr[0], uvexpr[1]])

            return core.std.StackVertical(
                [balancedLuma,
                 core.std.CropAbs(c2, cWidth * csize, (cHeight - cTop) * csize, 0,
                                  cTop * csize)]).resize.Point(cWidth, cHeight)

        if c.format.color_family == vs.YUV:
            yplane, uplane, vplane = split(c)
        elif c.format.color_family == vs.GRAY:
            yplane = c
        else:
            raise ValueError("bbmod: only YUV and GRAY clips are supported")
        if y:
            c2 = core.resize.Point(yplane, cWidth * csize, cHeight * csize)
            last = core.std.CropAbs(c2, cWidth * csize, csize, 0, cTop * csize)
            last = core.resize.Point(last, cWidth * csize, cTop * csize)

            referenceBlur = core.resize.Bicubic(last,
                                                blurWidth[0] * csize,
                                                cTop * csize,
                                                filter_param_a=1,
                                                filter_param_b=0).resize.Bicubic(cWidth * csize,
                                                                                 cTop * csize,
                                                                                 filter_param_a=1,
                                                                                 filter_param_b=0)

            original = core.std.CropAbs(c2, cWidth * csize, cTop * csize, 0, 0)

            last = core.resize.Bicubic(original, blurWidth[0] * csize, cTop * csize, filter_param_a=1, filter_param_b=0)

            originalBlur = last.resize.Bicubic(cWidth * csize, cTop * csize, filter_param_a=1, filter_param_b=0)
            balancedLuma = core.std.Expr(clips=[original, originalBlur, referenceBlur], expr=yexpr)

            yplane = core.std.StackVertical(
                clips=[balancedLuma,
                       core.std.CropAbs(c2, cWidth * csize, (cHeight - cTop) * csize, 0, cTop * csize)]).resize.Point(
                           cWidth, cHeight)

            if c.format.color_family == vs.GRAY:
                return yplane

        def btbc(c2: vs.VideoNode, blurWidth: int, p: int, csize: int) -> vs.VideoNode:
            c2 = core.resize.Point(c2, round(cWidth * csize / sw), round(cHeight * csize / sh))
            last = core.std.CropAbs(c2, round(cWidth * csize / sw), round(csize / sh), 0, round(cTop * csize / sh))
            last = core.resize.Point(last, round(cWidth * csize / sw), round(cTop * csize / sh))
            if cpass2:
                referenceBlurChroma = last.std.Expr(exprchroma).resize.Bicubic(round(blurWidth * csize / sw),
                                                                               round(cTop * csize / sh),
                                                                               filter_param_a=1,
                                                                               filter_param_b=0)
                referenceBlurChroma = referenceBlurChroma.resize.Bicubic(round(cWidth * csize / sw),
                                                                         round(cTop * csize / sh),
                                                                         filter_param_a=1,
                                                                         filter_param_b=0)

            referenceBlur = core.resize.Bicubic(last,
                                                round(blurWidth * csize / sw),
                                                round(cTop * csize / sh),
                                                filter_param_a=1,
                                                filter_param_b=0).resize.Bicubic(round(cWidth * csize / sw),
                                                                                 round(cTop * csize / sh),
                                                                                 filter_param_a=1,
                                                                                 filter_param_b=0)

            original = core.std.CropAbs(c2, round(cWidth * csize / sw), round(cTop * csize / sh), 0, 0)

            last = core.resize.Bicubic(original,
                                       round(blurWidth * csize / sw),
                                       round(cTop * csize / sh),
                                       filter_param_a=1,
                                       filter_param_b=0)

            originalBlur = last.resize.Bicubic(round(cWidth * csize / sw),
                                               round(cTop * csize / sh),
                                               filter_param_a=1,
                                               filter_param_b=0)

            if cpass2:
                originalBlurChroma = last.std.Expr(exprchroma).resize.Bicubic(round(blurWidth * csize / sw),
                                                                              round(cTop * csize / sh),
                                                                              filter_param_a=1,
                                                                              filter_param_b=0)
                originalBlurChroma = originalBlurChroma.resize.Bicubic(round(cWidth * csize / sw),
                                                                       round(cTop * csize / sh),
                                                                       filter_param_a=1,
                                                                       filter_param_b=0)

                balancedChroma = core.std.Expr(clips=[original, originalBlurChroma, referenceBlurChroma], expr=expruv)
                balancedLuma = core.std.Expr(clips=[balancedChroma, originalBlur, referenceBlur], expr=expruv)
            else:
                balancedLuma = core.std.Expr(clips=[original, originalBlur, referenceBlur], expr=uvexpr[p - 1])

            return core.std.StackVertical([
                balancedLuma,
                c2.std.CropAbs(left=0,
                               top=round(cTop * csize / sh),
                               width=round(cWidth * csize / sw),
                               height=round(cHeight * csize / sh) - round(cTop * csize / sh))
            ]).resize.Point(round(cWidth / sw), round(cHeight / sh))

        if c.format.color_family == vs.GRAY:
            return btbc(yplane, blurWidth[1], 1, csize)

        if u:
            uplane = btbc(uplane, blurWidth[1], 1, csize * max(sw, sh))
        if v:
            vplane = btbc(vplane, blurWidth[2], 2, csize * max(sw, sh))

        return core.std.ShufflePlanes([yplane, uplane, vplane], [0, 0, 0], vs.YUV)

    if cTop > 0:
        c = btb(c, cTop, thresh, blur).std.Transpose().std.FlipHorizontal()
    else:
        c = core.std.Transpose(c).std.FlipHorizontal()

    if cLeft > 0:
        c = btb(c, cLeft, thresh, blur).std.Transpose().std.FlipHorizontal()
    else:
        c = core.std.Transpose(c).std.FlipHorizontal()

    if cBottom > 0:
        c = btb(c, cBottom, thresh, blur).std.Transpose().std.FlipHorizontal()
    else:
        c = core.std.Transpose(c).std.FlipHorizontal()

    if cRight > 0:
        c = btb(c, cRight, thresh, blur).std.Transpose().std.FlipHorizontal()
    else:
        c = core.std.Transpose(c).std.FlipHorizontal()

    return c


def AddBordersMod(clip: vs.VideoNode,
                  left: int = 0,
                  top: int = 0,
                  right: int = 0,
                  bottom: int = 0,
                  lsat: float = 0.88,
                  tsat: float = 0.2,
                  rsat: Optional[float] = None,
                  bsat: float = 0.2,
                  color: Union[str, List[float]] = None) -> vs.VideoNode:
    """
    VapourSynth port of AddBordersMod.
    Fuck writing a proper docstring.
    :param clip:
    :param left:
    :param top:
    :param right:
    :param bottom:
    :param lsat:
    :param tsat:
    :param rsat:
    :param bsat:
    :param color:
    :return:
    """
    if clip.format.subsampling_w != 1 and clip.format.subsampling_h != 1:
        raise TypeError("AddBordersMod: input must be 4:2:0")

    if rsat is None:
        if right > 2:
            rsat = .4
        else:
            rsat = .28

    if left > 0:
        if lsat != 1:
            lcl = clip.std.AddBorders(left=left, color=color)
            lcl = lcl.std.CropAbs(left, clip.height)
            lcm = clip.std.CropAbs(2, clip.height)
            lcm = saturation(lcm, sat=lsat)
            lcr = clip.std.Crop(left=2)
            clip = core.std.StackHorizontal([lcl, lcm, lcr])
        else:
            clip = clip.std.AddBorders(left=left, color=color)

    if top > 2:
        tcl = clip.std.AddBorders(top=top, color=color)
        tcl = tcl.std.CropAbs(clip.width, top - 2)
        tcm = clip.std.CropAbs(clip.width, 2)
        tcm = saturation(tcm, sat=tsat)
        tcm = rektlvls(tcm, [0, 1], [16 - 235] * 2, prot_val=None)
        clip = core.std.StackVertical([tcl, tcm, clip])
    elif top == 2:
        tcl = clip.std.CropAbs(clip.width, 2)
        tcl = saturation(tcl, sat=tsat)
        tcl = rektlvls(tcl, [0, 1], [16 - 235] * 2, prot_val=None)
        clip = core.std.StackVertical([tcl, clip])

    if right > 2:
        rcm = clip.std.Crop(left=clip.width - 2)
        rcm = saturation(rcm, sat=rsat)
        rcm = rektlvls(rcm, colnum=[0, 1], colval=[16 - 235] * 2, prot_val=None)
        rcr = clip.std.AddBorders(right=right, color=color)
        rcr = rcr.std.Crop(left=clip.width + 2)
        clip = core.std.StackHorizontal([clip, rcm, rcr])
    elif right == 2:
        rcr = clip.std.Crop(left=clip.width - 2)
        rcr = saturation(rcr, sat=rsat)
        rcr = rektlvls(rcr, colnum=[0, 1], colval=[16 - 235] * 2, prot_val=None)
        clip = core.std.StackHorizontal([clip, rcr])

    if bottom > 2:
        bcm = clip.std.Crop(top=clip.height - 2)
        bcm = saturation(bcm, sat=bsat)
        bcm = rektlvls(bcm, [0, 1], [16 - 235] * 2, prot_val=None)
        bcr = clip.std.AddBorders(bottom=bottom)
        bcr = bcr.std.Crop(top=clip.height + 2)
        clip = core.std.StackVertical([clip, bcm, bcr])
    elif bottom == 2:
        bcr = clip.std.Crop(top=clip.height - 2)
        bcr = saturation(bcr, sat=bsat)
        bcr = rektlvls(bcr, [0, 1], [16 - 235] * 2, prot_val=None)
        clip = core.std.StackVertical([clip, bcr])

    return clip


def zresize(clip: vs.VideoNode,
            preset: Optional[int] = None,
            width: Optional[int] = None,
            height: Optional[int] = None,
            left: int = 0,
            right: int = 0,
            top: int = 0,
            bottom: int = 0,
            kernel: str = "spline36",
            ar: float = 16 / 9,
            dither_type="error_diffusion",
            **kwargs) -> vs.VideoNode:

    # VSEdit doesn't like the global dict
    RESIZEDICT: Dict = {
        'bilinear': core.resize.Bilinear,
        'bicubic': core.resize.Bicubic,
        'point': core.resize.Point,
        'lanczos': core.resize.Lanczos,
        'spline16': core.resize.Spline16,
        'spline36': core.resize.Spline36,
        'spline64': core.resize.Spline64
    }

    orig_w = clip.width
    orig_h = clip.height

    orig_cropped_w = orig_w - left - right
    orig_cropped_h = orig_h - top - bottom

    if preset:
        if orig_cropped_w / orig_cropped_h > ar:
            return zresize(clip,
                           width=int(ar * preset),
                           left=left,
                           right=right,
                           top=top,
                           bottom=bottom,
                           kernel=kernel,
                           **kwargs)

        return zresize(clip, height=preset, left=left, right=right, top=top, bottom=bottom, kernel=kernel, **kwargs)

    if (width is None) and (height is None):
        width = orig_w
        height = orig_h
        rh = rw = 1
    elif width is None:
        rh = rw = height / orig_cropped_h
    elif height is None:
        rh = rw = width / orig_cropped_w
    else:
        rh = height / orig_h
        rw = width / orig_w

    w = round((orig_cropped_w * rw) / 2) * 2
    h = round((orig_cropped_h * rh) / 2) * 2

    if orig_w == w and orig_h == h:
        # noop
        return clip

    resizer = RESIZEDICT[kernel.lower()]

    fun = dict(clip=clip,
               width=w,
               height=h,
               src_left=left,
               src_top=top,
               src_width=orig_cropped_w,
               src_height=orig_cropped_h,
               dither_type=dither_type,
               **kwargs)

    return resizer(**fun)


def DebandReader(clip: vs.VideoNode,
                 csvfile: Union[str, PathLike],
                 grain: int = 64,
                 f3kdb_range: int = 30,
                 delimiter: str = ' ',
                 mask: Optional[vs.VideoNode] = None) -> vs.VideoNode:
    """
    DebandReader, read a csv file to apply a f3kdb filter for given strengths and frames.
    > Usage: DebandReader(clip, csvfile, grain, f3kdb_range)
      * csvfile is the path to a csv file containing in each row: <startframe> <endframe> <strength>
      * grain is passed as grainy and grainc in the f3kdb filter
      * f3kdb_range is passed as range in the f3kdb filter
    """
    import csv

    depth = get_depth(clip)
    filtered = clip if depth <= 16 else Depth(clip, 16)

    with open(csvfile) as debandcsv:
        csvzones = csv.reader(debandcsv, delimiter=delimiter)
        for row in csvzones:
            strength = row[2]

            db = core.f3kdb.Deband(clip,
                                   y=strength,
                                   cb=strength,
                                   cr=strength,
                                   grainy=grain,
                                   grainc=grain,
                                   dynamic_grain=True,
                                   range=f3kdb_range,
                                   output_depth=depth)

            filtered = ReplaceFrames(filtered, db, mappings="[" + row[0] + " " + row[1] + "]")

        if mask:
            filtered = core.std.MaskedMerge(clip, filtered, mask)

    return filtered


class ScreenGenPrefix(str, Enum):
    Sequential = 'seq'
    FrameNo = 'frame'


class ScreenGenEncoder(str, Enum):
    imwri = 'imwri'
    fpng = 'fpng'

    def write_frame(self, clip: vs.VideoNode, frame_num: int, path: PathLike, compression: Optional[int] = None):
        if self == self.imwri:
            core.imwri.Write(clip, "PNG24", path, overwrite=True).get_frame(frame_num)
        elif self == self.fpng:
            core.fpng.Write(clip, filename=path, overwrite=True, compression=compression).get_frame(frame_num)


def ScreenGen(clip: Union[vs.VideoNode, List[vs.VideoNode]],
              folder: Union[str, PathLike],
              suffix: Optional[Union[str, List[str]]] = None,
              prefix: Union[ScreenGenPrefix, str] = ScreenGenPrefix.Sequential,
              frame_numbers: Union[Union[str, PathLike], List[int]] = "screens.txt",
              start: int = 1,
              delim: str = ' ',
              encoder: Union[ScreenGenEncoder, str] = ScreenGenEncoder.fpng,
              fpng_compression: int = 1,
              callback: Optional[Callable[[str], None]] = None) -> None:
    """
    Generates screenshots from a list of frame numbers

    :param clip: Clip or list of clips to generate screenshots from
    :param folder: is the folder name that is created
    :param suffix: str or list of str of the appended file name suffix(es).
        Optional, defaults to letters of the alphabet by order in the clip list
    :param prefix: the unique identifier for every screenshot file generated
        Must match `ScreenGenPrefix` enum, or literals 'seq' or 'frame'
    :param frame_numbers: the list of frames, defaults to a file named screens.txt. Either a list or a file
    :param start: is the number at which the filenames start
    :param encoder: plugin to use to write the PNG. Defaults to fpng if present.
        Must match `ScreenGenEncoder` enum.
          - imwri: https://github.com/vapoursynth/vs-imwri
          - fpng: https://github.com/Mikewando/vsfpng
    :param fpng_compression: Compression level to use for fpng. imwri compresses by default
        0 - fast compression
        1 - slow compression (Default)
        2 - uncompressed
    :param callback: Log callback instead of printing

    Usage:
    >>> ScreenGen(src, "Screenshots", "a")\n
    >>> ScreenGen(enc, "Screenshots", "b")
    or
    >>> ScreenGen([src, enc], "Screenshots") # equivalent: src is a, enc is b
    """
    from pathlib import Path

    folder_path = Path(folder).resolve()

    if isinstance(frame_numbers, str):
        frame_num_path = Path(frame_numbers).resolve()

        if frame_num_path.is_file():
            with open(frame_num_path) as f:
                screens = f.readlines()

            # Keep value before first delim, so that we can parse default detect zones files
            screens = [v.split(delim)[0] for v in screens]

            # str to int
            screens = [int(x.strip()) for x in screens]
        else:
            raise ValueError('ScreenGen: Path to frame numbers file does not exist')
    elif isinstance(frame_numbers, list):
        screens = frame_numbers
    else:
        raise TypeError('ScreenGen: frame_numbers must be a file path or a list of frame numbers')

    clips = clip

    if not isinstance(clip, list):
        clips = [clip]

    suffixes = suffix

    if suffix is None:
        import string

        suffixes = list(string.ascii_lowercase)[:len(clips)]
    elif not isinstance(suffix, list):
        suffixes = [suffix]

    if len(clips) != len(suffixes):
        raise ValueError('ScreenGen: number of clips must be equal to number of suffixes')

    clip_infos = [dict(clip=c, suffix=s) for (c, s) in zip(clips, suffixes)]

    if screens:
        if not folder_path.is_dir():
            folder_path.mkdir()

        encoder_final = encoder
        has_vsfpng_plugin = HasLoadedPlugin("tools.mike.fpng")

        if not has_vsfpng_plugin and encoder_final == ScreenGenEncoder.fpng:
            encoder_final = ScreenGenEncoder.imwri

        for info in clip_infos:
            clip = info['clip']
            suffix = info['suffix']

            rgb_clip = clip.resize.Spline36(format=vs.RGB24, matrix_in_s="709", dither_type="error_diffusion")

            for i, num in enumerate(screens, start=start):
                if prefix == ScreenGenPrefix.Sequential:
                    filename = f'{i:02d}{suffix}.png'
                elif prefix == ScreenGenPrefix.FrameNo:
                    filename = f'{num}{suffix}.png'
                else:
                    raise ValueError('ScreenGen: invalid prefix enum value')

                final_path = folder_path.joinpath(filename).resolve()

                log_str = f'\rScreenGen: Writing file: {filename}'
                if prefix != ScreenGenPrefix.FrameNo:
                    log_str += f', frame: {num}'

                if callable(callback):
                    callback(log_str)
                else:
                    print(end=log_str)

                try:
                    encoder_final.write_frame(rgb_clip, num, final_path, fpng_compression)
                except vs.Error:
                    new_path = folder_path.joinpath(f'%d{suffix}.png').resolve()
                    encoder_final.write_frame(rgb_clip, num, new_path, fpng_compression)

    else:
        raise ValueError('ScreenGen: No screenshots to write to disk')


def DynamicTonemap(clip: vs.VideoNode,
                   show: bool = False,
                   src_fmt: bool = False,
                   adjust_gamma: bool = False,
                   chromaloc_in_s: str = "top_left",
                   chromaloc_s: str = "left",
                   reference: Optional[vs.VideoNode] = None,
                   predetermined_targets: Optional[Union[str, List[Union[int, float]]]] = None,
                   target_nits: int = 203.0,
                   libplacebo: bool = True,
                   placebo_opts: Optional[PlaceboTonemapOpts] = None) -> vs.VideoNode:
    """
    Without libplacebo:
        The clip (or reference) is blurred, then plane stats are measured.
        The tonemapping is then done according to the max RGB value.

    :param clip: HDR clip. PQ only if not using libplacebo.
    :param show: Whether to show nits values as Subtitle.
    :param src_fmt: Whether to output source bit depth instead of 8-bit 4:4:4.
    :param adjust_gamma: Adjusts gamma/saturation dynamically on low brightness areas when target nits are high.
        Requires `adaptivegrain-rs` plugin.
    :param chromaloc_in_s: Chromaloc of input
    :param chromaloc_s: Chromaloc of output
    :param reference: Reference clip to calculate target brightness with.
        Use cases include source/encode comparisons
        Does not work when using libplacebo with `peak_detect=True`.
    :param predetermined_targets: List of target nits per frame.
        List of numbers or file containing a target per line
        Must be equal to clip length.
        Does not work when using libplacebo with `peak_detect=True`.
    :param target_nits: Target peak white brightness, in nits
    :param libplacebo: Whether to use libplacebo as tonemapper
        Requires `vs-placebo` plugin.
    :param placebo_opts: See `PlaceboTonemapOpts`, for use with `libplacebo=True`
    :return: SDR YUV444P16 clip by default.
    """

    from pathlib import Path

    def __get_rgb_prop_src(clip: vs.VideoNode, reference: Optional[vs.VideoNode],
                           target_list: Optional[List[int]]) -> vs.VideoNode:
        if reference is not None:
            if not libplacebo:
                clip_to_blur = core.resize.Spline36(reference,
                                                    format=vs.RGB48,
                                                    range_in_s="limited",
                                                    range_s="full",
                                                    matrix_in_s="2020ncl",
                                                    matrix_s="rgb",
                                                    primaries_in_s="2020",
                                                    primaries_s="xyz",
                                                    dither_type="none",
                                                    chromaloc_in_s=chromaloc_in_s,
                                                    chromaloc_s=chromaloc_in_s)
            else:
                clip_to_blur = clip = core.resize.Spline36(reference,
                                                           format=vs.RGB48,
                                                           chromaloc_in_s=chromaloc_in_s,
                                                           chromaloc_s=chromaloc_in_s)
        else:
            clip_to_blur = clip

        try:
            blurred_clip = core.bilateral.Bilateral(clip_to_blur, sigmaS=1)
        except Exception:
            blurred_clip = core.std.Convolution(clip_to_blur, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])

        if not target_list:
            r_props = core.std.PlaneStats(blurred_clip, plane=0)
            g_props = core.std.PlaneStats(blurred_clip, plane=1)
            b_props = core.std.PlaneStats(blurred_clip, plane=2)

            prop_src = [r_props, g_props, b_props]
        else:
            prop_src = core.std.PlaneStats(blurred_clip, plane=0)

        return prop_src

    def __calculate_max_rgb(n: int,
                            f: vs.VideoFrame,
                            targets: Optional[List[int]] = None,
                            levels: Optional[str] = None) -> vs.VideoNode:
        max_value = 65535.0

        if levels is None:
            if levels == 'limited':
                max_value = 60395

        if not targets:
            r_max = st2084_eotf((f[0].props['PlaneStatsMax'] / max_value)) * ST2084_PEAK_LUMINANCE
            g_max = st2084_eotf((f[1].props['PlaneStatsMax'] / max_value)) * ST2084_PEAK_LUMINANCE
            b_max = st2084_eotf((f[2].props['PlaneStatsMax'] / max_value)) * ST2084_PEAK_LUMINANCE

            max_rgb = round(max([r_max, g_max, b_max]))
        else:
            max_rgb = round(targets[n])

        # Don't go below 100 or over 10 000 nits
        peak = max(max_rgb, target_nits)
        peak = min(peak, ST2084_PEAK_LUMINANCE)

        return (max_rgb, peak)

    def __add_show_info(clip: vs.VideoNode,
                        max_rgb: int,
                        nits: int,
                        targets: Optional[List[int]],
                        adjusted: bool = False,
                        gamma_adjust: float = None,
                        luma_scaling: float = None) -> vs.VideoNode:
        if not targets:
            string = f"Max RGB: {max_rgb:.04f}, target: {nits} nits"
        else:
            string = f"Predetermined target: {nits} nits"

        if adjusted:
            string += f"\ngamma_adjust: {gamma_adjust:.04f}, luma_scaling: {luma_scaling:.04f}"

        return core.sub.Subtitle(clip, string)

    def __dt(n: int,
             f: vs.VideoFrame,
             clip: vs.VideoNode,
             show: bool,
             adjust_gamma: bool,
             targets: Optional[List[int]] = None) -> vs.VideoNode:
        max_rgb, peak = __calculate_max_rgb(n, f, targets)

        # Tonemap
        clip = core.resize.Spline36(clip,
                                    matrix_in_s="rgb",
                                    matrix_s="709",
                                    transfer_in_s="st2084",
                                    transfer_s="709",
                                    primaries_in_s="xyz",
                                    primaries_s="709",
                                    range_in_s="full",
                                    range_s="limited",
                                    dither_type="none",
                                    nominal_luminance=peak,
                                    format=vs.YUV444P16)

        do_adjust = (adjust_gamma and peak >= 256 and HasLoadedPlugin("moe.kageru.adaptivegrain"))

        gamma_adjust = None
        luma_scaling = None

        if do_adjust:
            gamma_adjust = 1.00 + (0.50 * (peak / 1536))
            gamma_adjust = min(gamma_adjust, 1.50)

            if peak >= 1024:
                luma_scaling = 1500 - (750 * (peak / 2048))
                luma_scaling = max(luma_scaling, 1000)
            else:
                luma_scaling = 1250 - (500 * (peak / 1024))
                luma_scaling = max(luma_scaling, 1000)

            clip = clip.std.PlaneStats()
            mask = core.adg.Mask(clip, luma_scaling=luma_scaling)
            fix = clip.std.Levels(gamma=gamma_adjust, min_in=4096, max_in=60160, min_out=4096, max_out=60160, planes=0)
            clip = core.std.MaskedMerge(clip, fix, mask)

            saturated = saturation(clip, 1.0 + (abs(1.0 - gamma_adjust) * 0.5))
            clip = core.std.MaskedMerge(clip, saturated, mask)

        if show:
            clip = __add_show_info(clip,
                                   max_rgb,
                                   peak,
                                   targets,
                                   adjusted=do_adjust,
                                   gamma_adjust=gamma_adjust,
                                   luma_scaling=luma_scaling)

        clip = core.resize.Spline36(clip, format=vs.RGB48)

        return clip

    def __pl_dt(n: int,
                f: vs.VideoFrame,
                clip: vs.VideoNode,
                show: bool,
                placebo_opts: PlaceboTonemapOpts,
                pl_tm_params: Dict,
                targets: Optional[List[int]] = None,
                levels: Optional[str] = None) -> vs.VideoNode:
        max_rgb, frame_nits = __calculate_max_rgb(n, f, targets, levels=levels)

        fprops = f.props if targets else f[0].props

        src_max = frame_nits
        src_min = None

        if 'MasteringDisplayMinLuminance' in fprops:
            src_min = fprops['MasteringDisplayMinLuminance']

        is_full_range = fprops['_ColorRange'] == 0

        final_tm_params = pl_tm_params
        if placebo_opts.use_planestats:
            final_tm_params = {'src_max': src_max, 'src_min': src_min, **pl_tm_params}

        clip = core.placebo.Tonemap(clip, **final_tm_params)

        if show and clip.format.color_family != vs.YUV:
            show_clip = core.resize.Spline36(clip, format=vs.YUV444P16, matrix_s="709")
            clip = __add_show_info(show_clip, max_rgb, frame_nits, targets)
            clip = core.resize.Spline36(clip, format=vs.RGB48)
        elif show:
            if is_full_range:
                clip = clip.resize.Spline36(range_in_s="full", range_s="limited")

            clip = __add_show_info(clip, max_rgb, frame_nits, targets)

        return clip

    clip_orig_format = clip.format

    target_list: Optional[List] = None
    if predetermined_targets:
        if isinstance(predetermined_targets, list):
            target_list = predetermined_targets
        elif isinstance(predetermined_targets, str):
            targets_path = Path(predetermined_targets)

            if targets_path.is_file():
                with open(targets_path) as f:
                    target_list = f.readlines()
                    target_list = [float(x.strip()) for x in target_list]

        if target_list:
            target_list = [int(round(x)) for x in target_list]

            if len(target_list) != clip.num_frames:
                raise ValueError('Number of targets != clip length')
        else:
            raise ValueError('No predetermined target list found')

    # Make sure libplacebo is properly loaded
    use_placebo = libplacebo and HasLoadedPlugin("com.vs.placebo")

    if use_placebo:
        placebo_opts_final = placebo_opts
        if not placebo_opts_final:
            placebo_opts_final = PlaceboTonemapOpts()

        dst_fmt = vs.YUV444P16 if placebo_opts_final.is_dovi_src() else vs.RGB48

        clip = core.resize.Spline36(clip, format=dst_fmt, chromaloc_in_s=chromaloc_in_s, chromaloc_s=chromaloc_in_s)

        dst_max = target_nits
        dst_min = None

        if placebo_opts_final.is_sdr_target():
            dst_min = dst_max / 1000.0  # 1000:1 default

        pl_tm_params = {'dst_max': dst_max, 'dst_min': dst_min, **placebo_opts_final.vsplacebo_dict()}

        if placebo_opts_final.peak_detect:
            # Tonemap
            tonemapped_clip = core.placebo.Tonemap(clip, **pl_tm_params)
        else:
            if not placebo_opts_final.is_hdr10_src():
                raise ValueError('Dynamic tonemapping using PlaneStats only supports HDR10 input clips')

            prop_src = __get_rgb_prop_src(clip, reference, target_list)

            tonemapped_clip = core.std.FrameEval(clip,
                                                 partial(__pl_dt,
                                                         clip=clip,
                                                         placebo_opts=placebo_opts_final,
                                                         pl_tm_params=pl_tm_params,
                                                         targets=target_list,
                                                         show=show),
                                                 prop_src=prop_src)

        if tonemapped_clip.format.color_family == vs.YUV:
            tonemapped_clip = core.std.SetFrameProps(tonemapped_clip, _Matrix=1, _Primaries=1, _Transfer=1)
    else:
        clip = core.resize.Spline36(clip,
                                    format=vs.RGB48,
                                    range_in_s="limited",
                                    range_s="full",
                                    matrix_in_s="2020ncl",
                                    matrix_s="rgb",
                                    primaries_in_s="2020",
                                    primaries_s="xyz",
                                    dither_type="none",
                                    chromaloc_in_s=chromaloc_in_s,
                                    chromaloc_s=chromaloc_in_s)

        prop_src = __get_rgb_prop_src(clip, reference, target_list)

        tonemapped_clip = core.std.FrameEval(clip,
                                             partial(__dt,
                                                     clip=clip,
                                                     show=show,
                                                     adjust_gamma=adjust_gamma,
                                                     targets=target_list),
                                             prop_src=prop_src)

    tonemapped_clip = core.resize.Spline36(tonemapped_clip,
                                           format=vs.YUV444P16,
                                           matrix_s="709",
                                           chromaloc_in_s=chromaloc_in_s,
                                           chromaloc_s=chromaloc_s)

    final_clip = None
    if src_fmt:
        final_clip = core.resize.Spline36(tonemapped_clip, format=clip_orig_format, dither_type="error_diffusion")
    else:
        final_clip = Depth(tonemapped_clip, 8)

    # Force props
    if final_clip.format.color_family == vs.YUV:
        final_clip = core.std.SetFrameProps(final_clip, _Matrix=1, _Primaries=1, _Transfer=1)
    else:
        final_clip = core.std.SetFrameProps(final_clip, _Matrix=0, _Primaries=1, _Transfer=1)

    return final_clip


def FillBorders(clip: vs.VideoNode,
                left: int = 0,
                right: int = 0,
                top: int = 0,
                bottom: int = 0,
                planes: Optional[Union[int, List[int]]] = None,
                mode: str = 'fixborders') -> vs.VideoNode:
    """
    FillBorders wrapper that automatically sets fixborders mode.
    If the chroma is subsampled, ceils the number of chroma rows to fill.
    This means that for 4:2:0 with left=1 1px luma and chroma is filled, 2px luma and 1px chroma for left=2,
    3px luma and 2px chroma for left=3 etc.
    """
    if planes is None:
        planes = [0, 1, 2]

    if clip.format.num_planes == 3:
        if isinstance(planes, int):
            planes = [planes]

        y, u, v = split(clip)

        if 0 in planes:
            y = y.fb.FillBorders(left=left, right=right, top=top, bottom=bottom, mode=mode)

        if clip.format.subsampling_w == 1:
            left, right = math.ceil(left / 2), math.ceil(right / 2)
        if clip.format.subsampling_h == 1:
            top, bottom = math.ceil(top / 2), math.ceil(bottom / 2)

        if 1 in planes:
            u = u.fb.FillBorders(left=left, right=right, top=top, bottom=bottom, mode=mode)
        if 2 in planes:
            v = v.fb.FillBorders(left=left, right=right, top=top, bottom=bottom, mode=mode)

        return join([y, u, v])

    return clip.fb.FillBorders(left=left, right=right, top=top, bottom=bottom, mode=mode)


#####################
# Utility functions #
#####################


def SelectRangeEvery(clip: vs.VideoNode,
                     every: int,
                     length: int,
                     offset: Optional[Union[int, List[int]]] = None) -> vs.VideoNode:
    """
    SelectRangeEvery, port from Avisynth's function.
    Offset can be an array with the first entry being the offset from the start and the second from the end.

    Usage:
    >>> # select <length> frames every <every> frames, starting at frame <offset>
    >>> SelectRangeEvery(clip, every, length, offset)
    """
    if offset is None:
        offset = [0, 0]
    if isinstance(offset, int):
        offset = [offset, 0]

    select = core.std.Trim(clip, first=offset[0], last=clip.num_frames - 1 - offset[1])
    select = core.std.SelectEvery(select, cycle=every, offsets=range(length))
    select = core.std.AssumeFPS(select, fpsnum=clip.fps.numerator, fpsden=clip.fps.denominator)

    return select


def FrameInfo(clip: vs.VideoNode,
              title: str,
              style: str = SUBTITLE_DEFAULT_STYLE,
              newlines: int = 3,
              pad_info: bool = False) -> vs.VideoNode:
    """
    FrameInfo. Frame number, frame type and title for the clip

    Usage:
    >>> # Print the frame number, the picture type and a title on each frame (hardcoded subtitle)
    >>> FrameInfo(clip, title)
    """

    def FrameProps(n: int, f: vs.VideoFrame, clip: vs.VideoNode, padding: Optional[str]) -> vs.VideoNode:
        if "_PictType" in f.props:
            info = f"Frame {n} of {clip.num_frames}\nPicture type: {f.props['_PictType'].decode()}"
        else:
            info = f"Frame {n} of {clip.num_frames}\nPicture type: N/A"

        if pad_info and padding:
            info_text = [padding + info]
        else:
            info_text = [info]

        clip = core.sub.Subtitle(clip, text=info_text, style=style)

        return clip

    padding_info: Optional[str] = None

    if pad_info:
        padding_info = " " + "".join(['\n'] * newlines)
        padding_title = " " + "".join(['\n'] * (newlines + 4))
    else:
        padding_title = " " + "".join(['\n'] * newlines)

    clip = core.std.FrameEval(clip, partial(FrameProps, clip=clip, padding=padding_info), prop_src=clip)
    clip = core.sub.Subtitle(clip, text=[padding_title + title], style=style)

    return clip


def InterleaveDir(folder: str,
                  PrintInfo: bool = False,
                  DelProp: bool = False,
                  first: Optional[vs.VideoNode] = None,
                  repeat: bool = False,
                  tonemap: bool = False,
                  source_filter: Optional[Callable[[Union[str, PathLike]], vs.VideoNode]] = None) -> vs.VideoNode:
    """
    InterleaveDir, load all mkv files located in a directory and interleave them.
    > Usage: InterleaveDir(folder, PrintInfo, DelProp, first, repeat)
      * folder is the folder path
      * PrintInfo = True prints the frame number, picture type and file name on each frame
      * DelProp = True means deleting primaries, matrix and transfer characteristics
      * first is an optional clip to append in first position of the interleaving list
      * repeat = True means that the appended clip is repeated between each loaded clip from the folder
      * tonemap = True tonemaps each clip before applying FrameInfo
      * source_filter = Source filter to use for loading clips.  Defaults to ffms2.
    """
    from pathlib import Path

    if source_filter is None:
        source_filter = core.ffms2.Source

    folder_path = Path(folder)
    files = sorted(folder_path.iterdir())

    if first is not None:
        sources = [first]
        j = 0
    else:
        sources = []
        j = -1

    for file in files:
        filename = file.name

        if file.is_file() and '.mkv' == file.suffix:

            j = j + 1
            sources.append(0)
            sources[j] = source_filter(folder + '/' + filename)

            if first is not None:
                sources[j] = core.std.AssumeFPS(clip=sources[j], src=first)

            if tonemap:
                sources[j] = DynamicTonemap(sources[j], libplacebo=False)

            if PrintInfo:
                sources[j] = FrameInfo(clip=sources[j], title=filename)
            elif PrintInfo is not False:
                raise TypeError('InterleaveDir: PrintInfo must be a boolean.')

            if DelProp is True:
                sources[j] = core.std.RemoveFrameProps(sources[j], props=["_Primaries", "_Matrix", "_Transfer"])
            elif DelProp is not False:
                raise TypeError('InterleaveDir: DelProp must be a boolean.')

            if first is not None and repeat:
                j = j + 1
                sources.append(0)
                sources[j] = first
            elif first is not None and repeat is not False:
                raise TypeError('InterleaveDir: repeat must be a boolean.')

    return core.std.Interleave(sources)


def ExtractFramesReader(clip: vs.VideoNode, csvfile: Union[str, PathLike]) -> vs.VideoNode:
    """
    ExtractFramesReader, reads a csv file to extract ranges of frames.

    Usage:
    >>> # csvfile is the path to a csv file containing in each row: <startframe> <endframe>
    >>> # the csv file may contain other columns, which will not be read
    >>> ExtractFramesReader(clip, csvfile)
    """
    import csv

    selec = core.std.BlankClip(clip=clip, length=1)

    with open(csvfile) as framescsv:
        csvzones = csv.reader(framescsv, delimiter=' ')
        for row in csvzones:
            start = row[0]
            end = row[1]

            selec = selec + core.std.Trim(clip, first=start, last=end)

    selec = core.std.Trim(selec, first=1)

    return selec


def fixlvls(clip: vs.VideoNode,
            gamma: float = 0.88,
            min_in: Optional[Union[int, List[Union[int, float]]]] = None,
            max_in: Optional[Union[int, List[Union[int, float]]]] = None,
            min_out: Optional[Union[int, List[Union[int, float]]]] = None,
            max_out: Optional[Union[int, List[Union[int, float]]]] = None,
            planes: Optional[Union[int, List[Union[int, float]]]] = None,
            input_depth: int = 8) -> vs.VideoNode:
    """
    A wrapper around std.Levels to fix what's commonly known as the gamma bug.
    :param clip: Processed clip.
    :param gamma: Gamma adjustment value.  Default of 0.88 is usually correct.
    :param min_in: Input minimum.
    :param max_in: Input maximum.
    :param min_out: Output minimum.
    :param max_out: Output maximum.
    :param planes: Planes to process.
    :param input_depth: Depth to scale values from.
    :return: Clip with gamma adjusted or levels fixed.
    """
    default_min = [16, 16]
    default_max = [235, 240]

    if min_in is None:
        min_in = default_min
    if max_in is None:
        max_in = default_max
    if min_out is None:
        min_out = default_min
    if max_out is None:
        max_out = default_max
    if planes is None:
        planes = [0]

    o_depth = clip.format.bits_per_sample
    clip = Depth(clip, 32)

    vals = [min_in.copy(), max_in.copy(), min_out.copy(), max_out.copy()]

    for (i, _) in enumerate(vals):
        if not isinstance(vals[i], list):
            vals[i] = [vals[i], vals[i]]
        for j in range(2):
            vals[i][j] = scale_value(vals[i][j], input_depth, 32, scale_offsets=True, chroma=j)

    if isinstance(planes, int):
        planes = [planes]

    chroma = planes.copy()
    if 0 in planes:
        clip = core.std.Levels(clip,
                               gamma=gamma,
                               min_in=vals[0][0],
                               max_in=vals[1][0],
                               min_out=vals[2][0],
                               max_out=vals[3][0],
                               planes=0)
        chroma.remove(0)
    if chroma:
        clip = core.std.Levels(clip,
                               gamma=gamma,
                               min_in=vals[0][1],
                               max_in=vals[1][1],
                               min_out=vals[2][1],
                               max_out=vals[3][1],
                               planes=chroma)

    return Depth(clip, o_depth)


def mt_lut(clip: vs.VideoNode, expr: str, planes: Optional[List[int]] = None) -> vs.VideoNode:
    """
    mt_lut, port from Avisynth's function.

    Usage:
    >>> # expr is an infix expression, not like avisynth's mt_lut which takes a postfix one
    >>> mt_lut(clip, expr, planes)
    """
    from ast import literal_eval

    if planes is None:
        planes = [0]

    minimum = 16 * ((1 << clip.format.bits_per_sample) - 1) // 256
    maximum = 235 * ((1 << clip.format.bits_per_sample) - 1) // 256

    def clampexpr() -> int:
        return int(max(minimum, min(round(literal_eval(expr)), maximum)))

    return core.std.Lut(clip=clip, function=clampexpr, planes=planes)


def UpscaleCheck(clip: vs.VideoNode, height: int = 720, kernel: str = 'spline36', interleave: bool = True, **kwargs):
    """
    Quick port of https://gist.github.com/pcroland/c1f1e46cd3e36021927eb033e5161298
    Really dumb "detail check" that exists because descaling
    Doesn't really work on a lot of live action stuff
    Output will always be more distorted than the input.

    Args:
        clip (vs.VideoNode): Your input clip
        height (int): Target resolution. Defaults to 720.
        kernel (str): Resampler of choice. Defaults to 'spline36'.
        interleave (bool): Interleave output with input. Defaults to True.

    Returns:
        [vs.VideoNode]: Resampled clip
    """

    # We can skip this if the input clip is int8, since zimg will handle this internally
    # If other resizers are ever added (fmtc, placebo), this should be reconsidered
    # But that's probably out of scope
    if 9 <= clip.format.bits_per_sample < 16:
        clip = Depth(clip, 16)

    # downsample & back up
    resample = zresize(clip, preset=height, kernel=kernel, **kwargs)
    resample = zresize(resample, width=clip.width, height=clip.height, kernel=kernel, **kwargs)

    if interleave:
        mix = core.std.Interleave([clip, resample])
        return Depth(mix, clip.format.bits_per_sample)

    return Depth(resample, clip.format.bits_per_sample)


def RescaleCheck(clip: vs.VideoNode,
                 res: int = 720,
                 kernel: str = "bicubic",
                 b: Optional[float] = None,
                 c: Optional[float] = None,
                 taps: Optional[int] = None,
                 bits: Optional[int] = None) -> vs.VideoNode:
    """
    Requires vapoursynth-descale: https://github.com/Irrational-Encoding-Wizardry/vapoursynth-descale
    :param res: Target resolution (720, 576, 480, ...)
    :param kernel: Rescale kernel, default bicubic
    :param b, c, taps: kernel params, default mitchell
    :param bits: Bit depth of output
    :return: Clip resampled to target resolution and back
    """
    DESCALEDICT: Dict = {
        'bilinear': core.descale.Debilinear,
        'bicubic': core.descale.Debicubic,
        'point': core.resize.Point,
        'lanczos': core.descale.Delanczos,
        'spline16': core.descale.Despline16,
        'spline36': core.descale.Despline36,
        'spline64': core.descale.Despline64
    }

    has_descale = HasLoadedPlugin("tegaf.asi.xe")

    if not has_descale:
        raise ModuleNotFoundError("RescaleCheck: Requires 'descale' plugin to be installed")

    src = clip
    # Generic error handling, YCOCG & COMPAT input not tested as such blocked by default
    if src.format.color_family not in [vs.YUV, vs.GRAY, vs.RGB]:
        raise TypeError("UpscaleCheck: Only supports YUV, GRAY or RGB input!")

    clip = core.resize.Spline36(src, format=vs.YUV444P16, matrix_s='709')

    if src.format.color_family is vs.RGB and src.format.bits_per_sample == 8:
        bits = 8
    elif bits is None:
        bits = src.format.bits_per_sample

    txt = kernel
    if taps:
        txt += (f", taps={taps}")
    elif b or c:
        txt += (f", b={b}, c={c}")

    if taps:
        b = taps
    elif not b:
        b = 1 / 3
    if not c:
        c = (1 - b) / 2

    b32 = Depth(clip, 32)

    lma = core.std.ShufflePlanes(b32, 0, vs.GRAY)
    dwn = zresize(lma, preset=res, kernel='point')  # lol
    w, h = dwn.width, dwn.height

    rsz = DESCALEDICT[kernel.lower()]

    if kernel.lower() == "bicubic":
        dwn = rsz(lma, w, h, b=b, c=c)
    elif kernel.lower() == "lanczos":
        dwn = rsz(lma, w, h, taps=taps)
    else:
        dwn = rsz(lma, w, h)

    ups = zresize(dwn, width=lma.width, height=lma.height, filter_param_a=b, filter_param_b=c, kernel=kernel)

    mrg = core.std.ShufflePlanes([ups, b32], [0, 1, 2], vs.YUV)
    txt = FrameInfo(mrg, txt)

    return Depth(txt, bits)


def Import(file: str, output_key: int = 0) -> vs.VideoNode:
    """
    Allows for easy import of your .vpy
    This will load a script even without set_output being present, however;
    if you would like to manipulate the clip you will need to specifiy an output.

    >>> script = Import(r'1080p.vpy')
    >>> script = FrameInfo(script, "Filtered")
    >>> script.set_output()

    Requires that the script contains `import vapoursynth as vs`.

    TODO: Find a way of keeping this more in-line with expected VS behavior (no output unless specified)
    """
    import inspect
    from importlib import machinery, util

    if file.endswith('.vpy'):
        loader = machinery.SourceFileLoader('script', file)
        spec = util.spec_from_loader(loader.name, loader)
        module = util.module_from_spec(spec)
        loader.exec_module(module)

        script_imports = inspect.getmembers(module, inspect.ismodule)
        imported_vs = any(info[0] == 'vs' and info[1].__name__ == 'vapoursynth' for info in script_imports)

        if not imported_vs:
            raise ValueError('The script does not import vapoursynth as vs')

        outputs: dict = module.vs.get_outputs()
        if outputs and output_key in outputs:
            output = outputs.get(output_key)

            try:
                if isinstance(output, vs.VideoOutputTuple):
                    return output.clip

                return output
            except AttributeError:
                return output
        else:
            raise ValueError(f'No outputs or invalid output key. Outputs: {[*outputs]}')
    else:
        raise TypeError("Import: Only .vpy is supported!")


def saturation(clip: vs.VideoNode, sat: float, dither_type: str = "error_diffusion") -> vs.VideoNode:
    if clip.format.color_family != vs.YUV:
        raise TypeError("saturation: YUV input is required!")

    sfmt = clip.format

    clip = Depth(clip, 32)
    expr = f"x {sat} * -0.5 max 0.5 min"

    return core.resize.Point(clip.std.Expr(["", expr]), format=sfmt, dither_type=dither_type)


def BorderResize(clip: vs.VideoNode,
                 ref: vs.VideoNode,
                 left: int = 0,
                 right: int = 0,
                 top: int = 0,
                 bottom: int = 0,
                 bb: List[int] = None,
                 planes: Optional[List[int]] = None,
                 sat: Optional[List[float]] = None) -> vs.VideoNode:
    """
    A wrapper to resize clips with borders.  This is meant for scenefiltering differing crops.
    :param ref: Reference resized clip, used purely to determine dimensions.
    :param left, right, top, bottom: Size of borders.  Uneven numbers will lead to filling.
    :param bb: bbmod args: [left, right, top, bottom, thresh, blur, planes]
    :param planes: Planes to be filled.
    :param sat: Saturation adjustment in AddBordersMod.  If None, std.AddBorders is used.
    """
    if planes is None:
        planes = [0, 1, 2]

    # save original dimensions
    ow, oh = clip.width, clip.height

    # we'll need the mod 2 values later for filling
    mod2 = [left % 2, right % 2, top % 2, bottom % 2]

    # figure out border size
    leftm = left - mod2[0]
    rightm = right - mod2[1]
    topm = top - mod2[2]
    bottomm = bottom - mod2[3]

    # use ref for output width and height
    rw, rh = ref.width, ref.height

    # crop off borders
    clip = clip.std.Crop(left=leftm, right=rightm, top=topm, bottom=bottomm)
    # fill it
    clip = FillBorders(clip, left=mod2[0], right=mod2[1], top=mod2[2], bottom=mod2[3], planes=planes)

    # optional bb call
    if bb:
        clip = bbmod(clip,
                     left=bb[0],
                     right=bb[1],
                     top=bb[2],
                     bottom=bb[3],
                     thresh=bb[4] if len(bb) > 4 else None,
                     blur=bb[5] if len(bb) > 4 else 999,
                     planes=bb[6] if len(bb) == 7 else None)

    # find new width and height
    nw = rw / ow * (ow - left - right)
    nh = rh / oh * (oh - top - bottom)

    # rounded versions
    rnw = round(nw / 2) * 2
    rnh = round(nh / 2) * 2

    # resize to that
    clip = zresize(clip, width=rnw, height=rnh, left=mod2[0], right=mod2[1], top=mod2[2], bottom=mod2[3])

    # now we figure out border size
    b_hor = (rw - rnw) / 2
    b_ver = (rh - rnh) / 2

    # shift image to top/left since we have to choose a place to shift shit to
    borders = []
    if left and right:
        borders += ([b_hor - b_hor % 2, b_hor + b_hor % 2])
    elif left:
        borders += ([b_hor * 2, 0])
    elif right:
        borders += ([0, b_hor * 2])
    else:
        borders += ([0, 0])

    if top and bottom:
        borders += ([b_ver - b_ver % 2, b_ver + b_ver % 2])
    elif top:
        borders += ([b_ver * 2, 0])
    elif bottom:
        borders += ([0, b_ver * 2])
    else:
        borders += ([0, 0])

    # add borders back
    if sat:
        return AddBordersMod(clip,
                             left=borders[0],
                             right=borders[1],
                             top=borders[2],
                             bottom=borders[3],
                             lsat=sat[0],
                             rsat=sat[1],
                             bsat=sat[2],
                             tsat=sat[3])

    return clip.std.AddBorders(left=borders[0], right=borders[1], top=borders[2], bottom=borders[3])


def RandomFrameNumbers(clip: vs.VideoNode,
                       num: int = 6,
                       start_offset: int = 1000,
                       end_offset: int = 10000,
                       output_file: Union[str, PathLike] = "screens.txt",
                       ftypes_first: Optional[Union[str, List[str]]] = None,
                       ftypes: Union[str, List[str]] = "B",
                       interleaved: Optional[int] = None,
                       clips: Optional[List[vs.VideoNode]] = None,
                       by_blocks: bool = True,
                       max_attempts: int = 50) -> vs.VideoNode:
    """
    Generates a list of random frame numbers, matched according to the specified frame types
    :param num: Amount of random frame numbers to generate
    :param start_offset: Amount of frames to skip from the start of the clip
    :param end_offset: Amount of frames to skip from the end of the clip
    :param output_file: Frame numbers output text file
    :param ftypes_first: Accepted frame types for `clip`
    :param ftypes: Accepted frame types of the compared clips
    :param interleaved: Number of interleaved clips, to allow matching from a single clip.
    :param clips: List of clips to be used for frame type matching, must be the same length as `clip`
    :param by_blocks: Split the random frames by equal length blocks, based on clip's `length` and `num`
    :param max_attempts: Number of frame type matching attempts in a row before ending recursion
    :return List of the generated random frame numbers
    """

    import os
    import random

    if ftypes_first is None:
        ftypes_first = ["P", "B"]

    def divisible_random(a: int, b: int, n: int) -> int:
        if not a % n:
            return random.choice(range(a, b, n))

        return random.choice(range(a + n - (a % n), b, n))

    # filter frame types function for frameeval
    def filter_ftype(
            n: int,  # pylint: disable=unused-argument
            f: vs.VideoFrame,
            clip: vs.VideoNode,
            frame_num: int,
            block: Dict) -> vs.VideoNode:
        match = False

        if isinstance(f, list):
            for i, p_src in enumerate(f):
                f_type = p_src.props["_PictType"].decode()

                if i == 0:
                    if f_type in ftypes_first:
                        match = True
                elif ftypes and f_type not in ftypes:
                    match = False
        else:
            # Single clip
            if f.props["_PictType"].decode() in ftypes_first:
                match = True

        if match:
            block['found_frame'] = frame_num

        return clip

    src_len = len(clip)

    if start_offset > src_len:
        raise ValueError(f'Invalid start offset: offset {start_offset} > {src_len} (clip len)')
    if end_offset > src_len:
        end_offset = 0

    # Multiply offsets for interleaved
    if interleaved:
        start_offset *= interleaved
        end_offset *= interleaved

    start, end = (start_offset, src_len - end_offset)

    if isinstance(ftypes, str):
        ftypes = [ftypes]

    if isinstance(ftypes_first, str):
        ftypes_first = [ftypes_first]

    # Blocks & found frames dicts
    generated_blocks: List[Dict] = []
    block_size = end - start

    if by_blocks:
        block_size = math.floor(block_size / num)

    for i in range(num):
        block_start = start

        if by_blocks:
            # Then first random is in range [start, start + block_size]
            block_start = start + (i * block_size)

        generated_blocks.append({
            'tested_frames': [],
            'found_frame': None,
            'block_start': block_start,
        })

    # Loop the number of frames we want
    for block in generated_blocks:
        block_start = block['block_start']
        block_end = block['block_start'] + block_size

        has_multiple_clips = (clips and isinstance(clips, list))

        if interleaved:
            has_multiple_clips = has_multiple_clips or interleaved > 1

        # Try matching if there are multiple clips or we want specific types
        if has_multiple_clips or ftypes_first or ftypes:
            with open(os.devnull, "wb") as f:
                while not block['found_frame']:
                    frame_num = random.randint(block_start, block_end)
                    clip_frames = []

                    if interleaved:
                        frame_num = divisible_random(block_start, block_end, interleaved)

                    if frame_num in block['tested_frames']:
                        continue

                    # Keep tested in memory
                    block['tested_frames'].append(frame_num)

                    if interleaved:
                        for i in range(0, interleaved):
                            clip_frames.append(clip[frame_num + i])
                    elif clips and isinstance(clips, list):
                        if len(clips) > 0:
                            clip_frames.append(clip[frame_num])

                            for other_clip in clips:
                                if len(other_clip) == len(clip):
                                    clip_frames.append(other_clip[frame_num])
                                else:
                                    raise ValueError('All compared clips must be the same length!')
                        else:
                            raise ValueError('Empty clips list!')

                    if clip_frames:
                        clip_f = clip_frames[0]
                    else:
                        clip_f = clip[frame_num]
                        clip_frames = list(clip_f)

                    clip_f = clip_f.std.FrameEval(partial(filter_ftype, clip=clip_f, frame_num=frame_num, block=block),
                                                  prop_src=clip_frames)
                    clip_f.output(f)

                    if block['found_frame'] and interleaved:
                        # Assumes the number was determined divisible already
                        block['found_frame'] = int(frame_num / interleaved)

                    if not block['found_frame'] and len(block['tested_frames']) >= max_attempts:
                        raise ValueError('Ended recursion because maximum matching attempts were reached')
        else:
            # Don't try matching frame types
            block['found_frame'] = random.randint(block_start, block_end)

    found_frames = sorted([b['found_frame'] for b in generated_blocks])

    if output_file:
        with open(output_file, "w") as txt:
            found_frames_lines = [f'{x}\n' for x in found_frames]
            txt.writelines(found_frames_lines)

    return found_frames


def HasLoadedPlugin(identifier: str) -> bool:
    return any(p.identifier == identifier for p in core.plugins())


def MapDolbyVision(base_layer: vs.VideoNode,
                   enhancement_layer: vs.VideoNode,
                   rpu: Optional[str] = None) -> vs.VideoNode:
    """
    Polynomial luma mapping, MMR chroma mapping on base layer.
    NLQ 12 bits restoration from enhancement layer.

    Input must be limited range, output is limited range.
    Both clips must have the same aspect ratio.

    Requires DolbyVisionRPU frame props to exist in both clips.
    Clips:
        `base_layer` clip: Profile 7 (BL+EL+RPU).
        `enhancement_layer` clip: Profile 7 (EL+RPU).

    Output: 12 bits mapped clip.

    Plugins needed: vs-placebo, vs-nlq.
    vs-placebo must be built with libdovi.
    """

    has_placebo = HasLoadedPlugin("com.vs.placebo")
    has_vsnlq = HasLoadedPlugin("com.vsnlq")

    if not has_placebo:
        raise ValueError('vs-placebo plugin must be installed!')
    if not has_vsnlq:
        raise ValueError("vs-nlq plugin must be installed!")

    base_16 = Depth(base_layer, 16)

    poly_mmr = core.placebo.Tonemap(
        base_16,
        src_csp=3,
        dst_csp=1,
    )
    poly_mmr = core.resize.Spline36(poly_mmr, format=vs.YUV420P16, chromaloc_in_s="top_left", chromaloc_s="top_left")

    scaled_el = core.resize.Point(enhancement_layer, width=base_layer.width, height=base_layer.height)

    return core.vsnlq.MapNLQ(poly_mmr, scaled_el, rpu=rpu)


def add_hdr_measurement_props(clip: vs.VideoNode,
                              maxrgb: bool = True,
                              hlg: bool = False,
                              as_nits: bool = True,
                              measurements: Optional[List[HdrMeasurement]] = None,
                              downscale: bool = True,
                              percentile: float = 100.0,
                              store_float: bool = False,
                              no_planestats: bool = False):
    """
    Measures the brightness of the frame using PlaneStats.
    Adds the info into props, and a list if available.

    The input clip is scaled down by 2x both horizontally and vertically.
    If the input is HLG, it is converted to PQ for measuring only.

    Props added:
        - `HDRMin`: Min brightness
        - `HDRMax`: Max brightness
        - `HDRAvg`: Average brightness
        - `HDRFALL`: Average of the frame's MaxRGB.
            - Only available when using `percentile` < 100 and `maxrgb=True`
            - Also when using `no_planestats` with percentile at 100%
        - `HDRMaxStd`: Standard deviation of the frame's MaxRGB
            - Only available when MaxFALL is calculated

        The values are PQ code by default.
        Can be converted to nits (cd/m^2) using `as_nits`.

    :param maxrgb: Whether to use MaxRGB or Luma for the brightness
    :param hlg: Whether the input clip is HLG
    :param as_nits: Set the prop to nits values instead of PQ codes
    :param measurements: List to store the measurements
    :param downscale: Downscale input by 2x both horizontally and vertically
    :param percentile: Compute percentile of the frame's max brightness (defaults to 100.0, actual max)
    :param store_float: If passing a list in `measurements`,
        whether to store the values as normalized float or int (16 bit)
    :param no_planestats: Always use the frame pixels and numpy to measure
    """
    import numpy as np

    no_numpy = math.isclose(percentile, 100.0) and not no_planestats

    def pq_props(n: int, f: list[vs.VideoFrame], maxrgb: bool, as_nits: bool, measurements: List[HdrMeasurement],
                 percentile: float):
        fout = f[0].copy()
        prop_src = f[1:]

        fall_pq = None
        fall_prop = None
        max_stdev_pq = None
        max_stdev_prop = None

        if no_numpy:
            # Using PlaneStats
            if maxrgb:
                min_pq = min(cmp.props["pqMin"] for cmp in prop_src)
                max_pq = max(cmp.props["pqMax"] for cmp in prop_src)
                avg_pq = max(cmp.props["pqAverage"] for cmp in prop_src)
            else:
                prop_src = prop_src[0]

                min_pq = prop_src.props["pqMin"]
                max_pq = prop_src.props["pqMax"]
                avg_pq = prop_src.props["pqAverage"]

            min_prop = min_pq / 65535.0
            max_prop = max_pq / 65535.0
            avg_prop = avg_pq
        else:
            prop_src = prop_src[0]

            if maxrgb:
                r_pixels = np.asarray(prop_src[0])
                g_pixels = np.asarray(prop_src[1])
                b_pixels = np.asarray(prop_src[2])

                rgb_pixels = [r_pixels, g_pixels, b_pixels]
                minrgb_pixels = np.minimum.reduce(rgb_pixels)
                maxrgb_pixels = np.maximum.reduce(rgb_pixels)

                min_pq = np.min(minrgb_pixels)
                max_pq = np.percentile(maxrgb_pixels, percentile)
                avg_pq = np.mean(rgb_pixels)

                fall_pq = np.mean(maxrgb_pixels)
                fall_prop = fall_pq / 65535.0
                max_stdev_pq = np.std(maxrgb_pixels)
                max_stdev_prop = max_stdev_pq / 65535.0
            else:
                y_pixels = np.asarray(prop_src[0])

                min_pq = np.min(y_pixels)
                max_pq = np.percentile(y_pixels, percentile)
                avg_pq = np.mean(y_pixels)

            min_prop = min_pq / 65535.0
            max_prop = max_pq / 65535.0
            avg_prop = avg_pq / 65535.0

        if as_nits:
            min_prop = st2084_eotf(min_prop) * ST2084_PEAK_LUMINANCE
            max_prop = st2084_eotf(max_prop) * ST2084_PEAK_LUMINANCE
            avg_prop = st2084_eotf(avg_prop) * ST2084_PEAK_LUMINANCE

            if fall_prop is not None:
                fall_prop = st2084_eotf(fall_prop) * ST2084_PEAK_LUMINANCE
            if max_stdev_prop is not None:
                max_stdev_prop = st2084_eotf(max_stdev_prop) * ST2084_PEAK_LUMINANCE

        fout.props["HDRMin"] = min_prop
        fout.props["HDRMax"] = max_prop
        fout.props["HDRAvg"] = avg_prop

        if fall_prop is not None:
            fout.props["HDRFALL"] = fall_prop
        if max_stdev_prop is not None:
            fout.props["HDRMaxStd"] = max_stdev_prop

        if measurements is not None:
            if store_float:
                min_pq /= 65535.0
                max_pq /= 65535.0

                # PlaneStats avg is already normalized
                if not no_numpy:
                    avg_pq /= 65535.0
                if fall_pq is not None:
                    fall_pq /= 65535.0
                if max_stdev_pq is not None:
                    max_stdev_pq /= 65535.0

            measurement = HdrMeasurement(frame=n,
                                         min=min_pq,
                                         max=max_pq,
                                         avg=avg_pq,
                                         fall=fall_pq,
                                         max_stdev=max_stdev_pq)
            measurements.append(measurement)

        return fout

    if hlg:
        clip = core.resize.Spline36(clip, transfer_in_s="std-b67", transfer_s="st2084")

    if maxrgb:
        scaled_format = vs.RGB48
    else:
        scaled_format = vs.YUV420P16

    final_w = clip.width
    final_h = clip.height

    if downscale:
        final_w /= 2
        final_h /= 2

    percentile = np.clip(percentile, 0.0, 100.0)

    scaled = core.resize.Spline36(clip,
                                  width=final_w,
                                  height=final_h,
                                  range_in_s="limited",
                                  range_s="full",
                                  transfer_in_s="st2084",
                                  transfer_s="st2084",
                                  primaries_in_s="2020",
                                  primaries_s="xyz",
                                  dither_type="none",
                                  chromaloc_in_s="top_left",
                                  format=scaled_format)

    if no_numpy:
        if maxrgb:
            r_props = core.std.PlaneStats(scaled, plane=0, prop='pq')
            g_props = core.std.PlaneStats(scaled, plane=1, prop='pq')
            b_props = core.std.PlaneStats(scaled, plane=2, prop='pq')
            prop_src = [r_props, g_props, b_props]
        else:
            y_props = core.std.PlaneStats(scaled, plane=0, prop='pq')
            prop_src = [y_props]
    else:
        prop_src = [scaled]

    fn_partial = partial(pq_props, maxrgb=maxrgb, as_nits=as_nits, measurements=measurements, percentile=percentile)

    return core.std.ModifyFrame(clip, clips=[clip] + prop_src, selector=fn_partial)


#####################
#      Aliases      #
#####################

rfs = ReplaceFrames
fb = FillBorders

zr = zresize

br = BorderResize
borderresize = BorderResize

#####################
#      Exports      #
#####################

__all__ = [
    "AddBordersMod",
    "add_hdr_measurement_props",
    "BorderResize",
    "DebandReader",
    "Depth",
    "DynamicTonemap",
    "ExtractFramesReader",
    "FillBorders",
    "FrameInfo",
    "HasLoadedPlugin",
    "Import",
    "InterleaveDir",
    "MapDolbyVision",
    "RandomFrameNumbers",
    "ReplaceFrames",
    "RescaleCheck",
    "ScreenGen",
    "ScreenGenPrefix",
    "ScreenGenEncoder",
    "SelectRangeEvery",
    "UpscaleCheck",
    "bbmod",
    "bbmoda",
    "borderresize",
    "br",
    "fb",
    "fixlvls",
    "mt_lut",
    "rfs",
    "saturation",
    "st2084_eotf",
    "st2084_inverse_eotf",
    "zr",
    "zresize",
]
