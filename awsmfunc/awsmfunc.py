from enum import Enum
import vapoursynth as vs
from vapoursynth import core

import math
from os import PathLike
from functools import partial

from typing import Callable, Dict, List, Union, Optional, Any

from vsutil import plane, get_depth, split, join, scale_value
from vsutil import depth as vsuDepth
from rekt import rektlvls, rekt_fast
"""
To-do list:

 - CropResize: default chroma fill might make more sense
 - CropResizeReader needs cfill
"""

SUBTITLE_DEFAULT_STYLE: str = "sans-serif,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1"

ST2084_PEAK_LUMINANCE = 10000
ST2084_M1 = 0.1593017578125
ST2084_M2 = 78.84375
ST2084_C1 = 0.8359375
ST2084_C2 = 18.8515625
ST2084_C3 = 18.6875


def Depth(clip: vs.VideoNode, bits: int, **kwargs) -> vs.VideoNode:
    """
    This is just a vsutil.depth wrapper that doesn't dither for high bit depth.
    The reason for this is that repeated error diffusion can be destructive, so if we assume most filtering is done at
    high bit depth (which it should), this should keep that from happening while still performing it before outputting.
    """
    if bits < 16:
        return vsuDepth(clip, bits, dither_type="error_diffusion", **kwargs)
    else:
        return vsuDepth(clip, bits, dither_type="none", **kwargs)


def FixColumnBrightnessProtect2(clip: vs.VideoNode, column: int, adj_val: int = 0, prot_val: int = 20) -> vs.VideoNode:
    return FixBrightnessProtect2(clip, column=column, adj_column=adj_val, prot_val=prot_val)


def FixRowBrightnessProtect2(clip: vs.VideoNode, row: int, adj_val: int = 0, prot_val: int = 20) -> vs.VideoNode:
    return FixBrightnessProtect2(clip, row=row, adj_row=adj_val, prot_val=prot_val)


def FixBrightnessProtect2(clip: vs.VideoNode,
                          row: Optional[Union[int, List[int]]] = None,
                          adj_row: Optional[Union[int, List[int]]] = None,
                          column: Optional[Union[int, List[int]]] = None,
                          adj_column: Optional[Union[int, List[int]]] = None,
                          prot_val: int = 20) -> vs.VideoNode:
    return rektlvls(clip, rownum=row, rowval=adj_row, colnum=column, colval=adj_column, prot_val=prot_val)


def FixColumnBrightness(clip: vs.VideoNode,
                        column: int,
                        input_low: int = 16,
                        input_high: int = 235,
                        output_low: int = 16,
                        output_high: int = 235) -> vs.VideoNode:
    hbd = Depth(clip, 16)
    lma = hbd.std.ShufflePlanes(0, vs.GRAY)
    adj = lambda x: core.std.Levels(
        x, min_in=input_low << 8, max_in=input_high << 8, min_out=output_low << 8, max_out=output_high << 8, planes=0)

    prc = rekt_fast(lma, adj, left=column, right=clip.width - column - 1)

    if clip.format.color_family is vs.YUV:
        prc = core.std.ShufflePlanes([prc, hbd], [0, 1, 2], vs.YUV)

    return Depth(prc, clip.format.bits_per_sample)


def FixRowBrightness(clip: vs.VideoNode,
                     row: int,
                     input_low: int = 16,
                     input_high: int = 235,
                     output_low: int = 16,
                     output_high: int = 235) -> vs.VideoNode:
    hbd = Depth(clip, 16)
    lma = hbd.std.ShufflePlanes(0, vs.GRAY)
    adj = lambda x: core.std.Levels(
        x, min_in=input_low << 8, max_in=input_high << 8, min_out=output_low << 8, max_out=output_high << 8, planes=0)

    prc = rekt_fast(lma, adj, top=row, bottom=clip.height - row - 1)

    if clip.format.color_family is vs.YUV:
        prc = core.std.ShufflePlanes([prc, hbd], [0, 1, 2], vs.YUV)

    return Depth(prc, clip.format.bits_per_sample)


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
                    mappings += '\n{}'.format(mf.read())
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
    quietvoid's bbmod helper for a significant speedup from cropping unnecessary pixels before processing.
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

    if clip.format.color_family != vs.YUV and clip.format.color_family != vs.GRAY:
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

        if 0 in planes:
            y = True
        else:
            y = False

        if 1 in planes:
            u = True
        else:
            u = False

        if 2 in planes:
            v = True
        else:
            v = False
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
    From sgvsfunc. I'm not updating the doc strings, here, read bbmod instead.
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
        1) At default values ​​of thresh = 128 blur = 999:
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

    if isinstance(thresh, int) or isinstance(thresh, float):
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

        for t in thresh[1:]:
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
            uvexpr.append(
                f"{uvexpr_} x - {thresh[t]} > x {thresh[t]} + {uvexpr_} x - -{thresh[t]} < x {thresh[t]} - {uvexpr_} ? ?"
            )
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
        else:
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

                last = core.resize.Bicubic(original,
                                           blurWidth[0] * csize,
                                           cTop * csize,
                                           filter_param_a=1,
                                           filter_param_b=0)

                originalBlur = last.resize.Bicubic(cWidth * csize, cTop * csize, filter_param_a=1, filter_param_b=0)
                balancedLuma = core.std.Expr(clips=[original, originalBlur, referenceBlur], expr=yexpr)

                yplane = core.std.StackVertical(clips=[
                    balancedLuma,
                    core.std.CropAbs(c2, cWidth * csize, (cHeight - cTop) * csize, 0, cTop * csize)
                ]).resize.Point(cWidth, cHeight)

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

                    balancedChroma = core.std.Expr(clips=[original, originalBlurChroma, referenceBlurChroma],
                                                   expr=expruv)
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
    VapourSynth port of AddBordersMod.  Replacement for BlackBorders from sgvsfunc.
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


def BlackBorders(clip: vs.VideoNode,
                 left: int = 0,
                 right: int = 0,
                 top: int = 0,
                 bottom: int = 0,
                 lsat: float = 0.88,
                 rsat: Optional[float] = None,
                 tsat: float = 0.2,
                 bsat: float = 0.2,
                 color: Union[str, List[float]] = None) -> vs.VideoNode:
    return AddBordersMod(clip, left, top, right, bottom, lsat, tsat, rsat, bsat, color)


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

    if preset:
        if clip.width / clip.height > ar:
            return zresize(clip,
                           width=int(ar * preset),
                           left=left,
                           right=right,
                           top=top,
                           bottom=bottom,
                           kernel=kernel,
                           **kwargs)
        else:
            return zresize(clip, height=preset, left=left, right=right, top=top, bottom=bottom, kernel=kernel, **kwargs)

    if (width is None) and (height is None):
        width = clip.width
        height = clip.height
        rh = rw = 1
    elif width is None:
        rh = rw = height / (clip.height - top - bottom)
    elif height is None:
        rh = rw = width / (clip.width - left - right)
    else:
        rh = height / clip.height
        rw = width / clip.width

    w = round(((clip.width - left - right) * rw) / 2) * 2
    h = round(((clip.height - top - bottom) * rh) / 2) * 2

    resizer = RESIZEDICT[kernel.lower()]
    return resizer(clip=clip,
                   width=w,
                   height=h,
                   src_left=left,
                   src_top=top,
                   src_width=clip.width - left - right,
                   src_height=clip.height - top - bottom,
                   dither_type="error_diffusion",
                   **kwargs)


def CropResize(clip: vs.VideoNode,
               preset: Optional[int] = None,
               width: Optional[int] = None,
               height: Optional[int] = None,
               left: int = 0,
               right: int = 0,
               top: int = 0,
               bottom: int = 0,
               bb: Any = None,
               fill: Optional[List[int]] = None,
               fillplanes: Optional[List[int]] = None,
               cfill: Optional[Union[List[int], Callable[[vs.VideoNode], vs.VideoNode]]] = None,
               resizer: str = "spline36",
               filter_param_a: Union[int, float] = None,
               filter_param_b: Union[int, float] = None,
               aspect_ratio: float = 16 / 9) -> vs.VideoNode:
    """
    Originally from sgvsfunc.  Added chroma filling option and preset parameter.
    This function is a wrapper around cropping and resizing with the option to fill and remove columns/rows.
    :param clip: Clip to be processed.
    :param preset: Desired output height as if output clip was 16 / 9, calculates width and height.
                   E.g. 1920x872 source with preset=720p will output 1280x582.
    :param width: Width of output clip.  If height is specified without width, width is auto-calculated.
    :param height: Height of output clip.  If width is specified without height, height is auto-calculated.
    :param left: Left offset of resized clip.
    :param right: Right offset of resized clip.
    :param top: Top offset of resized clip.
    :param bottom: Bottom offset of resized clip.
    :param bb: Parameters to be parsed to bbmod: cTop, cBottom, cLeft, cRight[, thresh=128, blur=999].
    :param fill: Parameters to be parsed to fb.FillBorders: left, right, top, bottom.
    :param fillplanes: Planes for fill to be applied to.
    :param cfill: If a list is specified, same as fill for chroma planes exclusively.  Else, a lambda function can be
                  specified, e.g. cfill=lambda c: c.edgefixer.ContinuityFixer(left=0, top=0, right=[2, 4, 4], bottom=0).
    :param resizer: Resize kernel to be used.
    :param filter_param_a, filter_param_b: Filter parameters for internal resizers, b & c for bicubic, taps for lanczos.
    :return: Resized clip.
    """
    from warnings import warn
    warn(
        "CropResize: this function is deprecated.  Please fix your borders separately and resize via awsmfunc.zresize.")
    if preset:
        if clip.width / clip.height > aspect_ratio:
            return CropResize(clip,
                              width=int(aspect_ratio * preset),
                              left=left,
                              right=right,
                              top=top,
                              bottom=bottom,
                              bb=bb,
                              fill=fill,
                              fillplanes=fillplanes,
                              cfill=cfill,
                              resizer=resizer,
                              filter_param_a=filter_param_a,
                              filter_param_b=filter_param_b)
        else:
            return CropResize(clip,
                              height=preset,
                              left=left,
                              right=right,
                              top=top,
                              bottom=bottom,
                              bb=bb,
                              fill=fill,
                              fillplanes=fillplanes,
                              cfill=cfill,
                              resizer=resizer,
                              filter_param_a=filter_param_a,
                              filter_param_b=filter_param_b)
    if fill is None:
        fill = [0, 0, 0, 0]
    if fillplanes is None:
        fillplanes = [0, 1, 2]
    if isinstance(fill, list):
        if len(fill) == 4:
            if left - int(fill[0]) >= 0 and right - int(fill[1]) >= 0 and top - int(fill[2]) >= 0 and bottom - int(
                    fill[3]) >= 0:
                left = left - int(fill[0])
                right = right - int(fill[1])
                top = top - int(fill[2])
                bottom = bottom - int(fill[3])
            else:
                raise ValueError('CropResize: filling exceeds cropping.')
        else:
            raise TypeError('CropResize: fill arguments not valid.')

    lr = left % 2
    rr = right % 2
    tr = top % 2
    br = bottom % 2

    if bb:
        if len(bb) == 4:
            bb.append(None)
            bb.append(999)
        elif len(bb) != 6:
            raise TypeError('CropResize: bbmod arguments not valid.')

    if left or right or top or bottom:
        cropeven = core.std.Crop(clip, left=left - lr, right=right - rr, top=top - tr, bottom=bottom - br)
        if fill is not False and (lr or rr or tr or br):
            cropeven = fb(cropeven,
                          left=lr + int(fill[0]),
                          right=rr + int(fill[1]),
                          top=tr + int(fill[2]),
                          bottom=br + int(fill[3]),
                          planes=fillplanes)
    else:
        cropeven = clip

    if cfill:
        y, u, v = split(cropeven)
        if isinstance(cfill, list):
            u = core.fb.FillBorders(u, cfill[0], cfill[1], cfill[2], cfill[3], mode="fillmargins")
            v = core.fb.FillBorders(v, cfill[0], cfill[1], cfill[2], cfill[3], mode="fillmargins")
        else:
            u = cfill(u)
            v = cfill(v)
        cropeven = core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV)

    if bb:
        bb = [
            int(bb[0]) + lr + int(fill[0]),
            int(bb[1]) + rr + int(fill[1]),
            int(bb[2]) + tr + int(fill[2]),
            int(bb[3]) + br + int(fill[3]),
            int(bb[4]),
            int(bb[5])
        ]
        cropeven = bbmod(cropeven,
                         cTop=int(bb[2]) + tr,
                         cBottom=int(bb[3]) + br,
                         cLeft=int(bb[0]) + lr,
                         cRight=int(bb[1]) + rr,
                         thresh=int(bb[4]),
                         blur=int(bb[5]),
                         scale_thresh=True)

    return zresize(clip=cropeven,
                   width=width,
                   height=height,
                   left=lr,
                   top=tr,
                   right=rr,
                   bottom=br,
                   filter_param_a=filter_param_a,
                   filter_param_b=filter_param_b,
                   kernel=resizer)


def CropResizeReader(clip: vs.VideoNode,
                     csvfile: Union[str, PathLike],
                     width: Optional[int] = None,
                     height: Optional[int] = None,
                     row: Optional[Union[int, List[int]]] = None,
                     adj_row: Optional[Union[int, List[int]]] = None,
                     column: Optional[Union[int, List[int]]] = None,
                     adj_column: Optional[Union[int, List[int]]] = None,
                     fill_max: int = 2,
                     bb: Any = None,
                     FixUncrop: Optional[List[int]] = None,
                     resizer: str = "spline36") -> vs.VideoNode:
    """
    CropResizeReader, cropResize for variable borders by loading crop values from a csv file
      Also fill small borders and fix brightness/apply bbmod relatively to the variable border
      From sgvsfunc.
    > Usage: CropResizeReader(clip, csvfile, width, height, row, adj_row, column, adj_column, fill_max, bb, FixUncrop, resizer)
      * csvfile is the path to a csv file containing in each row: <startframe> <endframe> <left> <right> <top> <bottom>
        where left, right, top, bottom are the number of pixels to crop
        Optionally, the number of pixels to fill can be appended to each line <left> <right> <top> <bottom> in order to reduce the black borders.
        Filling can be useful in case of small borders to equilibriate the number of pixels between right/top and left/bottom after resizing.
      * width and height are the dimensions of the resized clip
        If none of them is indicated, no resizing is performed. If only one of them is indicated, the other is deduced.
      * row, adj_row, column, adj_column are lists of values to use FixBrightnessProtect2, where row/column is relative to the border defined by the cropping
      * Borders <=fill_max will be filled instead of creating a black border
      * bb is a list containing bbmod values [cLeft, cRight, cTop, cBottom, thresh, blur] where thresh and blur are optional.
        Mind the order: it is different from usual cTop, cBottom, cLeft, cRight
      * FixUncrop is a list of 4 booleans [left right top bottom]
        False means that FixBrightness/bbmod is only apply where crop>0, True means it is applied on the whole clip
      * resizer should be Bilinear, Bicubic, Point, Lanczos, Spline16 or Spline36 (default)
    """
    if FixUncrop is None:
        FixUncrop = [False, False, False, False]
    import csv

    if len(FixUncrop) != 4:
        raise TypeError('CropResizeReader: FixUncrop argument not valid.')

    if (width is None) and (height is None):
        width = clip.width
        height = clip.height
        rh = rw = 1
    elif width is None:
        rh = rw = height / clip.height
        width = round((clip.width * rh) / 2) * 2
    elif height is None:
        rh = rw = width / clip.width
        height = round((clip.height * rw) / 2) * 2
    else:
        rh = height / clip.height
        rw = width / clip.width

    filtered = clip

    if bb is not None:
        if len(bb) == 4:
            bb.append(None)
            bb.append(999)
        elif len(bb) != 6:
            raise TypeError('CropResizeReader: bbmod arguments not valid.')
        bbtemp = [bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]]
        if FixUncrop[0] is False:
            bbtemp[0] = 0
        if FixUncrop[1] is False:
            bbtemp[1] = 0
        if FixUncrop[2] is False:
            bbtemp[2] = 0
        if FixUncrop[3] is False:
            bbtemp[3] = 0

        filtered = bbmod(filtered,
                         cTop=bbtemp[2],
                         cBottom=bbtemp[3],
                         cLeft=bbtemp[0],
                         cRight=bbtemp[1],
                         thresh=bbtemp[4],
                         blur=bbtemp[5])

    resized = core.resize.Spline36(clip=filtered, width=width, height=height, dither_type="error_diffusion")

    with open(csvfile) as cropcsv:
        cropzones = csv.reader(cropcsv, delimiter=' ')
        for zone in cropzones:

            cl = int(zone[2])
            cr = int(zone[3])
            ct = int(zone[4])
            cb = int(zone[5])

            filteredtemp = clip

            if row is not None:
                if not isinstance(row, list):
                    row = [int(row)]
                    adj_row = [int(adj_row)]
                for i in range(len(row)):
                    if row[i] < 0:
                        if FixUncrop[3] is True or cb > 0:
                            filteredtemp = FixBrightnessProtect2(clip=filteredtemp,
                                                                 row=int(row[i]) - cb,
                                                                 adj_row=adj_row[i])
                    else:
                        if FixUncrop[2] is True or ct > 0:
                            filteredtemp = FixBrightnessProtect2(clip=filteredtemp,
                                                                 row=ct + int(row[i]),
                                                                 adj_row=adj_row[i])

            if column is not None:
                if not isinstance(column, list):
                    column = [int(column)]
                    adj_column = [int(adj_column)]
                for j in range(len(column)):
                    if column[j] < 0:
                        if FixUncrop[1] is True or cr > 0:
                            filteredtemp = FixBrightnessProtect2(clip=filteredtemp,
                                                                 column=int(column[j]) - cr,
                                                                 adj_column=adj_column[j])
                    else:
                        if FixUncrop[0] is True or cl > 0:
                            filteredtemp = FixBrightnessProtect2(clip=filteredtemp,
                                                                 column=cl + int(column[j]),
                                                                 adj_column=adj_column[j])

            bbtemp = None
            if bb is not None:
                bbtemp = [bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]]
                if FixUncrop[0] is False and cl == 0:
                    bbtemp[0] = 0
                if FixUncrop[1] is False and cr == 0:
                    bbtemp[1] = 0
                if FixUncrop[2] is False and ct == 0:
                    bbtemp[2] = 0
                if FixUncrop[3] is False and cb == 0:
                    bbtemp[3] = 0

            if cl > 0 and cl <= fill_max:
                filteredtemp = core.fb.FillBorders(filteredtemp, left=cl, mode="fillmargins")
                if bbtemp is not None:
                    bbtemp[0] = int(bbtemp[0]) + cl
                cl = 0

            if cr > 0 and cr <= fill_max:
                filteredtemp = core.fb.FillBorders(filteredtemp, right=cr, mode="fillmargins")
                if bbtemp is not None:
                    bbtemp[1] = int(bbtemp[1]) + cr
                cr = 0

            if ct > 0 and ct <= fill_max:
                filteredtemp = core.fb.FillBorders(filteredtemp, top=ct, mode="fillmargins")
                if bbtemp is not None:
                    bbtemp[2] = int(bbtemp[2]) + ct
                ct = 0

            if cb > 0 and cb <= fill_max:
                filteredtemp = core.fb.FillBorders(filteredtemp, bottom=cb, mode="fillmargins")
                if bbtemp is not None:
                    bbtemp[3] = int(bbtemp[3]) + cb
                cb = 0

            if len(zone) == 6:
                fill = [0, 0, 0, 0]
            elif len(zone) == 10:
                fill = [int(zone[6]), int(zone[7]), int(zone[8]), int(zone[9])]
            else:
                raise TypeError('CropResizeReader: csv file not valid.')

            resizedcore = CropResize(filteredtemp,
                                     width=width,
                                     height=height,
                                     left=cl,
                                     right=cr,
                                     top=ct,
                                     bottom=cb,
                                     bb=bbtemp,
                                     fill=fill,
                                     resizer=resizer)

            x = round((cl * rw) / 2) * 2
            y = round((ct * rh) / 2) * 2
            resizedfull = BlackBorders(resizedcore,
                                       left=x,
                                       right=width - resizedcore.width - x,
                                       top=y,
                                       bottom=height - resizedcore.height - y)

            maps = "[" + zone[0] + " " + zone[1] + "]"
            resized = ReplaceFrames(resized, resizedfull, mappings=maps)
            filtered = ReplaceFrames(filtered, filteredtemp, mappings=maps)

    return resized


def DebandReader(clip: vs.VideoNode,
                 csvfile: Union[str, PathLike],
                 grain: int = 64,
                 range: int = 30,
                 delimiter: str = ' ',
                 mask: Optional[vs.VideoNode] = None) -> vs.VideoNode:
    """
    DebandReader, read a csv file to apply a f3kdb filter for given strengths and frames. From sgvsfunc.
    > Usage: DebandReader(clip, csvfile, grain, range)
      * csvfile is the path to a csv file containing in each row: <startframe> <endframe> <strength>
      * grain is passed as grainy and grainc in the f3kdb filter
      * range is passed as range in the f3kdb filter
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
                                   range=range,
                                   output_depth=depth)

            filtered = ReplaceFrames(filtered, db, mappings="[" + row[0] + " " + row[1] + "]")

        if mask:
            filtered = core.std.MaskedMerge(clip, filtered, mask)

    return filtered


def LumaMaskMerge(clipa: vs.VideoNode,
                  clipb: vs.VideoNode,
                  threshold: Optional[Union[int, float]] = None,
                  invert: bool = False,
                  scale_inputs: bool = False,
                  planes: Union[int, List[int]] = 0) -> vs.VideoNode:
    """
    LumaMaskMerge, merges clips using a binary mask defined by a brightness level. From sgvsfunc, with added planes.
    > Usage: LumaMaskMerge(clipa, clipb, threshold, invert, scale_inputs)
      * threshold is the brightness level. clipb is applied where the brightness is below threshold
      * If invert = True, clipb is applied where the brightness is above threshold
      * scale_inputs = True scales threshold from 8bits to current bit depth.
      * Use planes to specify which planes should be merged from clipb into clipa. Default is first plane.
    """
    p = (1 << clipa.format.bits_per_sample) - 1

    if scale_inputs and threshold is not None:
        threshold = scale_value(threshold, 8, clipa.format.bits_per_sample)
    elif threshold is None:
        threshold = (p + 1) / 2

    if not invert:
        mask = core.std.Binarize(clip=clipa.std.ShufflePlanes(0, vs.GRAY), threshold=threshold, v0=p, v1=0)
    else:
        mask = core.std.Binarize(clip=clipa.std.ShufflePlanes(0, vs.GRAY), threshold=threshold, v0=0, v1=p)

    return core.std.MaskedMerge(clipa=clipa, clipb=clipb, mask=mask, planes=planes)


def RGBMaskMerge(clipa: vs.VideoNode,
                 clipb: vs.VideoNode,
                 Rmin: int,
                 Rmax: int,
                 Gmin: int,
                 Gmax: int,
                 Bmin: int,
                 Bmax: int,
                 scale_inputs: bool = False) -> vs.VideoNode:
    """
    RGBMaskMerge, merges clips using a binary mask defined by a RGB range. From sgvsfunc.
    > Usage: RGBMaskMerge(clipa, clipb, Rmin, Rmax, Gmin, Gmax, Bmin, Bmax, scale_inputs)
      * clipb is applied where Rmin < R < Rmax and Gmin < G < Gmax and Bmin < B < Bmax
      * scale_inputs = True scales Rmin, Rmax, Gmin, Gmax, Bmin, Bmax from 8bits to current bit depth (8, 10 or 16).
    """
    p = (1 << clipa.format.bits_per_sample) - 1

    if scale_inputs:
        Rmin = scale_value(Rmin, 8, clipa.format.bits_per_sample)
        Rmax = scale_value(Rmax, 8, clipa.format.bits_per_sample)
        Gmin = scale_value(Gmin, 8, clipa.format.bits_per_sample)
        Gmax = scale_value(Gmax, 8, clipa.format.bits_per_sample)
        Bmin = scale_value(Bmin, 8, clipa.format.bits_per_sample)
        Bmax = scale_value(Bmax, 8, clipa.format.bits_per_sample)

    if clipa.format.bits_per_sample == 8:
        rgb = core.resize.Point(clipa, format=vs.RGB24, matrix_in_s="709")
    elif clipa.format.bits_per_sample == 10:
        rgb = core.resize.Point(clipa, format=vs.RGB30, matrix_in_s="709")
    elif clipa.format.bits_per_sample == 16:
        rgb = core.resize.Point(clipa, format=vs.RGB48, matrix_in_s="709")
    else:
        raise TypeError('RGBMaskMerge: only applicable to 8, 10 and 16 bits clips.')

    R = GetPlane(rgb, 0)
    G = GetPlane(rgb, 1)
    B = GetPlane(rgb, 2)
    rgbmask = core.std.Expr(
        clips=[R, G, B],
        expr=[f"x {Rmin} > x {Rmax} < y {Gmin} > y {Gmax} < z {Bmin} > z {Bmax} < and and and and and {p} 0 ?"])

    merge = core.std.MaskedMerge(clipa=clipa, clipb=clipb, mask=rgbmask)
    clip = core.std.ShufflePlanes(clips=[merge, merge, clipb], planes=[0, 1, 2], colorfamily=vs.YUV)

    return clip


class ScreenGenPrefix(str, Enum):
    Sequential = 'seq'
    FrameNo = 'frame'

def ScreenGen(clip: vs.VideoNode,
              folder: Union[str, PathLike],
              suffix: str,
              prefix: Union[ScreenGenPrefix, str] = ScreenGenPrefix.Sequential,
              frame_numbers: Union[Union[str, PathLike], List[int]] = "screens.txt",
              start: int = 1,
              delim: str = ' ') -> None:
    """
    quietvoid's screenshot generator.
    Generates screenshots from a list of frame numbers
    folder is the folder name that is created
    suffix is the final name appended
    frame_numbers is the list of frames, defaults to a file named screens.txt. Either a list or a file
    start is the number at which the filenames start

    > Usage: ScreenGen(src, "Screenshots", "a")
             ScreenGen(enc, "Screenshots", "b")
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

    if screens:
        if not folder_path.is_dir():
            folder_path.mkdir()

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

            print(end=log_str)
            try:
                core.imwri.Write(rgb_clip, "PNG24", final_path, overwrite=True).get_frame(num)
            except vs.Error:
                new_path = folder_path.joinpath(f'%d{suffix}.png').resolve()
                core.imwri.Write(rgb_clip, "PNG24", new_path, overwrite=True).get_frame(num)
    else:
        raise ValueError('ScreenGen: No screenshots to write to disk')


def DynamicTonemap(clip: vs.VideoNode,
                   show: bool = False,
                   src_fmt: bool = False,
                   libplacebo: bool = True,
                   placebo_dt: bool = True,
                   placebo_algo: int = 3,
                   placebo_param: Optional[float] = None,
                   adjust_gamma: bool = False,
                   chromaloc_in_s: str = "top_left",
                   chromaloc_s: str = "top_left",
                   reference: Optional[vs.VideoNode] = None,
                   predetermined_targets: Optional[Union[str, List[Union[int, float]]]] = None) -> vs.VideoNode:
    """
    quietvoid's dynamic tonemapping function.
    The clip (or reference) is blurred, then plane stats are measured.
    The tonemapping is then done according to the max RGB value.

    :param clip: PQ BT.2020 clip.
    :param show: Whether to show nits values.
    :param src_fmt: Whether to output source bit depth instead of 8-bit 4:4:4.
    :param libplacebo: Whether to use libplacebo as tonemapper
        Requires vs-placebo plugin.
    :param placebo_dt: Use libplacebo's dynamic peak detection instead of FrameEval
    :param placebo_algo: The tonemapping algo to use
    :param adjust_gamma: Adjusts gamma/saturation dynamically on low brightness areas when target nits are high.
        Requires adaptivegrain-rs plugin.
    :param chromaloc_in_s: Chromaloc of input
    :param chromaloc_s: Chromaloc of output
    :param reference: Reference clip to calculate target brightness with.
        Use cases include source/encode comparisons
    :param predetermined_targets: List of target nits per frame.
        List of numbers or file containing a target per line
        Must be equal to clip length
    :return: SDR YUV444P16 clip by default.
    """

    from pathlib import Path

    REF_WHITE = 203.0
    TARGET_NITS = 100.0

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
                                                    chromaloc_s=chromaloc_s)
            else:
                clip_to_blur = clip = core.resize.Spline36(reference,
                                                           format=vs.RGB48,
                                                           chromaloc_in_s=chromaloc_in_s,
                                                           chromaloc_s=chromaloc_s)
        else:
            clip_to_blur = clip

        try:
            blurred_clip = core.bilateral.Bilateral(clip_to_blur, sigmaS=1)
        except vs.Error:
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
                            range: Optional[str] = None) -> vs.VideoNode:
        max_value = 65535.0

        if range:
            if range == 'limited':
                max_value = 60395

        if not targets:
            r_max = st2084_eotf((f[0].props['PlaneStatsMax'] / max_value)) * ST2084_PEAK_LUMINANCE
            g_max = st2084_eotf((f[1].props['PlaneStatsMax'] / max_value)) * ST2084_PEAK_LUMINANCE
            b_max = st2084_eotf((f[2].props['PlaneStatsMax'] / max_value)) * ST2084_PEAK_LUMINANCE

            max_rgb = round(max([r_max, g_max, b_max]))
        else:
            max_rgb = round(targets[n])

        # Don't go below 100 or over 10 000 nits
        peak = max(max_rgb, TARGET_NITS)
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
            string = "Max RGB: {:.04f}, target: {} nits".format(max_rgb, nits)
        else:
            string = f"Predetermined target: {nits} nits"

        if adjusted:
            string += "\ngamma_adjust: {:.04f}, luma_scaling: {:.04f}".format(gamma_adjust, luma_scaling)

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

        do_adjust = (adjust_gamma and peak >= 256 and ("moe.kageru.adaptivegrain" in core.get_plugins()))

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
                targets: Optional[List[int]] = None,
                range: Optional[str] = None) -> vs.VideoNode:
        max_rgb, frame_nits = __calculate_max_rgb(n, f, targets, range=range)

        src_peak = frame_nits / 10.0
        src_scale = ST2084_PEAK_LUMINANCE / frame_nits

        dst_peak = TARGET_NITS / 10.0
        dst_avg = math.pow(0.5, 2.4)
        dst_scale = math.pow(REF_WHITE / TARGET_NITS, 2.4)

        clip = core.placebo.Tonemap(clip,
                                    dynamic_peak_detection=False,
                                    smoothing_period=-1,
                                    scene_threshold_low=-1,
                                    scene_threshold_high=-1,
                                    srcp=6,
                                    dstp=3,
                                    srct=11,
                                    dstt=1,
                                    src_peak=src_peak,
                                    src_scale=src_scale,
                                    dst_peak=dst_peak,
                                    dst_avg=dst_avg,
                                    dst_scale=dst_scale,
                                    tone_mapping_algo=placebo_algo,
                                    tone_mapping_param=placebo_param)

        if show:
            show_clip = core.resize.Spline36(clip, format=vs.YUV444P16, matrix_s="709")
            clip = __add_show_info(show_clip, max_rgb, frame_nits, targets)
            clip = core.resize.Spline36(clip, format=vs.RGB48)

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
    use_placebo = libplacebo and ("com.vs.placebo" in core.get_plugins())

    if use_placebo:
        clip = core.resize.Spline36(clip, format=vs.RGB48, chromaloc_in_s=chromaloc_in_s, chromaloc_s=chromaloc_s)

        if placebo_dt:
            # Tonemap
            tonemapped_clip = core.placebo.Tonemap(clip,
                                                   dynamic_peak_detection=True,
                                                   smoothing_period=1,
                                                   scene_threshold_low=-1,
                                                   scene_threshold_high=-1,
                                                   srcp=6,
                                                   dstp=3,
                                                   srct=11,
                                                   dstt=1,
                                                   tone_mapping_algo=placebo_algo,
                                                   tone_mapping_param=placebo_param)
        else:
            prop_src = __get_rgb_prop_src(clip, reference, target_list)

            tonemapped_clip = core.std.FrameEval(clip,
                                                 partial(__pl_dt, clip=clip, targets=target_list, show=show),
                                                 prop_src=prop_src)
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
                                    chromaloc_s=chromaloc_s)

        prop_src = __get_rgb_prop_src(clip, reference, target_list)

        tonemapped_clip = core.std.FrameEval(clip,
                                             partial(__dt,
                                                     clip=clip,
                                                     show=show,
                                                     adjust_gamma=adjust_gamma,
                                                     targets=target_list),
                                             prop_src=prop_src)

    tonemapped_clip = core.resize.Spline36(tonemapped_clip, format=vs.YUV444P16, matrix_s="709")

    if src_fmt:
        return core.resize.Spline36(tonemapped_clip, format=clip_orig_format, dither_type="error_diffusion")
    else:
        return Depth(tonemapped_clip, 8)


def FillBorders(clip: vs.VideoNode,
                left: int = 0,
                right: int = 0,
                top: int = 0,
                bottom: int = 0,
                planes: Union[int, List[int]] = [0, 1, 2],
                mode: str = 'fixborders') -> vs.VideoNode:
    """
    FillBorders wrapper that automatically sets fixborders mode.
    If the chroma is subsampled, ceils the number of chroma rows to fill.
    This means that for 4:2:0 with left=1 1px luma and chroma is filled, 2px luma and 1px chroma for left=2,
    3px luma and 2px chroma for left=3 etc.
    """
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
    else:
        return clip.fb.FillBorders(left=left, right=right, top=top, bottom=bottom, mode=mode)


#####################
# Utility functions #
#####################


def SelectRangeEvery(clip: vs.VideoNode,
                     every: int,
                     length: int,
                     offset: Union[int, List[int]] = [0, 0]) -> vs.VideoNode:
    """
    SelectRangeEvery, port from Avisynth's function. From sgvsfunc.
    Offset can be an array with the first entry being the offset from the start and the second from the end.
    > Usage: SelectRangeEvery(clip, every, length, offset)
      * select <length> frames every <every> frames, starting at frame <offset>
    """
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
    FrameInfo. From sgvsfunc, with additional style option.
    > Usage: FrameInfo(clip, title)
      * Print the frame number, the picture type and a title on each frame
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


def DelFrameProp(clip: vs.VideoNode,
                 primaries: bool = True,
                 matrix: bool = True,
                 transfer: bool = True) -> vs.VideoNode:
    """
    DelFrameProp, delete primaries, matrix or transfer frame properties. From sgvsfunc.
      Avoids "Unrecognized transfer characteristics" or
        "unrecognized color primaries" associated with Vapoursynth Editor
    > Usage: DelFrameProp(clip, primaries, matrix, transfer)
      * primaries, matrix, transfer are boolean, True meaning that the property is deleted (default)
    """
    if primaries:
        clip = core.std.SetFrameProp(clip, prop="_Primaries", delete=True)

    if matrix:
        clip = core.std.SetFrameProp(clip, prop="_Matrix", delete=True)

    if transfer:
        clip = core.std.SetFrameProp(clip, prop="_Transfer", delete=True)

    return clip


def InterleaveDir(folder: str,
                  PrintInfo: bool = False,
                  DelProp: bool = False,
                  first: Optional[vs.VideoNode] = None,
                  repeat: bool = False,
                  tonemap: bool = False,
                  source_filter: Optional[Callable[[Union[str, PathLike]], vs.VideoNode]] = None) -> vs.VideoNode:
    """
    InterleaveDir, load all mkv files located in a directory and interleave them. From sgvsfunc.
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

    for i in range(len(files)):
        filename = files[i].name

        if files[i].is_file() and '.mkv' in files[i].suffixes:

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
                sources[j] = DelFrameProp(sources[j])
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
    ExtractFramesReader, reads a csv file to extract ranges of frames. From sgvsfunc.
    > Usage: ExtractFramesReader(clip, csvfile)
      * csvfile is the path to a csv file containing in each row: <startframe> <endframe>
        the csv file may contain other columns, which will not be read
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
            gamma: Optional[float] = None,
            min_in: Union[int, List[Union[int, float]]] = [16, 16],
            max_in: Union[int, List[Union[int, float]]] = [235, 240],
            min_out: Optional[Union[int, List[Union[int, float]]]] = None,
            max_out: Optional[Union[int, List[Union[int, float]]]] = None,
            planes: Union[int, List[Union[int, float]]] = 0,
            preset: int = None,
            range: int = 0,
            input_depth: int = 8) -> vs.VideoNode:
    """
    A wrapper around std.Levels to fix what's commonly known as the gamma bug.
    :param clip: Processed clip.
    :param gamma: Gamma adjustment value.  Default of 0.88 is usually correct.
    :param min_in: Input minimum.
    :param max_in: Input maximum.
    :param min_out: Output minimum.
    :param max_out: Output maximum.
    :param preset: 1: standard gamma bug, 2: luma-only overflow, 3: overflow
    overflow explained: https://guide.encode.moe/encoding/video-artifacts.html#underflow--overflow
    :param range: Pixel value range.
    :return: Clip with gamma adjusted or levels fixed.
    """
    depth = 32
    clip_ = Depth(clip, depth, range=range, range_in=range)

    if gamma is None and preset is not None:
        gamma = 0.88
    elif gamma is None and preset is None:
        gamma = 1

    if isinstance(min_in, int):
        min_in = [
            scale_value(min_in, input_depth, depth, range, scale_offsets=True),
            scale_value(min_in, input_depth, depth, range, scale_offsets=True, chroma=True)
        ]
    else:
        min_in = [
            scale_value(min_in[0], input_depth, depth, range, scale_offsets=True),
            scale_value(min_in[1], input_depth, depth, range, scale_offsets=True, chroma=True)
        ]

    if isinstance(max_in, int):
        max_in = [
            scale_value(max_in, input_depth, depth, range, scale_offsets=True),
            scale_value(max_in, input_depth, depth, range, scale_offsets=True, chroma=True)
        ]
    else:
        max_in = [
            scale_value(max_in[0], input_depth, depth, range, scale_offsets=True),
            scale_value(max_in[1], input_depth, depth, range, scale_offsets=True, chroma=True)
        ]

    if min_out is None:
        min_out = min_in
    elif isinstance(min_out, int):
        min_out = [
            scale_value(min_out, input_depth, depth, range, scale_offsets=True),
            scale_value(min_out, input_depth, depth, range, scale_offsets=True, chroma=True)
        ]
    else:
        min_out = [
            scale_value(min_out[0], input_depth, depth, range, scale_offsets=True),
            scale_value(min_out[1], input_depth, depth, range, scale_offsets=True, chroma=True)
        ]

    if max_out is None:
        max_out = max_in
    elif isinstance(max_out, int):
        max_out = [
            scale_value(max_out, input_depth, depth, range, scale_offsets=True),
            scale_value(max_out, input_depth, depth, range, scale_offsets=True, chroma=True)
        ]
    else:
        max_out = [
            scale_value(max_out[0], input_depth, depth, range, scale_offsets=True),
            scale_value(max_out[1], input_depth, depth, range, scale_offsets=True, chroma=True)
        ]

    if preset is None:
        if isinstance(planes, int):
            p = 0 if planes == 0 else 1
            adj = core.std.Levels(clip_,
                                  gamma=gamma,
                                  min_in=min_in[p],
                                  max_in=max_in[p],
                                  min_out=min_out[p],
                                  max_out=max_out[p],
                                  planes=planes)
        else:
            adj = clip_
            for _ in planes:
                p = 0 if _ == 0 else 1
                adj = core.std.Levels(adj,
                                      gamma=gamma,
                                      min_in=min_in[p],
                                      max_in=max_in[p],
                                      min_out=min_out[p],
                                      max_out=max_out[p],
                                      planes=_)

    elif preset == 1:
        adj = core.std.Levels(clip_, gamma=gamma, planes=0)

    elif preset == 2:
        y, u, v = split(clip)
        y = y.resize.Point(range_in_s="full", range_s="limited", format=y.format, dither_type="error_diffusion")
        return join([y, u, v])

    elif preset == 3:
        return clip.resize.Point(range_in_s="full",
                                 range_s="limited",
                                 format=clip.format,
                                 dither_type="error_diffusion")

    return Depth(adj, clip.format.bits_per_sample, range=range, range_in=range)


def mt_lut(clip: vs.VideoNode, expr: str, planes: List[int] = [0]) -> vs.VideoNode:
    """
    mt_lut, port from Avisynth's function. From sgvsfunc.
    > Usage: mt_lut(clip, expr, planes)
      * expr is an infix expression, not like avisynth's mt_lut which takes a postfix one
    """
    minimum = 16 * ((1 << clip.format.bits_per_sample) - 1) // 256
    maximum = 235 * ((1 << clip.format.bits_per_sample) - 1) // 256

    def clampexpr(x: float) -> int:
        return int(max(minimum, min(round(eval(expr)), maximum)))

    return core.std.Lut(clip=clip, function=clampexpr, planes=planes)


def autogma(clip: vs.VideoNode, adj: float = 1.3, thr: float = 0.40) -> vs.VideoNode:
    """
    From https://gitlab.com/snippets/1895974.
    Just a simple function to help identify banding.
    First plane's gamma is raised by adj. If the average pixel value is greater than thr, the output will be inverted.
    :param clip: Clip to be processed. GRAY or YUV color family is required.
    :param adj: Gamma value to be adjusted by. Must be greater than or equal to 1.
    :param thr: Threshold above which the output will be inverted. Values span from 0 to 1, as generated by PlaneStats.
    :return: Clip with first plane's gamma adjusted by adj and inverted if average value above thr.
    """
    if clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError("autogma: Only GRAY and YUV color families are supported!")
    if adj < 1:
        raise ValueError("autogma: The value for adj must be greater than or equal to 1.")

    luma = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    s = luma.std.PlaneStats()

    def hilo(n: int, f: vs.VideoFrame, clip: vs.VideoNode, adj: float, thr: int) -> vs.VideoNode:
        g = core.std.Levels(clip, gamma=adj)

        if f.props.PlaneStatsAverage > thr:
            return g.std.Invert().sub.Subtitle("Current average: {}".format(str(f.props.PlaneStatsAverage)))
        else:
            return g.sub.Subtitle("Current average: {}".format(str(f.props.PlaneStatsAverage)))

    prc = core.std.FrameEval(luma, partial(hilo, clip=luma, adj=adj, thr=thr), prop_src=s)

    if clip.format.color_family == vs.YUV:
        return core.std.ShufflePlanes([prc, clip], [0, 1, 2], vs.YUV)
    else:
        return prc


def UpscaleCheck(clip: vs.VideoNode,
                 res: int = 720,
                 title: str = "Upscaled",
                 bits: Optional[int] = None) -> vs.VideoNode:
    """
    Quick port of https://gist.github.com/pcroland/c1f1e46cd3e36021927eb033e5161298
    Dumb detail check (but not as dumb as greyscale) 
    Resizes luma to specified resolution and scales back with Spline36
    Handles conversion automatically so you can input a raw RGB image, 
    Without needing to convert to YUV4XXPX manually like the AviSynth counterpart
    TODO: -

    :param res: Target resolution (720, 576, 480, ...)
    :param title: Custom text output ("540p upscale")
    :param bits: Bit depth of output
    :return: Clip resampled to target resolution and back
    """
    src = clip

    # Generic error handling, YCOCG & COMPAT input not tested as such blocked by default
    if src.format.color_family not in [vs.YUV, vs.GRAY, vs.RGB]:
        raise TypeError("UpscaleCheck: Only supports YUV, GRAY or RGB input!")
    elif src.format.color_family in [vs.GRAY, vs.RGB]:
        clip = core.resize.Spline36(src, format=vs.YUV444P16, matrix_s='709')

    if src.format.color_family is vs.RGB and src.format.bits_per_sample == 8:
        bits = 8
    elif bits is None:
        bits = src.format.bits_per_sample

    b16 = Depth(clip, 16)
    lma = core.std.ShufflePlanes(b16, 0, vs.GRAY)

    dwn = CropResize(lma, preset=res, resizer='Spline36')
    ups = core.resize.Spline36(dwn, clip.width, clip.height)

    mrg = core.std.ShufflePlanes([ups, b16], [0, 1, 2], vs.YUV)
    txt = FrameInfo(mrg, f"{title}")
    cmp = core.std.Interleave([b16, txt])

    return Depth(cmp, bits)


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

    has_descale = "tegaf.asi.xe" in core.get_plugins()

    if not has_descale:
        raise ModuleNotFoundError("RescaleCheck: Requires 'descale' plugin to be installed")

    src = clip
    # Generic error handling, YCOCG & COMPAT input not tested as such blocked by default
    if src.format.color_family not in [vs.YUV, vs.GRAY, vs.RGB]:
        raise TypeError("UpscaleCheck: Only supports YUV, GRAY or RGB input!")
    elif src.format.color_family in [vs.GRAY, vs.RGB]:
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


def Import(file: str) -> vs.VideoNode:
    """
    Allows for easy import of your .vpy
    This will load a script even without set_output being present, however;
    if you would like to manipulate the clip you will need to specifiy an output:
    --
    script = Import(r'1080p.vpy')
    script = FrameInfo(script, "Filtered")
    script.set_output()
    --
    TODO: Find a way of keeping this more in-line with expected VS behavior (no output unless specified)
    """
    from importlib.machinery import SourceFileLoader

    if file.endswith('.vpy'):
        output = SourceFileLoader('script', file).load_module().vs.get_output()

        try:
            if isinstance(output, vs.VideoOutputTuple):
                return output.clip
            else:
                return output
        except AttributeError:
            return output
    else:
        raise TypeError("Import: Only .vpy is supported!")


def greyscale(clip: vs.VideoNode) -> vs.VideoNode:
    """
    From https://gitlab.com/snippets/1895242.
    Really stupid function. Only advisable if you're not doing any other filtering. Replaces chroma planes with gray.
    """
    if clip.format.color_family != vs.YUV:
        raise TypeError("GreyScale: YUV input is required!")

    grey = core.std.BlankClip(clip)

    return core.std.ShufflePlanes([clip, grey], [0, 1, 2], vs.YUV)


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
                 planes: List[int] = [0, 1, 2],
                 sat: Optional[List[float]] = None) -> vs.VideoNode:
    """
    A wrapper to resize clips with borders.  This is meant for scenefiltering differing crops.
    :param ref: Reference resized clip, used purely to determine dimensions.
    :param left, right, top, bottom: Size of borders.  Uneven numbers will lead to filling.
    :param bb: bbmod args: [left, right, top, bottom, thresh, blur, planes]
    :param planes: Planes to be filled.
    :param sat: Saturation adjustment in AddBordersMod.  If None, std.AddBorders is used.
    """
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
    else:
        return clip.std.AddBorders(left=borders[0], right=borders[1], top=borders[2], bottom=borders[3])


def RandomFrameNumbers(clip: vs.VideoNode,
                       num: int = 6,
                       start_offset: int = 1000,
                       end_offset: int = 10000,
                       output_file: Union[str, PathLike] = "screens.txt",
                       ftypes_first: Union[str, List[str]] = ["P", "B"],
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

    def divisible_random(a: int, b: int, n: int) -> int:
        if not a % n:
            return random.choice(range(a, b, n))
        else:
            return random.choice(range(a + n - (a % n), b, n))

    # filter frame types function for frameeval
    def filter_ftype(n: int, f: vs.VideoFrame, clip: vs.VideoNode, frame_num: int, block: Dict) -> vs.VideoNode:
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


def st2084_eotf(x: float) -> float:
    y = float(0.0)
    if (x > 0.0):
        xpow = math.pow(x, float(1.0) / ST2084_M2)
        num = max(xpow - ST2084_C1, float(0.0))
        den = max(ST2084_C2 - ST2084_C3 * xpow, float('-inf'))
        y = float(math.pow(num / den, float(1.0) / ST2084_M1))

    return y


def st2084_inverse_eotf(x: float) -> float:
    y = x / ST2084_PEAK_LUMINANCE

    return math.pow((ST2084_C1 + (ST2084_C2 * math.pow(y, ST2084_M1))) / (1 + (ST2084_C3 * math.pow(y, ST2084_M1))),
                    ST2084_M2)


#####################
#      Aliases      #
#####################

GetPlane = plane
rfs = ReplaceFrames
fb = FillBorders

zr = zresize

cr = CropResize
CR = CropResize
cropresize = CropResize

br = BorderResize
borderresize = BorderResize

gs = greyscale
grayscale = greyscale
GreyScale = greyscale
GrayScale = greyscale

########
# Dict #
########
