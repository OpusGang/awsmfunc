import vapoursynth as vs
from vapoursynth import core

from .awsmfunc import Depth

from typing import List, Union, Optional
from rekt import rektlvls, rekt_fast


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

    def adj(x):
        return core.std.Levels(x,
                               min_in=input_low << 8,
                               max_in=input_high << 8,
                               min_out=output_low << 8,
                               max_out=output_high << 8,
                               planes=0)

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

    def adj(x):
        return core.std.Levels(x,
                               min_in=input_low << 8,
                               max_in=input_high << 8,
                               min_out=output_low << 8,
                               max_out=output_high << 8,
                               planes=0)

    prc = rekt_fast(lma, adj, top=row, bottom=clip.height - row - 1)

    if clip.format.color_family is vs.YUV:
        prc = core.std.ShufflePlanes([prc, hbd], [0, 1, 2], vs.YUV)

    return Depth(prc, clip.format.bits_per_sample)
