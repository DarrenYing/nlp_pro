r"""undocumented"""

__all__ = [
    "Loader"
]

from typing import Union, Dict


from ...core.dataset import DataSet


class Loader:
    r"""
    各种数据 Loader 的基类，提供了 API 的参考.
    Loader支持以下的三个函数

    - download() 函数：自动将该数据集下载到缓存地址，默认缓存地址为~/.fastNLP/datasets/。由于版权等原因，不是所有的Loader都实现了该方法。该方法会返回下载后文件所处的缓存地址。
    - _load() 函数：从一个数据文件中读取数据，返回一个 :class:`~fastNLP.DataSet` 。返回的DataSet的内容可以通过每个Loader的文档判断出。
    - load() 函数：将文件分别读取为DataSet，然后将多个DataSet放入到一个DataBundle中并返回

    """

    def __init__(self):
        pass

    def _load(self, path: str) -> DataSet:
        r"""
        给定一个路径，返回读取的DataSet。

        :param str path: 路径
        :return: DataSet
        """
        raise NotImplementedError

