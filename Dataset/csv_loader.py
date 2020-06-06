import csv
import numpy as np

class CSVLoader():
    r"""
    读取CSV格式的数据集

    """

    def __init__(self, path, headers=None, sep=",", dropna=False):
        r"""

        :param List[str] headers: CSV文件的文件头.定义每一列的属性名称,即返回的DataSet中`field`的名称
            若为 ``None`` ,则将读入文件的第一行视作 ``headers`` . Default: ``None``
        :param str sep: CSV文件中列与列之间的分隔符. Default: ","
        :param bool dropna: 是否忽略非法数据,若 ``True`` 则忽略
        """
        super().__init__()
        self.path = path
        self.headers = headers
        self.sep = sep
        self.dropna = dropna
        self.sentences = []
        self.labels = []
        self.input_cols = []
        self.target_cols = []

        # self.read_csv(self.path)

    def data_init(self):
        self.read_csv(self.path)

    def read_csv(self, path, encoding='utf-8-sig', dropna=True):

        with open(path, 'r', encoding=encoding) as csv_file:
            f = csv.reader(csv_file, delimiter=self.sep)
            start_idx = 0

            if self.headers is None:
                self.headers = next(f)
                # print(headers)
                start_idx += 1

            # ID,txt,Label
            labels = []
            for line_idx, line in enumerate(f, start_idx):
                contents = line

                _dict = {}
                for header, content in zip(self.headers, contents):
                    if str(header).lower() in self.target_cols:
                        labels.append(content)
                    else:
                        _dict[header] = str(content).lower()  # 小写
                self.sentences.append(_dict)

            #处理成数字
            self.labels = np.array([1 if label == '1' else 0 for label in labels])

    def set_input_cols(self, col_name):
        if isinstance(col_name, str):
            self.input_cols.append(col_name.lower())
        elif isinstance(col_name, list):
            for item in col_name:
                self.input_cols.append(item.lower())

    def set_target_cols(self, col_name):
        self.target_cols.append(str(col_name).lower())