"""异常类型与退出码，匹配入口与批次清洗预览工具共用。"""


class MatchingError(Exception):
    """所有业务异常的基类；入口捕获后以 exit_code 退出。"""

    exit_code = 1


class ConfigError(MatchingError):
    """规则表/映射表/编码表错误、未知规则编号、批次别名/选科映射/备注分隔符配置错误。"""

    exit_code = 2


class InputError(MatchingError):
    """文件不存在、工作表不存在、无法读取；批次.csv 读取失败或缺列。"""

    exit_code = 3


class FatalDataError(MatchingError):
    """缺“数据颗粒度”列、参与匹配 B 为 0、bi 找不到；预览工具：B 表无数据行。"""

    exit_code = 4


class OutputWriteError(MatchingError):
    """写出失败；预览工具：输出路径越界、目录无法创建、写失败。"""

    exit_code = 5
