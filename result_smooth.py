# 姿态分类结果平滑
class EMADictSmoothing(object):
    """平滑姿势分类"""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """平滑给定的姿势分类。
        平滑是通过计算在给定时间窗口中观察到的每个姿势类别的指数移动平均值来完成的。错过的姿势类将替换为 0。
        参数:
          数据：姿态分类的字典. Sample:
              {
                'pushups_down': 8,
                'pushups_up': 2,
              }
        结果:相同格式的字典，但有平滑的和浮动的而不是整数的值. Sample:
            {
              'pushups_down': 8.3,
              'pushups_up': 1.7,
            }
        """
        # 将新数据添加到窗口的开头以获得更简单的代码.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # 得到平滑值
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data

