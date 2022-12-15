class OutputModel(object):

    def __init__(self, result, pred_confidence=None, pred_component=None):
        """
        模型输出结果

        :param result: 模型输出结果（历史值+预测值）
        :param pred_confidence: prophet的置信区间
        :param pred_component: prophet的分量
        """
        self.result = result
        self.pred_confidence = pred_confidence
        self.pred_component = pred_component

    def get_output(self):
        """
        获得实例的属性

        :return:
        """
        return self.__dict__
