from collections import defaultdict
import numpy as np
import paddle
from paddlenlp.utils.log import logger
from seqeval.metrics.sequence_labeling import get_entities




def get_entity_and_relation(label):
    parts = label.split('_')
    if label == 'O':  # 直接处理'O'标签的情况
        return 'O', ''
    elif len(parts) == 2:
        return label, ''
    elif len(parts) > 2:
        return '_'.join(parts[:-2]), parts[-2]
    else:
        return None, None  # 对于不合法的标签格式

# 假设 get_entities 已经正确实现，这里直接使用
def extract_tp_actual_correct(y_true, y_pred, suffix=False):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)

    for sent_true, sent_pred in zip(y_true, y_pred):
        # 假设 get_entities 函数返回标签对应的实体类型及其在句子中的位置
        for label, start, end in get_entities(sent_true, suffix):
            entity_type, relation_type = get_entity_and_relation(label)
            entities_true[(entity_type, relation_type)].add((start, end))

        for label, start, end in get_entities(sent_pred, suffix):
            entity_type, relation_type = get_entity_and_relation(label)
            entities_pred[(entity_type, relation_type)].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
    tp_sum, pred_sum, true_sum = np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum



# def extract_tp_actual_correct(y_true, y_pred, suffix=False):
#     entities_true = defaultdict(set)
#     entities_pred = defaultdict(set)
#
#     # 定义函数提取实体和关系
#     def get_entity_and_relation(label):
#         # 判断标签是否包含关系类型
#         if '_' in label:
#             parts = label.split('_')
#             if len(parts) >= 2:
#                 # 处理包含关系类型的标签
#                 entity_type = parts[0]
#                 relation_type = parts[1]
#             else:
#                 # 只有实体类型，没有关系类型
#                 entity_type = parts[0]
#                 relation_type = ''
#         else:
#             # 标签只有实体类型
#             entity_type = label
#             relation_type = ''
#
#         return entity_type, relation_type
#
#     # 使用 get_entities 提取实体信息及其位置
#     for sent_true, sent_pred in zip(y_true, y_pred):
#         for label, start, end in get_entities(sent_true, suffix):
#             entity_type, relation_type = get_entity_and_relation(label)
#             entities_true[(entity_type, relation_type)].add((start, end))
#
#         for label,start, end in get_entities(sent_pred, suffix):
#             entity_type, relation_type = get_entity_and_relation(label)
#             entities_pred[(entity_type, relation_type)].add((start, end))
#
#     target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
#
#     tp_sum = np.array([], dtype=np.int32)
#     pred_sum = np.array([], dtype=np.int32)
#     true_sum = np.array([], dtype=np.int32)
#
#     # for type_name in target_names:
#     #     entities_true_type = entities_true.get(type_name, set())
#     #     entities_pred_type = entities_pred.get(type_name, set())
#     #     tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
#     #     pred_sum = np.append(pred_sum, len(entities_pred_type))
#     #     true_sum = np.append(true_sum, len(entities_true_type))
#     #
#     # return pred_sum, tp_sum, true_sum
#
#     for type_name in target_names:
#         entities_true_type = entities_true.get(type_name, set())
#         entities_pred_type = entities_pred.get(type_name, set())
#
#         # 如果relation_type为空，只比较entity_type
#         if type_name[1] == '':
#             tp_count = sum(1 for et in entities_pred_type if
#                            et == type_name[0] and et in [et_true for et_true, _ in entities_true_type])
#             tp_sum = np.append(tp_sum, tp_count)
#         else:
#             # 否则，比较entity_type和relation_type
#             tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
#
#         pred_sum = np.append(pred_sum, len(entities_pred_type))
#         true_sum = np.append(true_sum, len(entities_true_type))
#
#     return pred_sum, tp_sum, true_sum

# def extract_tp_actual_correct(y_true, y_pred, suffix):
#     entities_true = defaultdict(set)
#     entities_pred = defaultdict(set)
#
#     # 定义函数提取实体和关系
#     def get_entity_and_relation(label):
#         parts = label.split('_')
#         if len(parts) >= 3:
#             # 处理包含关系类型的标签
#             entity_type = '_'.join(parts[:2])
#             relation_type = parts[2]
#         else:
#             # 处理只包含实体类型的标签
#             entity_type = label
#             relation_type = ''  # 使用空字符串而不是 None
#         return entity_type, relation_type
#
#     # 提取实体信息
#     for sent_true, sent_pred in zip(y_true, y_pred):
#         for label in sent_true:
#             entity_type, relation_type = get_entity_and_relation(label)
#             entities_true[(entity_type, relation_type)].add(label)
#
#         for label in sent_pred:
#             entity_type, relation_type = get_entity_and_relation(label)
#             entities_pred[(entity_type, relation_type)].add(label)
#
#     target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
#
#     tp_sum = np.array([], dtype=np.int32)
#     pred_sum = np.array([], dtype=np.int32)
#     true_sum = np.array([], dtype=np.int32)
#
#     for type_name in target_names:
#         entities_true_type = entities_true.get(type_name, set())
#         entities_pred_type = entities_pred.get(type_name, set())
#         tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
#         pred_sum = np.append(pred_sum, len(entities_pred_type))
#         true_sum = np.append(true_sum, len(entities_true_type))
#
#     return pred_sum, tp_sum, true_sum

class ChunkEvaluator(paddle.metric.Metric):
    """ChunkEvaluator computes the precision, recall and F1-score for chunk detection.
    It is often used in sequence tagging tasks, such as Named Entity Recognition(NER).

    Args:
        label_list (list): The label list.
        suffix (bool): if set True, the label ends with '-B', '-I', '-E' or '-S', else the label starts with them.
    """

    def __init__(self, label_list, suffix=False):
        super(ChunkEvaluator, self).__init__()
        self.id2label_dict = dict(enumerate(label_list))
        self.suffix = suffix
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def compute(self, lengths, predictions, labels, dummy=None):
        """Computes the precision, recall and F1-score for chunk detection.

        Args:
            lengths (tensor): The valid length of every sequence, a tensor with shape `[batch_size]`
            predictions (tensor): The predictions index, a tensor with shape `[batch_size, sequence_length]`.
            labels (tensor): The labels index, a tensor with shape `[batch_size, sequence_length]`.
            dummy (tensor, optional): Unnecessary parameter for compatibility with older versions with parameters list `inputs`, `lengths`, `predictions`, `labels`. Defaults to None.

        Returns:
            num_infer_chunks (tensor): the number of the inference chunks.
            num_label_chunks (tensor): the number of the label chunks.
            num_correct_chunks (tensor): the number of the correct chunks.
        """
        if dummy is not None:
            # TODO(qiujinxuan): rm compatibility support after lic.
            dummy, lengths, predictions, labels = lengths, predictions, labels, dummy
            if not getattr(self, "has_warn", False):
                logger.warning(
                    'Compatibility Warning: The params of ChunkEvaluator.compute has been modified. The old version is `inputs`, `lengths`, `predictions`, `labels` while the current version is `lengths`, `predictions`, `labels`.  Please update the usage.'
                )
                self.has_warn = True
        labels = labels.numpy()
        predictions = predictions.numpy()
        unpad_labels = [[
            self.id2label_dict[index]
            for index in labels[sent_index][:lengths[sent_index]]
        ] for sent_index in range(len(lengths))]
        unpad_predictions = [[
            self.id2label_dict.get(index, "O")
            for index in predictions[sent_index][:lengths[sent_index]]
        ] for sent_index in range(len(lengths))]

        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(
           unpad_labels, unpad_predictions, self.suffix)
        # pred_sum, tp_sum, true_sum = extract_tp_actual_correct(
        #     unpad_labels, unpad_predictions, self.id2label_dict, self.suffix)
        num_correct_chunks = paddle.to_tensor([tp_sum.sum()])
        num_infer_chunks = paddle.to_tensor([pred_sum.sum()])
        num_label_chunks = paddle.to_tensor([true_sum.sum()])

        return num_infer_chunks, num_label_chunks, num_correct_chunks

    def _is_number_or_matrix(self, var):
        def _is_number_(var):
            return isinstance(
                var, int) or isinstance(var, np.int64) or isinstance(
                    var, float) or (isinstance(var, np.ndarray) and
                                    var.shape == (1, ))

        return _is_number_(var) or isinstance(var, np.ndarray)

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        """
        This function takes (num_infer_chunks, num_label_chunks, num_correct_chunks) as input,
        to accumulate and update the corresponding status of the ChunkEvaluator object. The update method is as follows:

        .. math::
                   \\\\ \\begin{array}{l}{\\text { self. num_infer_chunks }+=\\text { num_infer_chunks }} \\\\ {\\text { self. num_Label_chunks }+=\\text { num_label_chunks }} \\\\ {\\text { self. num_correct_chunks }+=\\text { num_correct_chunks }}\\end{array} \\\\

        Args:
            num_infer_chunks(int|numpy.array): The number of chunks in Inference on the given minibatch.
            num_label_chunks(int|numpy.array): The number of chunks in Label on the given mini-batch.
            num_correct_chunks(int|float|numpy.array): The number of chunks both in Inference and Label on the
                                                  given mini-batch.
        """
        if not self._is_number_or_matrix(num_infer_chunks):
            raise ValueError(
                "The 'num_infer_chunks' must be a number(int) or a numpy ndarray."
            )
        if not self._is_number_or_matrix(num_label_chunks):
            raise ValueError(
                "The 'num_label_chunks' must be a number(int, float) or a numpy ndarray."
            )
        if not self._is_number_or_matrix(num_correct_chunks):
            raise ValueError(
                "The 'num_correct_chunks' must be a number(int, float) or a numpy ndarray."
            )
        self.num_infer_chunks += num_infer_chunks
        self.num_label_chunks += num_label_chunks
        self.num_correct_chunks += num_correct_chunks

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            float: mean precision, recall and f1 score.
        """
        precision = float(
            self.num_correct_chunks /
            self.num_infer_chunks) if self.num_infer_chunks else 0.
        recall = float(self.num_correct_chunks /
                       self.num_label_chunks) if self.num_label_chunks else 0.
        f1_score = float(2 * precision * recall / (
            precision + recall)) if self.num_correct_chunks else 0.
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"