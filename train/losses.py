from torch.nn import CrossEntropyLoss
from torch import where, randint, square, stack, tensor, where
from torch.nn import Module


def pair_distance(index, label, class_map):
	pairs = class_map[label]
	if len(pairs) == 1:
		return tensor(index).cuda()
	return pairs[where(pairs != index)[0][randint(0, len(pairs) - 1, (1,))[0]]]


class MyContrastiveLoss(Module):
	def __init__(self, temp=0.01):
		super().__init__()
		self.cross_entropy_loss = CrossEntropyLoss()
		self.temp = temp

	def forward(self, pre_projection_activations, post_projection_activations, labels):
		class_map = [None] * 50
		for i, label in enumerate(labels):
			if class_map[label] is None:
				class_map[label] = [pre_projection_activations[i]]

		pair_indexes = stack([pair_distance(index, label, class_map) for index, label in enumerate(labels)])
		sum_square = square(pre_projection_activations - pre_projection_activations[pair_indexes]).sum(dim=1)
		distance = where(sum_square == 0, 0.00001, sum_square).sqrt().sum()
		return self.cross_entropy_loss(post_projection_activations, labels) + (self.temp * distance)
