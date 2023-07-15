import torch
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment as hungarian


class Confusion(object):
	"""
	column of confusion matrix: predicted index
	row of confusion matrix: target index
	"""
	def __init__(self, k, normalized = False):
		super(Confusion, self).__init__()
		self.k = k
		self.conf = torch.LongTensor(k,k)
		self.normalized = normalized
		self.reset()

	def reset(self):
		self.conf.fill_(0)
		self.gt_n_cluster = None

	def cuda(self):
		self.conf = self.conf.cuda()

	def add(self, output, target):
		output = output.squeeze()
		target = target.squeeze()
		assert output.size(0) == target.size(0), \
				'number of targets and outputs do not match'
		if output.ndimension()>1: #it is the raw probabilities over classes
			assert output.size(1) == self.conf.size(0), \
				'number of outputs does not match size of confusion matrix'
		
			_, pred = output.max(1) #find the predicted class
		else: #it is already the predicted class
			pred = output
		indices = (target * self.conf.stride(0) + pred.squeeze_().type_as(target)).type_as(self.conf)
		ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
		self._conf_flat = self.conf.view(-1)
		self._conf_flat.index_add_(0, indices, ones)
		
	def acc(self):
		TP = self.conf.diag().sum().item()
		total = self.conf.sum().item()
		if total == 0:
			return 0
		return float(TP) / total
		
	def optimal_assignment(self, gt_n_cluster=None, assign=None):
		if assign is None:
			mat = -self.conf.cpu().numpy() # hungaian finds the minimum cost
			r, assign = hungarian(mat)
			self.assign = assign
		self.conf = self.conf[:, assign]
		self.gt_n_cluster = gt_n_cluster
		return assign
		
	def conf2label(self):
		conf = self.conf
		gt_classes_count = conf.sum(1).squeeze()
		n_sample = gt_classes_count.sum().item()
		gt_label = torch.zeros(n_sample)
		pred_label = torch.zeros(n_sample)
		cur_idx = 0
		for c in range(conf.size(0)):
			if gt_classes_count[c] > 0:
				gt_label[cur_idx:cur_idx + gt_classes_count[c]].fill_(c)
			for p in range(conf.size(1)):
				if conf[c][p] > 0:
					pred_label[cur_idx : cur_idx + conf[c][p]].fill_(p)
				cur_idx = cur_idx + conf[c][p]
		return gt_label, pred_label
	
	def clusterscores(self):
		target,pred = self.conf2label()
		NMI = normalized_mutual_info_score(target, pred)
		ARI = adjusted_rand_score(target, pred)
		AMI = adjusted_mutual_info_score(target, pred)
		return {'NMI': NMI,'ARI': ARI,'AMI': AMI}