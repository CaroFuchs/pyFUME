from ModelEstimation import RuleCreator
from simpfulfier import SimpfulConverter

class SugenoFISBuilder(object):
	"""docstring for SugenoFISBuilder"""
	def __init__(self, path=None, clusters=0, variable_names=None):
		super(SugenoFISBuilder, self).__init__()

		self._RC = RuleCreator(datapath=path, nrclus=clusters, varnames=variable_names)
		self._SC = SimpfulConverter(
			input_variables_names = variable_names,
			consequents_matrix = self._RC.Cons,
			fuzzy_sets = self._RC.MFs
			)
		self._SC.save_code("TEST.py")



if __name__ == '__main__':
	
	SFB = SugenoFISBuilder(path="dataset6_withlabels", clusters=3, variable_names=["Simone", "Caro"])