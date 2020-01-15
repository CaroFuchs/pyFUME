from ModelEstimation import ModelCreator
from simpfulfier import SimpfulConverter

class SugenoFISBuilder(object):
    """docstring for SugenoFISBuilder"""
    def __init__(self, path=None, clusters=0, variable_names=None):
        super(SugenoFISBuilder, self).__init__()

        self._RC = ModelCreator(datapath=path, nrclus=clusters, varnames=variable_names)
        self._SC = SimpfulConverter(
            input_variables_names = variable_names,
            consequents_matrix = self._RC.cons,
            fuzzy_sets = self._RC.mfs
            )
        
        self._SC.generate_object()
        self._SC.save_code("TEST.py")

if __name__ == '__main__':
    
    SFB = SugenoFISBuilder(path='C:/Users/20115284/Desktop/FitMF/data/dataset6_withlabels', clusters=2, variable_names=["Simone", "Caro"])
    print(globals())