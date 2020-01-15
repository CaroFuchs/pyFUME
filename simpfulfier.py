
class SimpfulConverter(object):
    """    This object converts a description of a Fuzzy System into a readable
        Simpful project file.
    """
    
    def __init__(self, 
        input_variables_names,
        consequents_matrix,
        fuzzy_sets
        ):
        super(SimpfulConverter, self).__init__()
        self._input_variables = input_variables_names
        self._consequents_matrix = consequents_matrix
        self._clusters = len(self._consequents_matrix)
        self._fuzzy_sets = fuzzy_sets

        assert(len(self._input_variables)+1 == len(self._consequents_matrix[0]))

        print(" * Detected %d rules / clusters" % self._clusters)

        self._source_code = []
        self._source_code.append( '# WARNING: this source code was automatically generated by Simpfulifier.' )
        self._source_code.append( "from simpful import *" )
        self._source_code.append("\nFR = FuzzyReasoner()")

        
    def save_code(self, path):
        code = self.generate_code()
        with open(path, "w") as fo:
            fo.write(code)
        print (" * Code saved to file %s" % path)

    def generate_object(self):
        code = self.generate_code()
        exec(code, globals()) 
        from copy import deepcopy
        self._fuzzyreasoner = deepcopy(FR)

    def generate_code(self, use_main=False):

        # rules
        rule_texts = self.create_rules()
        for i in range(1, self._clusters+1):
            self._source_code.append('RULE%d = "%s"' % (i, rule_texts[i-1]))
        self._source_code.append("FR.add_rules([%s])" % (", ". join(["RULE%d" % i for i in range(1,self._clusters+1)])))

        self._source_code.append("")

        # output functions
        B = self._create_consequents()
        for i in range(self._clusters):
            self._source_code.append("FR.set_output_function('%s', '%s')" % ("fun%d" % (i+1),  B[i]))

        self._source_code.append("")
    
        # fuzzy sets and membership functions
        result = self._create_fuzzy_sets()
        self._source_code.append(result)

        self._source_code.append("# end of automatically generated code #")

        return "\n".join(self._source_code)


    def _create_fuzzy_sets(self):
        j=0
        chunk = ""
        for var in self._input_variables:
            subchunk = []
            for cluster in range(self._clusters):
                print (" * Creating fuzzy set for variable %s, cluster%d" % (var, cluster+1))
                chunk += 'FS_%d = FuzzySet(' % (j+1)

                term = 'cluster%d' % (cluster+1)

                fstype, params = self._fuzzy_sets[j]
                if fstype == 'gauss':
                    chunk += "function=Gaussian_MF(%f, %f), term='%s')" % (params[0], params[1], term) 
 
                elif fstype == 'gauss2':
                    chunk += "function=DoubleGaussian_MF(%f, %f, %f, %f), term='%s')" % (params[0], params[1], params[2], params[3], term) 

                elif fstype == 'sigmoid':
                    chunk += "function=Sigmoid_MF(%f, %f), term='%s')" % (params[0], params[1], term) 

                elif fstype == 'invgauss':
                    chunk += "function=InvGaussian_MF(%f, %f), term='%s')" % (params[0], params[1], term) 
                else:
                    raise Exception("Fuzzy set type not supported,"+fstype)
                subchunk.append("FS_%d" % (j+1))
                #print ( self._fuzzy_sets[j] )
                j += 1
                chunk += "\n"

            chunk += "MF_%s = MembershipFunction([%s], concept='%s')\n" % (var, ", ".join(subchunk), var )
            chunk += "FR.add_membership_function('%s', MF_%s)\n\n" % (var, var)

        return chunk

    def _create_consequents(self):
        result = []
        for row in self._consequents_matrix:
            result.append(("+".join(["%.2e*%s" % (value, name) for (name, value) in zip(self._input_variables, row[:-1])])))
            result[-1] += "+%.2e" % row[-1]
        return result

    def _create_antecedents(self):
        result = []
        for i in range(self._clusters):
            result.append( (" AND ".join(["(%s IS cluster%d)" % (name, i+1) for name in self._input_variables])) )
        return result
        
    def create_rules(self):
        A = self._create_antecedents()
        # B = self._create_consequents()
        B = ["fun%d" % (i+1) for i in range(self._clusters)]
        result = ["IF %s THEN (OUTPUT IS %s)" % (a,b) for a,b in zip(A,B)]
        return result


if __name__ == '__main__':
    
    SC = SimpfulConverter(
        input_variables_names = ["pippo", "pluto"],
        consequents_matrix = [[1,2,3], 
                              [2,3,5]],
        fuzzy_sets = [
                ["gauss", [0,1]],
                ["sigmoid", [1,2]],
                ["gauss2", [0,1,2,3]],
                ["invgauss", [0,1]]
                ]
    )
    
    SC.save_code("TEST.py")
    SC.generate_object()
    print(FR._mfs['pippo'])
    