class gaussMix:
    def __init__(self,**kwargs):
        if kwargs is None:
            raise Exception('Some arguments must be supplied. dim = [dimension] \
                            or pack = [dictionary with a previously created package]')
        else:
            if 'dim' in kwargs:
                self.dim = kwargs['dim']
                self.idNum = 0
                self.gaussList = list()
                self.cardinality = 0

            elif 'pack' in kwargs:
                self.unpack(kwargs['pack'])
            else:
                raise Exception('dim = [dimension] or pack = [dictionary with a previously created package]')
        
    def unpack(self,pack):
        self.dim = pack['dim']
        self.gaussList = pack['all']
        self.idNum = len(self.gaussList)
        self.cardinality = pack['cardinality']
        
    def add(self,mean,variance,numPoints):
        if len(mean)!= self.dim:
            raise Exception("wrong dimension")
        
        self.cardinality +=  numPoints
        
        g = {'id':self.idNum,'mean':mean,'var':variance,
             'card':numPoints,'dist':np.random.normal(mean,variance,(numPoints,self.dim))}
        """
        if self.mixture =  =  None:
            self.mixture = g['dist']
        else:
            self.mixture = np.concatenate((self.mixture,g['dist']))
        """
        self.gaussList.append(g)
        self.updateWeights()
        
        self.idNum += 1
        
    def getMixture(self):
        returnMix = []
        
        for i,g in enumerate(self.gaussList):
            returnMix.append(g['dist'])
            
        returnMix = np.concatenate(tuple(returnMix))
        return returnMix
        
    def updateWeights(self):
        for g in self.gaussList:
            g['weight'] = float(g['card'])/self.cardinality

    def sample(self,numPoints):
        returnMix = []

        if numPoints > self.cardinality:
            return self.getMixture()
       
        for i,g in enumerate(self.gaussList):
            num = int(numPoints*g['weight'])
            returnMix.append(g['dist'][0:num,:])
                
        returnMix = np.concatenate(tuple(returnMix))
        return returnMix
    
    def getWeights(self):
        w = [0]*len(self.gaussList)
        for i,g in enumerate(self.gaussList):
            w[i] = g['weight']
                   
        return w
    
    def package(self):
        pack = dict()

        pack['all'] = self.gaussList
        pack['data'] = self.getMixture()
        pack['dim'] = self.dim
        pack['cardinality'] = self.cardinality
        
        return pack
    
    def gaussAsList(self):
        m = list()
        for i,g in enumerate(self.gaussList):
            m.append(g['dist'])
        return m
    
    """
    # better function without redundacy
    def package(self):
        pack = dict()
        
        pack['all'] = [None]*len(self.gaussList)
            pack['all']
        for i,g in enumerate(self.gaussList):
            
        pack['all'] = self.gaussList
        pack['data'] = self.getMixture()
        pack['dim'] = self.dim
        pack['cardinality'] = self.cardinality
        
        return pack
        
        """