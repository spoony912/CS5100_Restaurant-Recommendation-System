import operator

from FAIproject.algorithm.classComparator import ClassComparator


class Knn(object):
    def __init__(self, inputData=None):
        self.inputData = inputData

    #find k nearest neighbours
    def getNearestNeighbours(self, k):

        nDict = {}
        neighbours = []
        for dataRow in self.inputData.testData:
            nDict[dataRow] = self.getSimilarityFactor(dataRow)
        sortedList = sorted(nDict.items(), key=operator.itemgetter(1), reverse=True)
        for sortedValue in sortedList[:k]:
            sortedValue[0].predictionScore = sortedValue[1]
            neighbours.append(sortedValue[0])
        return neighbours

    #Calculate the similarity factor of the most given dataRow and the user.
    def getSimilarityFactor(self, dataRow):

        cc = ClassComparator()
        cc.user = self.inputData.userData
        cc.business = dataRow
        factor = [cc.wifi(), cc.alcohol(), cc.noise_level(), cc.music(), cc.attire(), cc.ambience(),
        cc.price_range(), cc.good_for(), cc.parking(), cc.categories(), cc.dietary_restrictions(), cc.misc_attributes()]
        total = 0
        count = 0
        for f in factor:
            count += 1
            total += f
        return (total/count) * 10

