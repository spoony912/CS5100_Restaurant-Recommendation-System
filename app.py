import importlib
import sys
from algorithm.dataSet import DataSet
from algorithm.knn import Knn
from util.errorCheck import getRating, MAE



from settings import SYS_ENCODING_UTF, JSON_FILE_PATH, JSON_FILE_NAME, PLOT_RESULTS, DISTANCE_TO_FILTER, TIME_TO_FILTER, \
    KNN_NEIGHBOURS, ENABLE_DISTANCE_FILTER, ENABLE_TIME_FILTER

importlib.reload(sys)
sys.getdefaultencoding()
dataSet = DataSet(JSON_FILE_PATH + JSON_FILE_NAME)
dataSet.loadRawData()
dataSet.processBusinessModels()
print("\nNumber of Business Models: %s" % len(dataSet.businessModels))
dataSet.sliceData()

dataSet.trainUserModel()


if ENABLE_TIME_FILTER:
    dataSet.timeFilterBusinessModel(TIME_TO_FILTER)

if ENABLE_DISTANCE_FILTER:
    dataSet.distFilterBusinessModel(DISTANCE_TO_FILTER)
print("Test Data: %s" % len(dataSet.testData))
print("Training Data: %s \n" % len(dataSet.trainingData))
knn = Knn()
knn.inputData = dataSet
predictions = knn.getNearestNeighbours(KNN_NEIGHBOURS)

for index, p in enumerate(predictions):
    print ("Name: %s\n" \
           "User Rating: %s\n" \
           "Business Rating: %s\n" \
           "Prediction Score: %s\n" \
            
           "Prediction Rank: %s\n"
           % (p.name,
              p.stars,
              p.findHighestUserRating(dataSet.businessModels),
              p.predictionScore,
              #getRating(round(p.predictionScore)),
              index + 1))

print("Mean Absolute Error : %s: : %s" % ((ENABLE_DISTANCE_FILTER or ENABLE_TIME_FILTER), MAE(predictions, dataSet.businessModels)))
if ENABLE_DISTANCE_FILTER: print( "-")
else: print( "-")
if ENABLE_TIME_FILTER: print("-")
else: print("-")










