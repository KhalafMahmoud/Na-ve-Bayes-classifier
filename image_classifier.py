from PIL import Image
import statistics 
import math
import pandas as pd
import matplotlib.pyplot as plt

#Get standard deviation
def stdev(numbers):
    avg = statistics.mean(numbers)
    variance = statistics.variance(numbers)
    if math.sqrt(variance) == 0.0:
        return .1
    else: 
        return math.sqrt(variance)
#----------------------
#Calculate the mean and the standard deviation for each pixel of the seven training images
def summarize(dataset):
    summaries = []
    for i in range(0, len(dataset)):
        s_class = []
        for j in range(0, len(dataset[i])):
            mean_stdev = statistics.mean(dataset[i][j]), stdev(dataset[i][j])
            s_class.append(mean_stdev)
        summaries.append(s_class)
    return summaries
#----------------------
#Calculate probability given x, mean, standard deviation values
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x - mean, 2)/(2 * math.pow(stdev, 2))))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
#----------------------
#Get the propability for each pixel of the test image and then multiplying all these probablities to get probabillity for each class used
def calculateClassProbabilities(summaries, inputVector):
    probabilities = 1
    for i in range(0, len(summaries)):
        mean, stdev = summaries[i]
        x = inputVector[i]
        if calculateProbability(x, mean, stdev) < 0.001:
            probabilities *= 0.1
        else:
            probabilities *= calculateProbability(x, mean, stdev)
        #print(calculateProbability(x, mean, stdev))
    return probabilities
#----------------------
#Get all class probabilities and returnn the class with the greatest probability
def predict(summaries, inputVector):
    probabilities = []
    for summary in summaries:
        probability = calculateClassProbabilities(summary, inputVector)
        probabilities.append(probability)
    bestClass = None
    bestProb = -1
    for probability in probabilities:
        classValue = probabilities.index(probability)
        if bestClass is None or probability > bestProb:
            bestProb = probability
            bestClass = classValue
    return bestClass
#----------------------
#Get predictions for all test images
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        for j in range(0, 2):
            test_image = []
            for t in testSet[i]:
                test_image.append(t[j])
            result = predict(summaries, test_image)
            predictions.append(result)
    return predictions
#----------------------
def main():
    #Read all pixels from train file
    dataset = []
    for i in range(1, 27):
        pixels = [[]]
        for t in range(1, 144):
            pixels.append([])
        for j in range(1, 8):
            filename = 'Problem 2 Dataset/Train/A1' + str(chr(96 + i)) + str(j) + '.jpg'
            image = Image.open(filename)
            size = width, height = image.size
            for w in range(0, width):
                for h in range(0, height):
                    coordinate = x, y = w, h
                    pixels[w * 12 + h].append(image.getpixel(coordinate)/255)
                    #print(image.getpixel(coordinate))
            del image
        dataset.append(pixels)
    #Read all pixels from test file
    data_test = []
    for i in range(1, 27):
        pixels = [[]]
        for t in range(1, 144):
            pixels.append([])
        for j in range(8, 10):
            filename = 'Problem 2 Dataset/Test/A1' + str(chr(96 + i)) + str(j) + '.jpg'
            image = Image.open(filename)
            size = width, height = image.size
            for w in range(0, width):
                for h in range(0, height):
                    coordinate = x, y = w, h
                    pixels[w * 12 + h].append(image.getpixel(coordinate)/255)
            del image
        data_test.append(pixels)
    #Get summaries of the dataset
    summaries = summarize(dataset)
    #Get predications of our data test
    predictions = getPredictions(summaries, data_test)
    #print(predictions)
    #Calculate the correct_preness of the predictions
    correct_pre = []
    for i in range(0, 26):
        counter = 0
        for j in range(0, 2):
            if i == predictions[i*2+j]:
                counter += 1
        correct_pre.append(counter)
    #print(correct_pre)
    #cor = 0
    #for i in correct_pre: cor += i
    #print((cor/52) * 100)
    data = {'a':correct_pre[0], 'b':correct_pre[1], 'c':correct_pre[2], 'd':correct_pre[3], 'e':correct_pre[4], 'f':correct_pre[5], 
            'g':correct_pre[6], 'h':correct_pre[7], 'i':correct_pre[8], 'j':correct_pre[9], 'k':correct_pre[10], 'l':correct_pre[11], 
            'm':correct_pre[12], 'n':correct_pre[13], 'o':correct_pre[14], 'p':correct_pre[15], 'q':correct_pre[16], 'r':correct_pre[17], 
            's':correct_pre[18], 't':correct_pre[19], 'u':correct_pre[20], 'v':correct_pre[21], 'w':correct_pre[22], 'x':correct_pre[23], 
            'y':correct_pre[24], 'z':correct_pre[25]}
    s2 = pd.Series(data)
    s2.plot.bar()
    plt.xlabel('Character')
    plt.ylabel('Correct prediction')
    yint = range(min(correct_pre), math.ceil(max(correct_pre))+1)
    plt.yticks(yint)
    plt.savefig('Accuracy.jpg')
    plt.show()
#----------------------
main()