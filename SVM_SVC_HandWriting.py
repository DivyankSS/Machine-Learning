import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
#the digits datasets
digits=datasets.load_digits()
print ("digits",digits.target)
images_and_labels=list(zip(digits.images,digits.target))
print ("len(images_and_labels)",len(images_and_labels))

for index,(image,label) in enumerate(images_and_labels[:6]):
    print ("index: ",index,"images : ",image,"label : ",label)
    plt.subplot(2,6,index+1)
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Training %i' % label)
#plt.show()
#to apply a classifier on this data, we need to flatten the image, to
# turn the data in a (sample,feature) matrix:
n_sample=len(digits.images)
print ("a_samples :", n_sample)
imagedata= digits.images.reshape((n_sample,-1))

print ("After Reshaped : len(data[o]",len(imagedata))

#create classifier : asupport vector classifier
classifier=svm.SVC(gamma=0.001)

#we learn the digits on thee first half of the digits
classifier.fit(imagedata[:n_sample//2],digits.target[:n_sample//2])

#now predict the value of the digit on the second half
expected=digits.target[n_sample//2:]
predicted=classifier.predict(imagedata[n_sample//2:])

print("Classification report for classifier %s : \n%s\n"
      %(classifier,metrics.classification_report(expected,predicted)))
images_predicition=list(zip(digits.images[n_sample//2:],predicted))
for index,(image,prediction) in enumerate(images_predicition[:6]):
    plt.subplot(2, 6, index + 7)
    plt.axis('on')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('predicted %i' % prediction)
plt.show()


#================================================================================
from scipy.misc import imread,imresize,bytescale
img=imread("sev.jpg")
img=imresize(img,(8,8))
img=img.astype(digits.images.dtype)
img=bytescale(img,high=8.0,low=0)

print("img : ",img)
x_testdata=[]

for c in img:
    for r in c:
        x_testdata.append(sum(r)/3.0)
print("x_testdata: ",x_testdata)
x_testdata=[x_testdata]
print("len(testdata)",len(x_testdata))
print("machine output",classifier.predict(x_testdata))


