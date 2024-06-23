# siamese_on_objects
A commonly  used neural network for identifying faces being applied to differentiate objects

The initial problem statement was retail planogramming. This is a small subset of the problem.
The problem that the code in this repository aims to solve is:
- to find out the percentage of [facings](https://www.monash.edu/business/marketing/marketing-dictionary/s/shelf-facings#:~:text=the%20number%20of%20units%20of,facings%20than%20low%2Dvolume%20categories.) a particular brand recieves on a shelf
- this is to be done keeping in mind that there might be new product releases and the model should not require frequent retraining
This is solved in the following way:
- The objects on the shelf are detected and cropped out using [YOLOv8](https://docs.ultralytics.com/) trained on the dataset [SKU110K](https://paperswithcode.com/dataset/sku110k)
- One shot learning through a Siamese Network is applied on the cropped images where the images are matched to a database of images of a certain brand to count the number of shelf facings of products of that brand

The main advantage of this approach over a traditional CNN is that just by updating the database to which images are being compared, new products can be added, products that are out of circulation can be deleted and the model need not be retrained frequently. 

This project is still incomplete, a training set specific to this problem is being constructed
