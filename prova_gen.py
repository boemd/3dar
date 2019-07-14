from utils import create_generator

#train_dir = 'D:/tesisti/Boem/ar2/val_set/val_set'
train_dir = 'D:/tesisti/Boem/ar3/val'
gen, num_training_samples = create_generator(train_dir, 10, extA='1.jpg', extB='2.jpg', extL='ab.txt')#create_generator(train_dir, 10)

x0, y0 = gen.__next__()
gen.__iter__()
x1, y1 = gen.__next__()
gen.__iter__()
x2, y2 = gen.__next__()
a = 4

